from __future__ import annotations

import cupy as cp
import numpy as np
from mdpy import env
from mdpy.core.particle_table import ParticleTable
from mdpy.core.tile_list import TileList
from mdpy.core.topology import build_exclusion_map_gpu
from mdpy.core.pbc import compute_pbc_inv
from mdpy.core.gpu_context import GPUContext


class System:

    def __init__(self, topology, pbc_matrix, cutoff=12.0, skin=1.0):
        self.topology = topology
        self.particles = ParticleTable(topology.num_particles)
        self.pbc_matrix = np.ascontiguousarray(pbc_matrix, dtype=env.NUMPY_FLOAT)
        self.pbc_inv = compute_pbc_inv(self.pbc_matrix)
        self.cutoff = cutoff

        self.gpu = GPUContext()
        self.gpu.initialize(topology, self.pbc_matrix.flatten())

        self.tile_list = TileList(cutoff, skin=skin)
        self.force_terms = []

        self._step_count = 0
        self._positions_uploaded = False
        self._velocities_uploaded = False
        self._profiling_enabled = False
        self._profile_data = {}
        self._pdb_to_current_sorted = None
        self._particle_types_pdb = topology.particle_types.copy()


    def add_force_term(self, term):
        self.force_terms.append(term)
        self.gpu.allocate_energy_accumulator(len(self.force_terms))

    def enable_profiling(self):
        self._profiling_enabled = True
        self._profile_data = {}
        for term in self.force_terms:
            self._profile_data[term.name] = []
        self._profile_data['integrator'] = []

    def disable_profiling(self):
        self._profiling_enabled = False

    def dump_profile(self):
        cp.cuda.Stream.null.synchronize()
        result = {}
        for key, pairs in self._profile_data.items():
            if pairs:
                times = [cp.cuda.get_elapsed_time(s, e) for s, e in pairs]
                result[key] = {
                    'total_ms': sum(times),
                    'avg_ms': sum(times) / len(times),
                    'count': len(times),
                }
        self._profile_data = {k: [] for k in self._profile_data}
        return result

    def compute_forces(self):
        self.gpu.zero_forces()
        self.tile_list.update_sorted_positions(
            self.gpu.d_positions_x,
            self.gpu.d_positions_y,
            self.gpu.d_positions_z)
        pbc_2d = self.pbc_matrix.reshape(3, 3)
        self.gpu.set_box_dims(
            abs(float(pbc_2d[0, 0])),
            abs(float(pbc_2d[1, 1])),
            abs(float(pbc_2d[2, 2]))
        )
        for term_index, term in enumerate(self.force_terms):
            if self._profiling_enabled:
                s = cp.cuda.Event()
                e = cp.cuda.Event()
                s.record()
            self.gpu.zero_energy()
            term.compute(self.gpu, self.tile_list)
            self.gpu.accumulate_energy(term_index)
            if self._profiling_enabled:
                e.record()
                self._profile_data[term.name].append((s, e))

    def dump_energy(self):
        if self.gpu.d_energy_accumulator is None:
            return {}
        raw = cp.asnumpy(self.gpu.d_energy_accumulator)
        result = {}
        for term_index, term in enumerate(self.force_terms):
            value = float(raw[term_index])
            if value != 0.0:
                result[term.name] = value
        return result

    def _permute_all_arrays(self, pdb_to_sorted_gpu, pdb_to_sorted_np):
        N = self.topology.num_particles
        tpb = 256
        grid = ((N + tpb - 1) // tpb,)
        gpu = self.gpu

        perm_gpu = self.tile_list.d_sorted_to_pdb

        for old_arr, name in [
            (gpu.d_positions_x, 'd_positions_x'),
            (gpu.d_positions_y, 'd_positions_y'),
            (gpu.d_positions_z, 'd_positions_z'),
            (gpu.d_velocities_x, 'd_velocities_x'),
            (gpu.d_velocities_y, 'd_velocities_y'),
            (gpu.d_velocities_z, 'd_velocities_z'),
            (gpu.d_forces_x, 'd_forces_x'),
            (gpu.d_forces_y, 'd_forces_y'),
            (gpu.d_forces_z, 'd_forces_z'),
            (gpu.d_prev_positions_x, 'd_prev_positions_x'),
            (gpu.d_prev_positions_y, 'd_prev_positions_y'),
            (gpu.d_prev_positions_z, 'd_prev_positions_z'),
            (gpu.d_masses, 'd_masses'),
        ]:
            new_arr = cp.empty_like(old_arr)
            self.tile_list._kernels['permute'](grid, (tpb,),
                (old_arr, perm_gpu, np.int32(N), new_arr))
            setattr(gpu, name, new_arr)

        if pdb_to_sorted_np is None:
            pdb_to_sorted_np = cp.asnumpy(pdb_to_sorted_gpu)

        if self._pdb_to_current_sorted is None:
            self._pdb_to_current_sorted = pdb_to_sorted_np.copy()
        else:
            self._pdb_to_current_sorted = pdb_to_sorted_np[self._pdb_to_current_sorted]

        perm_np = cp.asnumpy(perm_gpu)
        remap = np.empty(N, dtype=np.int32)
        remap[perm_np] = np.arange(N, dtype=np.int32)
        self.topology.remap_bonded_indices(remap)
        d_excl_offset, d_excl_neighbors, d_excl_scale = build_exclusion_map_gpu(self.topology, scale_14=1.0)
        self.tile_list.set_gpu_exclusion(d_excl_offset, d_excl_neighbors, d_excl_scale)

        for term in self.force_terms:
            if hasattr(term, 'remap_indices'):
                term.remap_indices(self.topology)

    def _sorted_to_pdb_np(self):
        inv = np.empty_like(self._pdb_to_current_sorted)
        inv[self._pdb_to_current_sorted] = np.arange(len(self._pdb_to_current_sorted), dtype=inv.dtype)
        return inv

    def step(self, integrator, number_steps=1):
        if not self._positions_uploaded:
            self.gpu.upload_positions(self.particles)
            self._positions_uploaded = True
        if not self._velocities_uploaded:
            self.gpu.upload_velocities(self.particles)
            self._velocities_uploaded = True

        prof = self._profiling_enabled
        for _ in range(number_steps):
            positions_soa = self.gpu.get_positions_2d()
            if self.tile_list.check_rebuild(positions_soa):
                pdb_to_sorted_gpu, pdb_to_sorted_np = self.tile_list.rebuild(
                    positions_soa, self.topology,
                    self.pbc_matrix, self.pbc_inv,
                )
                self._permute_all_arrays(pdb_to_sorted_gpu, pdb_to_sorted_np)
                self.tile_list.build_tiles(self.topology, self.pbc_matrix)
                for term in self.force_terms:
                    if hasattr(term, 'bind_sorted'):
                        s2p = self._sorted_to_pdb_np()
                        term.bind_sorted(
                            self.topology, self.tile_list, self.gpu,
                            sorted_to_pdb_np=s2p,
                            sorted_particle_types=self._particle_types_pdb[s2p],
                        )
            self.compute_forces()
            if prof:
                s = cp.cuda.Event()
                e = cp.cuda.Event()
                s.record()
            integrator.step(self.gpu)
            if prof:
                e.record()
                self._profile_data['integrator'].append((s, e))
            self._step_count += 1

    def minimize(self, minimizer, number_steps=100):
        if not self._positions_uploaded:
            self.gpu.upload_positions(self.particles)
            self._positions_uploaded = True
        if not self._velocities_uploaded:
            self.gpu.upload_velocities(self.particles)
            self._velocities_uploaded = True

        positions_soa = self.gpu.get_positions_2d()
        if self.tile_list.check_rebuild(positions_soa):
            pdb_to_sorted_gpu, pdb_to_sorted_np = self.tile_list.rebuild(
                positions_soa, self.topology,
                self.pbc_matrix, self.pbc_inv,
            )
            self._permute_all_arrays(pdb_to_sorted_gpu, pdb_to_sorted_np)
            self.tile_list.build_tiles(self.topology, self.pbc_matrix)
        for term in self.force_terms:
            if hasattr(term, 'bind_sorted'):
                s2p = self._sorted_to_pdb_np()
                term.bind_sorted(
                    self.topology, self.tile_list, self.gpu,
                    sorted_to_pdb_np=s2p,
                    sorted_particle_types=self._particle_types_pdb[s2p],
                )
        self.compute_forces()
        for _ in range(number_steps):
            minimizer.step(self)
        self.gpu.download_positions(self.particles)

    def dump_state(self):
        if self.tile_list.d_sorted_to_pdb.size > 0 and self.tile_list.num_particles > 0:
            sorted_to_pdb_gpu = cp.asarray(self._sorted_to_pdb_np())
            pdb_x = self.tile_list.permute_from_sorted(
                sorted_to_pdb_gpu, self.gpu.d_positions_x)
            pdb_y = self.tile_list.permute_from_sorted(
                sorted_to_pdb_gpu, self.gpu.d_positions_y)
            pdb_z = self.tile_list.permute_from_sorted(
                sorted_to_pdb_gpu, self.gpu.d_positions_z)
            pos = np.stack([pdb_x.get(), pdb_y.get(), pdb_z.get()], axis=1)
            self.particles.positions[:] = pos

            pdb_vx = self.tile_list.permute_from_sorted(
                sorted_to_pdb_gpu, self.gpu.d_velocities_x)
            pdb_vy = self.tile_list.permute_from_sorted(
                sorted_to_pdb_gpu, self.gpu.d_velocities_y)
            pdb_vz = self.tile_list.permute_from_sorted(
                sorted_to_pdb_gpu, self.gpu.d_velocities_z)
            vel = np.stack([pdb_vx.get(), pdb_vy.get(), pdb_vz.get()], axis=1)
            self.particles.velocities[:] = vel
        else:
            self.gpu.download_positions(self.particles)
            self.gpu.download_velocities(self.particles)
        return self.particles.positions.copy(), self.particles.velocities.copy()

    @property
    def step_count(self):
        return self._step_count
