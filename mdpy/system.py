from __future__ import annotations

import cupy as cp
import numpy as np
from mdpy import env
from mdpy.core.particle_table import ParticleTable
from mdpy.core.tile_list import TileList
from mdpy.core.topology import build_exclusion_map_gpu, permute_exclusion_pairs_gpu
from mdpy.core.pbc import compute_pbc_inv
from mdpy.core.gpu_context import GPUContext


class System:

    def __init__(
        self, topology, pbc_matrix, cutoff=12.0, skin=1.0, rebuild_check_interval=10
    ):
        self.topology = topology
        self.particles = ParticleTable(topology.num_particles)
        self.pbc_matrix = np.ascontiguousarray(pbc_matrix, dtype=env.NUMPY_FLOAT)
        self.pbc_inv = compute_pbc_inv(self.pbc_matrix)
        self.cutoff = cutoff

        self.gpu = GPUContext()
        self.gpu.initialize(topology, self.pbc_matrix.flatten())

        self.tile_list = TileList(
            cutoff, skin=skin, rebuild_check_interval=rebuild_check_interval
        )
        self.force_terms = []

        self._step_count = 0
        self._positions_uploaded = False
        self._velocities_uploaded = False
        self._compute_energy = False
        self._profiling_enabled = False
        self._profile_data = {}
        self._d_cached_unique_i = None
        self._d_cached_unique_j = None
        self._d_cached_unique_scale = None
        self._steps_since_check = 0
        self._step_graph = None
        self._graph_stream = cp.cuda.Stream(non_blocking=True)
        self._graph_needs_capture = True
        self._cached_integrator_id = None

    def add_force_term(self, term):
        self.force_terms.append(term)
        self.gpu.allocate_energy_accumulator(len(self.force_terms))

    def enable_profiling(self):
        self._profiling_enabled = True
        self._profile_data = {}
        for term in self.force_terms:
            self._profile_data[term.name] = []
        self._profile_data["integrator"] = []

    def disable_profiling(self):
        self._profiling_enabled = False

    def dump_profile(self):
        cp.cuda.Stream.null.synchronize()
        result = {}
        for key, pairs in self._profile_data.items():
            if pairs:
                times = [cp.cuda.get_elapsed_time(s, e) for s, e in pairs]
                result[key] = {
                    "total_ms": sum(times),
                    "avg_ms": sum(times) / len(times),
                    "count": len(times),
                }
        self._profile_data = {k: [] for k in self._profile_data}
        return result

    def compute_forces(self):
        self.gpu.zero_forces()
        for term_index, term in enumerate(self.force_terms):
            if self._profiling_enabled:
                s = cp.cuda.Event()
                e = cp.cuda.Event()
                s.record()
            if self._compute_energy == False:
                term.compute(self.gpu, self.tile_list)
            else:
                self.gpu.zero_energy()
                term.compute(self.gpu, self.tile_list)
                self.gpu.accumulate_energy(term_index)
            if self._profiling_enabled:
                e.record()
                self._profile_data[term.name].append((s, e))

    def _emit_step_kernels(self, integrator):
        positions_soa = (
            self.gpu.d_wrapped_positions_x,
            self.gpu.d_wrapped_positions_y,
            self.gpu.d_wrapped_positions_z,
        )
        self.tile_list.check_rebuild_async(positions_soa)
        self.compute_forces()
        integrator.step(self.gpu)
        self.gpu.refresh_wrapped_positions()

    def _capture_step_graph(self, integrator):
        if self._step_graph is not None:
            del self._step_graph
            self._step_graph = None

        s = self._graph_stream
        with s:
            s.begin_capture()
            self._emit_step_kernels(integrator)
            self._step_graph = s.end_capture()

        self._cached_integrator_id = id(integrator)
        self._graph_needs_capture = False

    def dump_energy(self):
        if self.gpu.d_energy_accumulator is None:
            return {}
        self._compute_energy = True
        self.compute_forces()
        self._compute_energy = False
        raw = cp.asnumpy(self.gpu.d_energy_accumulator)
        result = {}
        for term_index, term in enumerate(self.force_terms):
            value = float(raw[term_index])
            if value != 0.0:
                result[term.name] = value
        return result

    def _permute_all_arrays(self):
        N = self.topology.num_particles
        gpu = self.gpu
        tl = self.tile_list

        perm_gpu = tl.d_raw_order

        for name, new_arr in gpu.permute_state_arrays(perm_gpu, [
            ("d_positions_x", gpu.d_positions_x),
            ("d_positions_y", gpu.d_positions_y),
            ("d_positions_z", gpu.d_positions_z),
            ("d_velocities_x", gpu.d_velocities_x),
            ("d_velocities_y", gpu.d_velocities_y),
            ("d_velocities_z", gpu.d_velocities_z),
            ("d_forces_x", gpu.d_forces_x),
            ("d_forces_y", gpu.d_forces_y),
            ("d_forces_z", gpu.d_forces_z),
            ("d_prev_positions_x", gpu.d_prev_positions_x),
            ("d_prev_positions_y", gpu.d_prev_positions_y),
            ("d_prev_positions_z", gpu.d_prev_positions_z),
            ("d_masses", gpu.d_masses),
            ("d_wrapped_positions_x", gpu.d_wrapped_positions_x),
            ("d_wrapped_positions_y", gpu.d_wrapped_positions_y),
            ("d_wrapped_positions_z", gpu.d_wrapped_positions_z),
        ]):
            setattr(gpu, name, new_arr)

        d_remap = tl.d_pdb_to_sorted

        if self._d_cached_unique_i is not None:
            d_composed_perm = cp.empty(N, dtype=cp.int32)
            d_composed_perm[perm_gpu] = cp.arange(N, dtype=cp.int32)
            result = permute_exclusion_pairs_gpu(
                self._d_cached_unique_i,
                self._d_cached_unique_j,
                self._d_cached_unique_scale,
                d_composed_perm,
                N,
            )
            d_excl_offset = result[0]
            d_excl_neighbors = result[1]
            d_excl_scale = result[2]
            self._d_cached_unique_i = result[3]
            self._d_cached_unique_j = result[4]
            self._d_cached_unique_scale = result[5]
        else:
            remap_np = cp.asnumpy(d_remap)
            for field in (
                "bond_indices",
                "angle_indices",
                "dihedral_indices",
                "improper_indices",
            ):
                indices = getattr(self.topology, field, None)
                if indices is not None and len(indices) > 0:
                    setattr(self.topology, field, remap_np[indices])
            d_excl_offset, d_excl_neighbors, d_excl_scale, d_unique_i = (
                build_exclusion_map_gpu(self.topology, scale_14=1.0)
            )
            self._d_cached_unique_i = d_unique_i
            self._d_cached_unique_j = d_excl_neighbors
            self._d_cached_unique_scale = d_excl_scale

        tl.set_gpu_exclusion(d_excl_offset, d_excl_neighbors, d_excl_scale)

        for term in self.force_terms:
            if hasattr(term, "remap_indices_gpu"):
                term.remap_indices_gpu(d_remap)

    def step(self, integrator, number_steps=1):
        if not self._positions_uploaded:
            self.gpu.upload_positions(self.particles)
            self._positions_uploaded = True
        if not self._velocities_uploaded:
            self.gpu.upload_velocities(self.particles)
            self._velocities_uploaded = True
        self.gpu.refresh_wrapped_positions()

        if id(integrator) != self._cached_integrator_id:
            self._graph_needs_capture = True

        if self._graph_needs_capture:
            positions_soa = (
                self.gpu.d_wrapped_positions_x,
                self.gpu.d_wrapped_positions_y,
                self.gpu.d_wrapped_positions_z,
            )
            if self.tile_list.check_rebuild(positions_soa):
                self._do_full_rebuild(positions_soa)
            self._capture_step_graph(integrator)

        prof = self._profiling_enabled
        for _ in range(number_steps):
            if self._step_graph is not None:
                self._step_graph.launch(self._graph_stream)
            else:
                self._emit_step_kernels(integrator)

            if prof:
                s = cp.cuda.Event()
                e = cp.cuda.Event()
                s.record(self._graph_stream)
                e.record(self._graph_stream)
                self._profile_data["integrator"].append((s, e))

            self._step_count += 1
            self._steps_since_check += 1
            if self._steps_since_check >= self.tile_list.rebuild_check_interval:
                self._graph_stream.synchronize()
                if int(self.tile_list.d_rebuild_flag[0]) == 1:
                    positions_soa = (
                        self.gpu.d_wrapped_positions_x,
                        self.gpu.d_wrapped_positions_y,
                        self.gpu.d_wrapped_positions_z,
                    )
                    self._do_full_rebuild(positions_soa)
                    self._capture_step_graph(integrator)
                self._steps_since_check = 0

    def _do_full_rebuild(self, positions_soa):
        self.tile_list.rebuild(
            positions_soa,
            self.topology,
            self.pbc_matrix,
            self.pbc_inv,
        )
        self._permute_all_arrays()
        self.tile_list.build_tiles(self.topology, self.pbc_matrix)
        for term in self.force_terms:
            if hasattr(term, "bind_sorted"):
                term.bind_sorted(self.topology, self.tile_list, self.gpu)

    def minimize(self, minimizer, number_steps=100):
        if not self._positions_uploaded:
            self.gpu.upload_positions(self.particles)
            self._positions_uploaded = True
        if not self._velocities_uploaded:
            self.gpu.upload_velocities(self.particles)
            self._velocities_uploaded = True
        self.gpu.refresh_wrapped_positions()

        positions_soa = (
            self.gpu.d_wrapped_positions_x,
            self.gpu.d_wrapped_positions_y,
            self.gpu.d_wrapped_positions_z,
        )
        if self.tile_list.check_rebuild(positions_soa):
            self.tile_list.rebuild(
                positions_soa,
                self.topology,
                self.pbc_matrix,
                self.pbc_inv,
            )
            self._permute_all_arrays()
            self.tile_list.build_tiles(self.topology, self.pbc_matrix)
        for term in self.force_terms:
            if hasattr(term, "bind_sorted"):
                term.bind_sorted(self.topology, self.tile_list, self.gpu)
        self.compute_forces()
        for _ in range(number_steps):
            minimizer.step(self)
            self.gpu.refresh_wrapped_positions()
        self.gpu.download_positions(self.particles)

    def dump_state(self):
        tl = self.tile_list
        if tl.d_sorted_to_pdb.size > 0 and tl.num_particles > 0:
            sorted_to_pdb = tl.d_sorted_to_pdb
            pdb_x = self.gpu.permute_from_sorted(sorted_to_pdb, self.gpu.d_positions_x)
            pdb_y = self.gpu.permute_from_sorted(sorted_to_pdb, self.gpu.d_positions_y)
            pdb_z = self.gpu.permute_from_sorted(sorted_to_pdb, self.gpu.d_positions_z)
            pos = np.stack([pdb_x.get(), pdb_y.get(), pdb_z.get()], axis=1)
            self.particles.positions[:] = pos

            pdb_vx = self.gpu.permute_from_sorted(sorted_to_pdb, self.gpu.d_velocities_x)
            pdb_vy = self.gpu.permute_from_sorted(sorted_to_pdb, self.gpu.d_velocities_y)
            pdb_vz = self.gpu.permute_from_sorted(sorted_to_pdb, self.gpu.d_velocities_z)
            vel = np.stack([pdb_vx.get(), pdb_vy.get(), pdb_vz.get()], axis=1)
            self.particles.velocities[:] = vel
        else:
            self.gpu.download_positions(self.particles)
            self.gpu.download_velocities(self.particles)
        return self.particles.positions.copy(), self.particles.velocities.copy()

    @property
    def step_count(self):
        return self._step_count
