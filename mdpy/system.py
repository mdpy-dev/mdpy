from __future__ import annotations

import cupy as cp
import numpy as np
from mdpy import env
from mdpy.core.block_list import BlockList
from mdpy.core.topology import build_exclusion_map_gpu, permute_exclusion_pairs_gpu
from mdpy.core.pbc import compute_pbc_inv
from mdpy.core.gpu_context import GPUContext


class System:

    def __init__(
        self, topology, pbc_matrix, cutoff=12.0, skin=1.0, rebuild_check_interval=10
    ):
        self.topology = topology
        self.num_particles = topology.num_particles
        self.pbc_matrix = np.ascontiguousarray(pbc_matrix, dtype=env.NUMPY_FLOAT)
        self.pbc_inv = compute_pbc_inv(self.pbc_matrix)
        self.cutoff = cutoff

        self.gpu = GPUContext()
        self.gpu.initialize(topology, self.pbc_matrix.flatten())

        self.block_list = BlockList(
            cutoff, skin=skin, rebuild_check_interval=rebuild_check_interval
        )
        self.force_terms = []
        self.constraints = []

        self._positions_uploaded = False
        self._velocities_uploaded = False
        self._step_counter = 0
        self._d_cached_unique_i = None
        self._d_cached_unique_j = None
        self._d_cached_unique_scale = None

    def add_force_term(self, term):
        self.force_terms.append(term)
        self.gpu.allocate_energy_accumulator(len(self.force_terms))

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def apply_constraints(self, dt):
        d_pdb_to_sorted = self.block_list.d_pdb_to_sorted
        for constraint in self.constraints:
            constraint.apply(self.gpu, dt, d_pdb_to_sorted=d_pdb_to_sorted)

    def upload_positions(self, positions):
        self.gpu.upload_positions(positions)
        self._positions_uploaded = True

    def upload_velocities(self, velocities):
        self.gpu.upload_velocities(velocities)
        self._velocities_uploaded = True

    def _ensure_uploaded(self):
        if not self._positions_uploaded or not self._velocities_uploaded:
            raise RuntimeError(
                "Positions and/or velocities not uploaded to GPU. "
                "Call system.upload_positions() and system.upload_velocities() first."
            )

    def compute_forces(self):
        self._ensure_uploaded()
        self.gpu.zero_forces()
        for term_index, term in enumerate(self.force_terms):
            term.compute(self.gpu, self.block_list, compute_energy=False)

    def update_neighbor_list(self, sync_interval=10):
        self._ensure_uploaded()
        positions_soa = (
            self.gpu.d_positions_x,
            self.gpu.d_positions_y,
            self.gpu.d_positions_z,
        )
        needs_sync = self.block_list.check_rebuild_async(positions_soa)
        if needs_sync:
            cp.cuda.Stream.null.synchronize()
            self._do_rebuild(positions_soa)
            self._step_counter = 0
            return
        self._step_counter += 1
        if self._step_counter < sync_interval:
            return
        cp.cuda.Stream.null.synchronize()
        if int(self.block_list.d_rebuild_flag[0]) == 1:
            self._do_rebuild(positions_soa)
        self._step_counter = 0

    def _do_rebuild(self, positions_soa):
        self.block_list.rebuild(
            positions_soa,
            self.topology,
            self.pbc_matrix,
            self.pbc_inv,
        )
        self._permute_all_arrays()
        self.gpu.wrap_positions_with_prev_correction()
        self.block_list.build_block_pairs(self.topology, self.pbc_matrix)
        for term in self.force_terms:
            if hasattr(term, "bind_sorted"):
                term.bind_sorted(self.topology, self.block_list, self.gpu)

    def dump_energy(self):
        if self.gpu.d_energy_accumulator is None:
            return {}
        self.gpu.zero_forces()
        for term_index, term in enumerate(self.force_terms):
            self.gpu.zero_energy()
            term.compute(self.gpu, self.block_list, compute_energy=True)
            self.gpu.accumulate_energy(term_index)
        raw = cp.asnumpy(self.gpu.d_energy_accumulator)
        result = {}
        for term_index, term in enumerate(self.force_terms):
            value = float(raw[term_index])
            if value != 0.0:
                result[term.name] = value
        return result

    def dump_state(self):
        bl = self.block_list
        gpu = self.gpu
        if bl.d_sorted_to_pdb.size > 0 and bl.num_particles > 0:
            sorted_to_pdb = bl.d_sorted_to_pdb
            pos = np.stack([
                gpu.permute_from_sorted(sorted_to_pdb, gpu.d_positions_x).get(),
                gpu.permute_from_sorted(sorted_to_pdb, gpu.d_positions_y).get(),
                gpu.permute_from_sorted(sorted_to_pdb, gpu.d_positions_z).get(),
            ], axis=1)
            vel = np.stack([
                gpu.permute_from_sorted(sorted_to_pdb, gpu.d_velocities_x).get(),
                gpu.permute_from_sorted(sorted_to_pdb, gpu.d_velocities_y).get(),
                gpu.permute_from_sorted(sorted_to_pdb, gpu.d_velocities_z).get(),
            ], axis=1)
        else:
            pos = np.stack([
                gpu.d_positions_x.get(),
                gpu.d_positions_y.get(),
                gpu.d_positions_z.get(),
            ], axis=1)
            vel = np.stack([
                gpu.d_velocities_x.get(),
                gpu.d_velocities_y.get(),
                gpu.d_velocities_z.get(),
            ], axis=1)
        return pos, vel

    def dump_forces(self):
        bl = self.block_list
        gpu = self.gpu
        if bl.d_sorted_to_pdb.size > 0 and bl.num_particles > 0:
            sorted_to_pdb = bl.d_sorted_to_pdb
            return np.stack([
                gpu.permute_from_sorted(sorted_to_pdb, gpu.d_forces_x).get(),
                gpu.permute_from_sorted(sorted_to_pdb, gpu.d_forces_y).get(),
                gpu.permute_from_sorted(sorted_to_pdb, gpu.d_forces_z).get(),
            ], axis=1)
        return np.stack([
            gpu.d_forces_x.get(),
            gpu.d_forces_y.get(),
            gpu.d_forces_z.get(),
        ], axis=1)

    def minimize(self, minimizer, number_steps=100):
        self._ensure_uploaded()

        positions_soa = (
            self.gpu.d_positions_x,
            self.gpu.d_positions_y,
            self.gpu.d_positions_z,
        )
        if self.block_list.check_rebuild(positions_soa):
            self.block_list.rebuild(
                positions_soa,
                self.topology,
                self.pbc_matrix,
                self.pbc_inv,
            )
            self._permute_all_arrays()
            self.gpu.wrap_positions_with_prev_correction()
            self.block_list.build_block_pairs(self.topology, self.pbc_matrix)
        for term in self.force_terms:
            if hasattr(term, "bind_sorted"):
                term.bind_sorted(self.topology, self.block_list, self.gpu)
        self.compute_forces()
        for _ in range(number_steps):
            minimizer.step(self)

    def _permute_all_arrays(self):
        N = self.topology.num_particles
        gpu = self.gpu
        bl = self.block_list

        perm_gpu = bl.d_raw_order

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
            ("d_charges", gpu.d_charges),
        ]):
            setattr(gpu, name, new_arr)

        d_remap = bl.d_pdb_to_sorted

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

        bl.set_gpu_exclusion(d_excl_offset, d_excl_neighbors, d_excl_scale)

        for term in self.force_terms:
            if hasattr(term, "remap_indices_gpu"):
                term.remap_indices_gpu(d_remap)

        for constraint in self.constraints:
            constraint.remap_indices_gpu(d_remap)
