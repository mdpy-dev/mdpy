from __future__ import annotations

import cupy as cp
import numpy as np
from mdpy import env
from mdpy.core.block_list import BlockList
from mdpy.core.topology import build_exclusion_map_gpu, permute_exclusion_pairs_gpu
from mdpy.core.gpu_context import GPUContext

_INVERT_PERMUTATION_KERNEL_SRC = r"""
extern "C" __global__
void invert_permutation_kernel(
    int* out, const int* __restrict__ perm, int N
) {
    /* out[perm[i]] = i  =>  out is the inverse of perm */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[perm[i]] = i;
}
"""

_invert_permutation_kernel = None


def _get_invert_permutation_kernel():
    global _invert_permutation_kernel
    if _invert_permutation_kernel is None:
        _invert_permutation_kernel = cp.RawKernel(
            _INVERT_PERMUTATION_KERNEL_SRC, "invert_permutation_kernel"
        )
    return _invert_permutation_kernel


class System:

    def __init__(self, topology):
        self.topology = topology
        self.num_particles = topology.num_particles
        self.gpu = GPUContext()
        self.gpu.initialize(topology, np.eye(3, dtype=np.float32))

        self._cutoff = None
        self._skin = 1.0
        self._rebuild_check_interval = 10
        self._block_list = None
        self._pbc_set = False

        self.force_terms = []
        self._primary_force_terms = []
        self._pme_force_terms = []
        self._pme_stream = None
        self._ev_zero_forces = None
        self._ev_pme_done = None
        self.constraints = []

        self._positions_uploaded = False
        self._velocities_uploaded = False
        self._step_counter = 0
        self._d_cached_unique_i = None
        self._d_cached_unique_j = None
        self._d_cached_unique_scale = None
        self._excl_pool_A = {}
        self._excl_pool_B = {}
        self._excl_flip = False

    def _ensure_pme_stream(self):
        if self._pme_stream is None:
            self._pme_stream = cp.cuda.Stream(non_blocking=True)
            # disable_timing: these are hot-path per-step ordering events;
            # timing-enabled events add ~1-2 us of sync overhead each.
            self._ev_zero_forces = cp.cuda.Event(disable_timing=True)
            self._ev_pme_done = cp.cuda.Event(disable_timing=True)

    def upload_pbc(self, pbc_matrix):
        self.gpu.upload_pbc(pbc_matrix)
        self._pbc_set = True

    @property
    def block_list(self):
        if self._block_list is None:
            raise RuntimeError(
                "Neighbor list not initialized. Call update_neighbor_list() first."
            )
        return self._block_list

    @property
    def cutoff(self):
        return self._cutoff

    def add_force_term(self, term, stream=None):
        if stream not in (None, 'pme'):
            raise ValueError(
                f"stream must be None or 'pme', got {stream!r}"
            )
        self.force_terms.append(term)
        if stream == 'pme':
            self._ensure_pme_stream()
            self._pme_force_terms.append(term)
        else:
            self._primary_force_terms.append(term)
        self.gpu.allocate_energy_accumulator(len(self.force_terms))
        term_cutoff = getattr(term, '_cutoff', None)
        if term_cutoff is not None:
            if self._cutoff is None:
                self._cutoff = term_cutoff
            else:
                self._cutoff = max(self._cutoff, term_cutoff)
            if self._block_list is not None:
                self._block_list.set_cutoff(self._cutoff)

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def apply_constraints(self, time_step):
        d_pdb_to_sorted = self._block_list.d_pdb_to_sorted
        for constraint in self.constraints:
            constraint.apply(self.gpu, time_step, d_pdb_to_sorted=d_pdb_to_sorted)

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

        if not self._pme_force_terms:
            for term in self._primary_force_terms:
                term.compute(self.gpu, self._block_list, compute_energy=False)
            return

        # PME runs on a non-blocking stream concurrent with primary terms.
        # zero_forces just ran on the null stream; record an event so the PME
        # stream does not atomicAdd into d_forces before the zeroing completes.
        self._ev_zero_forces.record()

        # Launch PME pipeline on its stream, gated on zero_forces.
        self._pme_stream.wait_event(self._ev_zero_forces)
        with self._pme_stream:
            for term in self._pme_force_terms:
                term.compute(self.gpu, self._block_list, compute_energy=False)
            # Record INSIDE the with-block so the event is recorded on the
            # pme stream, not the null stream (record() uses the current stream).
            self._ev_pme_done.record()

        # Primary terms run on the null stream, overlapping with PME.
        for term in self._primary_force_terms:
            term.compute(self.gpu, self._block_list, compute_energy=False)

        # Make the null stream wait for PME to finish writing forces before
        # compute_forces returns, so the next null-stream op (integrator) sees
        # fully-accumulated forces.
        cp.cuda.Stream.null.wait_event(self._ev_pme_done)

    def update_neighbor_list(self, sync_interval=10, force_rebuild=False):
        self._ensure_uploaded()
        if not self._pbc_set:
            raise RuntimeError("PBC not set. Call upload_pbc() first.")

        if self._block_list is None:
            if self._cutoff is None:
                raise RuntimeError(
                    "No cutoff available. Add a force term with cutoff first."
                )
            self._block_list = BlockList(
                self._cutoff, skin=self._skin,
                rebuild_check_interval=self._rebuild_check_interval,
            )
            force_rebuild = True

        positions_soa = (
            self.gpu.d_positions_x,
            self.gpu.d_positions_y,
            self.gpu.d_positions_z,
        )

        if force_rebuild:
            cp.cuda.Stream.null.synchronize()
            self._do_rebuild(positions_soa, force=True)
            self._step_counter = 0
            return

        self._block_list.check_rebuild_async(positions_soa)
        self._step_counter += 1
        if self._step_counter < sync_interval:
            return

        self._do_rebuild(positions_soa, force=False)
        self._step_counter = 0

    def _do_rebuild(self, positions_soa, *, force=False):
        if not force:
            flag_val = int(self._block_list.d_rebuild_flag[0])
            if flag_val == 0:
                return
        self._block_list.rebuild(
            positions_soa,
            self.topology,
            self.gpu,
            force=force,
        )
        self._permute_all_arrays()
        self.gpu.wrap_positions_with_prev_correction()
        self._block_list.capture_snapshot((
            self.gpu.d_positions_x,
            self.gpu.d_positions_y,
            self.gpu.d_positions_z,
        ))
        self._block_list.build_block_pairs(self.topology, self.gpu)
        for term in self.force_terms:
            if hasattr(term, "bind_sorted"):
                term.bind_sorted(self.topology, self._block_list, self.gpu)
        self._block_list.d_rebuild_flag[0] = 0

    def dump_energy(self):
        if self.gpu.d_energy_accumulator is None:
            return {}
        self.gpu.zero_forces()
        for term_index, term in enumerate(self.force_terms):
            self.gpu.zero_energy()
            term.compute(self.gpu, self._block_list, compute_energy=True)
            self.gpu.accumulate_energy(term_index)
        raw = cp.asnumpy(self.gpu.d_energy_accumulator)
        result = {}
        for term_index, term in enumerate(self.force_terms):
            value = float(raw[term_index])
            if value != 0.0:
                result[term.name] = value
        return result

    def dump_state(self):
        bl = self._block_list
        gpu = self.gpu
        if bl is not None and bl.d_sorted_to_pdb.size > 0 and bl.num_particles > 0:
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
        bl = self._block_list
        gpu = self.gpu
        if bl is not None and bl.d_sorted_to_pdb.size > 0 and bl.num_particles > 0:
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

    def _permute_all_arrays(self):
        N = self.topology.num_particles
        gpu = self.gpu
        bl = self._block_list

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
            d_inverse_perm = bl._pool_get("inverse_perm", N, env.NUMPY_INT)
            kernel = _get_invert_permutation_kernel()
            grid = ((N + 255) // 256,)
            kernel(grid, (256,), (d_inverse_perm, perm_gpu, np.int32(N)))
            pool = self._excl_pool_B if self._excl_flip else self._excl_pool_A
            self._excl_flip = not self._excl_flip
            result = permute_exclusion_pairs_gpu(
                self._d_cached_unique_i,
                self._d_cached_unique_j,
                self._d_cached_unique_scale,
                d_inverse_perm,
                N,
                pool,
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
