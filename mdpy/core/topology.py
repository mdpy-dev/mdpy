from __future__ import annotations

import cupy as cp
import numpy as np
from mdpy import env



class Topology:

    __slots__ = [
        'num_particles',
        'bond_indices', 'num_bonds',
        'angle_indices', 'num_angles',
        'dihedral_indices', 'num_dihedrals',
        'improper_indices', 'num_impropers',
        '_exclusion_dirty', '_exclusion_pairs', '_exclusion_csr',
        '_exclusion_reverse_csr',
    ]

    def __init__(self, builder: Builder | None = None):
        if builder is None:
            self._init_legacy()
            return
        self.num_particles = builder._num_particles

        if builder._bonds:
            self.bond_indices = np.array(
                [b[:2] for b in builder._bonds], dtype=env.NUMPY_INT
            )
        else:
            self.bond_indices = np.empty((0, 2), dtype=env.NUMPY_INT)
        self.num_bonds = self.bond_indices.shape[0]

        if builder._angles:
            self.angle_indices = np.array(
                [a[:3] for a in builder._angles], dtype=env.NUMPY_INT
            )
        else:
            self.angle_indices = np.empty((0, 3), dtype=env.NUMPY_INT)
        self.num_angles = self.angle_indices.shape[0]

        if builder._dihedrals:
            self.dihedral_indices = np.array(
                [d[:4] for d in builder._dihedrals], dtype=env.NUMPY_INT
            )
        else:
            self.dihedral_indices = np.empty((0, 4), dtype=env.NUMPY_INT)
        self.num_dihedrals = self.dihedral_indices.shape[0]

        if builder._impropers:
            self.improper_indices = np.array(
                [im[:4] for im in builder._impropers], dtype=env.NUMPY_INT
            )
        else:
            self.improper_indices = np.empty((0, 4), dtype=env.NUMPY_INT)
        self.num_impropers = self.improper_indices.shape[0]

        self._init_exclusion_cache()

    def _init_legacy(self):
        self.num_particles = 0
        self.bond_indices = np.empty((0, 2), dtype=env.NUMPY_INT)
        self.num_bonds = 0
        self.angle_indices = np.empty((0, 3), dtype=env.NUMPY_INT)
        self.num_angles = 0
        self.dihedral_indices = np.empty((0, 4), dtype=env.NUMPY_INT)
        self.num_dihedrals = 0
        self.improper_indices = np.empty((0, 4), dtype=env.NUMPY_INT)
        self.num_impropers = 0
        self._init_exclusion_cache()

    def _init_exclusion_cache(self):
        """Lazy GPU exclusion state. Built on first read of any exclusion_*
        property, or rebuilt after the bond graph is marked dirty."""
        self._exclusion_dirty = True
        self._exclusion_pairs = None       # (d_i, d_j) unique pairs
        self._exclusion_csr = None         # (offset, neighbors)
        self._exclusion_reverse_csr = None  # (rev_offset, rev_neighbors)

    def _derive_exclusion_state(self):
        """Build unique pairs + forward CSR + reverse CSR from the bond graph.

        All atom-indexed (PDB order), independent of any spatial block layout.
        """
        N = self.num_particles
        kernels = _get_gpu_kernels()
        threads_per_block = 256

        # --- raw bond-graph exclusion pairs (CPU walk) ---
        pair_i_np, pair_j_np, total_pairs, _, _, _ = \
            _build_bond_graph_exclusion_pairs(self.bond_indices, N)

        if total_pairs == 0:
            zi = cp.empty(0, dtype=env.NUMPY_INT)
            self._exclusion_pairs = (zi, zi)
            empty_off = cp.zeros(N + 1, dtype=env.NUMPY_INT)
            self._exclusion_csr = (empty_off, zi)
            self._exclusion_reverse_csr = (empty_off.copy(), zi)
            self._exclusion_dirty = False
            return

        # --- sort + dedup + bidirectional -> unique pairs ---
        d_pair_i = cp.asarray(pair_i_np)
        d_pair_j = cp.asarray(pair_j_np)

        sort_key_stride = np.int64(2_000_000_000)
        sort_key = d_pair_i.astype(cp.int64) * sort_key_stride + d_pair_j.astype(cp.int64)
        order = cp.argsort(sort_key)
        d_pair_i, d_pair_j = d_pair_i[order], d_pair_j[order]

        d_flags = cp.zeros(total_pairs, dtype=env.NUMPY_INT)
        grid_p = ((total_pairs + threads_per_block - 1) // threads_per_block,)
        kernels['parallel_dedup'](grid_p, (threads_per_block,),
            (d_pair_i, d_pair_j, np.int32(total_pairs), d_flags))
        scatter_idx = cp.cumsum(d_flags) - 1
        uniq_count = int(scatter_idx[total_pairs - 1]) + 1

        d_u_i = cp.full(uniq_count, -1, dtype=env.NUMPY_INT)
        d_u_j = cp.full(uniq_count, -1, dtype=env.NUMPY_INT)
        d_u_i[scatter_idx] = d_pair_i
        d_u_j[scatter_idx] = d_pair_j

        # bidirectional
        d_bi_i = cp.concatenate([d_u_i, d_u_j])
        d_bi_j = cp.concatenate([d_u_j, d_u_i])
        bi_count = d_bi_i.shape[0]
        bi_key = d_bi_i.astype(cp.int64) * sort_key_stride + d_bi_j.astype(cp.int64)
        bi_order = cp.argsort(bi_key)
        d_bi_i, d_bi_j = d_bi_i[bi_order], d_bi_j[bi_order]

        d_bi_flags = cp.zeros(bi_count, dtype=env.NUMPY_INT)
        grid_b = ((bi_count + threads_per_block - 1) // threads_per_block,)
        kernels['parallel_dedup'](grid_b, (threads_per_block,),
            (d_bi_i, d_bi_j, np.int32(bi_count), d_bi_flags))
        bi_scatter = cp.cumsum(d_bi_flags) - 1
        bi_uniq = int(bi_scatter[bi_count - 1]) + 1
        d_unique_i = cp.full(bi_uniq, -1, dtype=env.NUMPY_INT)
        d_unique_j = cp.full(bi_uniq, -1, dtype=env.NUMPY_INT)
        d_unique_i[bi_scatter] = d_bi_i
        d_unique_j[bi_scatter] = d_bi_j

        self._exclusion_pairs = (d_unique_i, d_unique_j)

        # --- forward CSR ---
        num_pairs = bi_uniq
        d_count = cp.zeros(N + 1, dtype=env.NUMPY_INT)
        grid_c = ((num_pairs + threads_per_block - 1) // threads_per_block,)
        kernels['count_row'](grid_c, (threads_per_block,),
            (d_unique_i, np.int32(num_pairs), d_count))
        d_offset = cp.empty(N + 1, dtype=env.NUMPY_INT)
        cp.cumsum(d_count, dtype=cp.int32, out=d_offset)
        d_neighbors = cp.empty(num_pairs, dtype=env.NUMPY_INT)
        d_fwd_write_cursor = cp.empty(N + 1, dtype=env.NUMPY_INT)
        d_fwd_write_cursor[:] = d_offset
        kernels['scatter_pairs'](grid_c, (threads_per_block,),
            (d_unique_i, d_unique_j, d_offset, np.int32(num_pairs),
             d_neighbors, d_fwd_write_cursor))
        self._exclusion_csr = (d_offset, d_neighbors)

        # --- reverse CSR ---
        n1 = (N + threads_per_block - 1) // threads_per_block
        d_rev_offset = cp.zeros(N + 1, dtype=env.NUMPY_INT)
        kernels['rev_count']((n1,), (threads_per_block,),
            (d_offset, d_neighbors, np.int32(N), d_rev_offset))
        d_rev_offset = cp.cumsum(d_rev_offset, dtype=cp.int32).astype(env.NUMPY_INT)
        max_rev = num_pairs if num_pairs > 0 else int(d_rev_offset[-1])
        d_rev_neighbors = cp.empty(max_rev, dtype=env.NUMPY_INT)
        d_rev_write_cursor = d_rev_offset.copy()
        kernels['rev_fill']((n1,), (threads_per_block,),
            (d_offset, d_neighbors, d_rev_offset, np.int32(N),
             d_rev_neighbors, d_rev_write_cursor))
        self._exclusion_reverse_csr = (d_rev_offset, d_rev_neighbors)

        self._exclusion_dirty = False

    @property
    def exclusion_pairs(self):
        """Unique bidirectional exclusion pairs (d_i, d_j), GPU."""
        if self._exclusion_dirty:
            self._derive_exclusion_state()
        return self._exclusion_pairs

    @property
    def exclusion_csr(self):
        """Forward CSR (offset, neighbors), atom-indexed, GPU."""
        if self._exclusion_dirty:
            self._derive_exclusion_state()
        return self._exclusion_csr

    @property
    def exclusion_reverse_csr(self):
        """Reverse (transposed) CSR (rev_offset, rev_neighbors), GPU."""
        if self._exclusion_dirty:
            self._derive_exclusion_state()
        return self._exclusion_reverse_csr

    def invalidate_exclusions(self):
        """Mark the cached exclusion state stale. Call after mutating bonds.
        Recomputation is deferred to the next exclusion_* property read."""
        self._exclusion_dirty = True

    def __repr__(self) -> str:
        return (
            '<mdpy.core.Topology: %d particles, %d bonds, %d angles, '
            '%d dihedrals, %d impropers>'
            % (
                self.num_particles, self.num_bonds, self.num_angles,
                self.num_dihedrals, self.num_impropers,
            )
        )


_PARALLEL_DEDUP_KERNEL = r'''
extern "C" __global__
void parallel_dedup_kernel(
    const int* __restrict__ sorted_i,
    const int* __restrict__ sorted_j,
    const int total_pairs,
    int* __restrict__ flags
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        flags[0] = 1;
    } else if (tid < total_pairs) {
        flags[tid] = (sorted_i[tid] != sorted_i[tid - 1]
                      || sorted_j[tid] != sorted_j[tid - 1]) ? 1 : 0;
    }
}
'''

_COUNT_ROW_KERNEL = r"""
extern "C" __global__
void count_row_kernel(
    const int* __restrict__ pair_i,
    int num_pairs,
    int* __restrict__ count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_pairs) return;
    atomicAdd(&count[pair_i[tid] + 1], 1);
}
"""

_SCATTER_PAIRS_KERNEL = r"""
extern "C" __global__
void scatter_pairs_kernel(
    const int* __restrict__ pair_i,
    const int* __restrict__ pair_j,
    const int* __restrict__ offset,
    int num_pairs,
    int* __restrict__ neighbors_out,
    int* __restrict__ temp_offset
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_pairs) return;
    int i = pair_i[tid];
    int pos = atomicAdd(&temp_offset[i], 1);
    neighbors_out[pos] = pair_j[tid];
}
"""

_BUILD_REVERSE_COUNT_KERNEL = r"""
extern "C" __global__
void build_reverse_count_kernel(
    const int* __restrict__ exclusion_offset,
    const int* __restrict__ exclusion_neighbors,
    const int num_particles,
    int* __restrict__ reverse_offset
) {
    int atom_a = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom_a >= num_particles) return;
    int start = exclusion_offset[atom_a];
    int end = exclusion_offset[atom_a + 1];
    for (int k = start; k < end; k++) {
        int neighbor = exclusion_neighbors[k];
        atomicAdd(&reverse_offset[neighbor + 1], 1);
    }
}
"""

_FILL_REVERSE_KERNEL = r"""
extern "C" __global__
void fill_reverse_kernel(
    const int* __restrict__ exclusion_offset,
    const int* __restrict__ exclusion_neighbors,
    const int* __restrict__ reverse_offset,
    const int num_particles,
    int* __restrict__ reverse_neighbors,
    int* __restrict__ temp_offset
) {
    int atom_a = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom_a >= num_particles) return;
    int start = exclusion_offset[atom_a];
    int end = exclusion_offset[atom_a + 1];
    for (int k = start; k < end; k++) {
        int neighbor = exclusion_neighbors[k];
        int pos = atomicAdd(&temp_offset[neighbor], 1);
        reverse_neighbors[pos] = atom_a;
    }
}
"""

_gpu_kernels = None


def _get_gpu_kernels():
    global _gpu_kernels
    if _gpu_kernels is None:
        _gpu_kernels = {
            'parallel_dedup': cp.RawKernel(_PARALLEL_DEDUP_KERNEL, 'parallel_dedup_kernel'),
            'count_row': cp.RawKernel(_COUNT_ROW_KERNEL, 'count_row_kernel'),
            'scatter_pairs': cp.RawKernel(_SCATTER_PAIRS_KERNEL, 'scatter_pairs_kernel'),
            'rev_count': cp.RawKernel(_BUILD_REVERSE_COUNT_KERNEL, 'build_reverse_count_kernel'),
            'rev_fill': cp.RawKernel(_FILL_REVERSE_KERNEL, 'fill_reverse_kernel'),
        }
    return _gpu_kernels


def _build_bond_graph_exclusion_pairs(bond_indices, num_particles):
    adj = [[] for _ in range(num_particles)]
    for i, j in bond_indices:
        i, j = int(i), int(j)
        adj[i].append(j)
        adj[j].append(i)

    pair_12 = set()
    pair_13 = set()
    pair_14 = set()
    for bond_i, bond_j in bond_indices:
        a2, a3 = int(bond_i), int(bond_j)
        pair_12.add((min(a2, a3), max(a2, a3)))
        for a1 in adj[a2]:
            if a1 != a3:
                pair_13.add((min(a1, a3), max(a1, a3)))
        for a4 in adj[a3]:
            if a4 != a2:
                pair_13.add((min(a2, a4), max(a2, a4)))
    for bond_i, bond_j in bond_indices:
        a2, a3 = int(bond_i), int(bond_j)
        for a1 in adj[a2]:
            for a4 in adj[a3]:
                if a1 != a3 and a2 != a4 and a1 != a4:
                    pair_14.add((min(a1, a4), max(a1, a4)))

    pair_13 -= pair_12
    pair_14 -= pair_12
    pair_14 -= pair_13

    all_i = []
    all_j = []
    for lo, hi in pair_12:
        all_i.append(lo)
        all_j.append(hi)
    for lo, hi in pair_13:
        all_i.append(lo)
        all_j.append(hi)
    for lo, hi in pair_14:
        all_i.append(lo)
        all_j.append(hi)

    n12 = len(pair_12)
    n13 = len(pair_13)
    n14 = len(pair_14)
    if not all_i:
        return (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32),
                0, 0, 0, 0)
    return (np.array(all_i, dtype=np.int32),
            np.array(all_j, dtype=np.int32),
            len(all_i), n12, n13, n14)


class Builder:

    def __init__(self):
        self._num_particles = 0
        self._particles_set = False
        self._bonds: list[list] = []
        self._angles: list[list] = []
        self._dihedrals: list[list] = []
        self._impropers: list[list] = []

    def set_particles(self, num_particles: int) -> 'Builder':
        self._num_particles = num_particles
        self._particles_set = True
        return self

    def add_bond(self, i: int, j: int, k: float, r0: float) -> Builder:
        self._bonds.append([i, j, k, r0])
        return self

    def add_angle(
        self, i: int, j: int, k: int, force_constant: float,
        equilibrium_angle: float, k_ub: float = 0.0, r_ub: float = 0.0,
    ) -> Builder:
        self._angles.append([i, j, k, force_constant, equilibrium_angle, k_ub, r_ub])
        return self

    def add_dihedral(
        self, i: int, j: int, k: int, l: int,
        force_constant: float, periodicity: float, phase: float,
    ) -> Builder:
        self._dihedrals.append([i, j, k, l, force_constant, periodicity, phase])
        return self

    def add_improper(
        self, i: int, j: int, k: int, l: int,
        force_constant: float, equilibrium_angle: float,
    ) -> Builder:
        self._impropers.append([i, j, k, l, force_constant, equilibrium_angle])
        return self

    def add_bond_indices(
        self, indices: np.ndarray, parameters: np.ndarray,
    ) -> Builder:
        for row in range(indices.shape[0]):
            self._bonds.append(
                [indices[row, 0], indices[row, 1],
                 parameters[row, 0], parameters[row, 1]]
            )
        return self

    def build(self) -> tuple:
        if not self._particles_set:
            raise ValueError('set_particles() must be called before build()')
        topology = Topology(self)
        term_params = {}
        if self._bonds:
            bond_data = np.array(self._bonds, dtype=env.NUMPY_FLOAT)
            term_params['bond'] = bond_data[:, 2:].astype(env.NUMPY_FLOAT)
        if self._angles:
            angle_data = np.array(self._angles, dtype=env.NUMPY_FLOAT)
            term_params['angle'] = angle_data[:, 3:].astype(env.NUMPY_FLOAT)
        if self._dihedrals:
            dihed_data = np.array(self._dihedrals, dtype=env.NUMPY_FLOAT)
            term_params['dihedral'] = dihed_data[:, 4:].astype(env.NUMPY_FLOAT)
        if self._impropers:
            improd_data = np.array(self._impropers, dtype=env.NUMPY_FLOAT)
            term_params['improper'] = improd_data[:, 4:].astype(env.NUMPY_FLOAT)
        return topology, term_params
