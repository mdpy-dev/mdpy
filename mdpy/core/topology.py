from __future__ import annotations

import cupy as cp
import numpy as np
from mdpy import precision



class Topology:

    __slots__ = [
        'num_particles',
        '_bond_list', '_angle_list', '_dihedral_list', '_improper_list',
        '_bond_np', '_angle_np', '_dihedral_np', '_improper_np',
        '_exclusion_dirty', '_exclusion_pairs', '_exclusion_csr',
        '_exclusion_reverse_csr',
    ]

    def __init__(self):
        self.num_particles = 0
        self._bond_list: list[tuple[int, int]] = []
        self._angle_list: list[tuple[int, int, int]] = []
        self._dihedral_list: list[tuple[int, int, int, int]] = []
        self._improper_list: list[tuple[int, int, int, int]] = []
        self._bond_np = None
        self._angle_np = None
        self._dihedral_np = None
        self._improper_np = None
        self._init_exclusion_cache()

    def _invalidate_numpy_caches(self):
        self._bond_np = None
        self._angle_np = None
        self._dihedral_np = None
        self._improper_np = None

    def add_bond(self, i: int, j: int):
        self._bond_list.append((int(i), int(j)))
        self._invalidate_numpy_caches()
        self._exclusion_dirty = True

    def add_angle(self, i: int, j: int, k: int):
        self._angle_list.append((int(i), int(j), int(k)))
        self._invalidate_numpy_caches()

    def add_dihedral(self, i: int, j: int, k: int, l: int):
        self._dihedral_list.append((int(i), int(j), int(k), int(l)))
        self._invalidate_numpy_caches()

    def add_improper(self, i: int, j: int, k: int, l: int):
        self._improper_list.append((int(i), int(j), int(k), int(l)))
        self._invalidate_numpy_caches()

    @property
    def bond_indices(self):
        if self._bond_np is None:
            if self._bond_list:
                self._bond_np = np.array(self._bond_list, dtype=precision.INT)
            else:
                self._bond_np = np.empty((0, 2), dtype=precision.INT)
        return self._bond_np

    @property
    def num_bonds(self):
        return len(self._bond_list)

    @property
    def angle_indices(self):
        if self._angle_np is None:
            if self._angle_list:
                self._angle_np = np.array(self._angle_list, dtype=precision.INT)
            else:
                self._angle_np = np.empty((0, 3), dtype=precision.INT)
        return self._angle_np

    @property
    def num_angles(self):
        return len(self._angle_list)

    @property
    def dihedral_indices(self):
        if self._dihedral_np is None:
            if self._dihedral_list:
                self._dihedral_np = np.array(self._dihedral_list, dtype=precision.INT)
            else:
                self._dihedral_np = np.empty((0, 4), dtype=precision.INT)
        return self._dihedral_np

    @property
    def num_dihedrals(self):
        return len(self._dihedral_list)

    @property
    def improper_indices(self):
        if self._improper_np is None:
            if self._improper_list:
                self._improper_np = np.array(self._improper_list, dtype=precision.INT)
            else:
                self._improper_np = np.empty((0, 4), dtype=precision.INT)
        return self._improper_np

    @property
    def num_impropers(self):
        return len(self._improper_list)

    def _init_exclusion_cache(self):
        self._exclusion_dirty = True
        self._exclusion_pairs = None
        self._exclusion_csr = None
        self._exclusion_reverse_csr = None

    def _derive_exclusion_state(self):
        N = self.num_particles
        kernels = _get_gpu_kernels()
        threads_per_block = 256

        pair_i_np, pair_j_np, total_pairs, _, _, _ = \
            _build_bond_graph_exclusion_pairs(self.bond_indices, N)

        if total_pairs == 0:
            zi = cp.empty(0, dtype=precision.INT)
            self._exclusion_pairs = (zi, zi)
            empty_off = cp.zeros(N + 1, dtype=precision.INT)
            self._exclusion_csr = (empty_off, zi)
            self._exclusion_reverse_csr = (empty_off.copy(), zi)
            self._exclusion_dirty = False
            return

        d_pair_i = cp.asarray(pair_i_np)
        d_pair_j = cp.asarray(pair_j_np)

        sort_key_stride = np.int64(2_000_000_000)
        sort_key = d_pair_i.astype(cp.int64) * sort_key_stride + d_pair_j.astype(cp.int64)
        order = cp.argsort(sort_key)
        d_pair_i, d_pair_j = d_pair_i[order], d_pair_j[order]

        d_flags = cp.zeros(total_pairs, dtype=precision.INT)
        grid_p = ((total_pairs + threads_per_block - 1) // threads_per_block,)
        kernels['parallel_dedup'](grid_p, (threads_per_block,),
            (d_pair_i, d_pair_j, np.int32(total_pairs), d_flags))
        scatter_idx = cp.cumsum(d_flags) - 1
        uniq_count = int(scatter_idx[total_pairs - 1]) + 1

        d_u_i = cp.full(uniq_count, -1, dtype=precision.INT)
        d_u_j = cp.full(uniq_count, -1, dtype=precision.INT)
        d_u_i[scatter_idx] = d_pair_i
        d_u_j[scatter_idx] = d_pair_j

        d_bi_i = cp.concatenate([d_u_i, d_u_j])
        d_bi_j = cp.concatenate([d_u_j, d_u_i])
        bi_count = d_bi_i.shape[0]
        bi_key = d_bi_i.astype(cp.int64) * sort_key_stride + d_bi_j.astype(cp.int64)
        bi_order = cp.argsort(bi_key)
        d_bi_i, d_bi_j = d_bi_i[bi_order], d_bi_j[bi_order]

        d_bi_flags = cp.zeros(bi_count, dtype=precision.INT)
        grid_b = ((bi_count + threads_per_block - 1) // threads_per_block,)
        kernels['parallel_dedup'](grid_b, (threads_per_block,),
            (d_bi_i, d_bi_j, np.int32(bi_count), d_bi_flags))
        bi_scatter = cp.cumsum(d_bi_flags) - 1
        bi_uniq = int(bi_scatter[bi_count - 1]) + 1
        d_unique_i = cp.full(bi_uniq, -1, dtype=precision.INT)
        d_unique_j = cp.full(bi_uniq, -1, dtype=precision.INT)
        d_unique_i[bi_scatter] = d_bi_i
        d_unique_j[bi_scatter] = d_bi_j

        self._exclusion_pairs = (d_unique_i, d_unique_j)

        num_pairs = bi_uniq
        d_count = cp.zeros(N + 1, dtype=precision.INT)
        grid_c = ((num_pairs + threads_per_block - 1) // threads_per_block,)
        kernels['count_row'](grid_c, (threads_per_block,),
            (d_unique_i, np.int32(num_pairs), d_count))
        d_offset = cp.empty(N + 1, dtype=precision.INT)
        cp.cumsum(d_count, dtype=cp.int32, out=d_offset)
        d_neighbors = cp.empty(num_pairs, dtype=precision.INT)
        d_fwd_write_cursor = cp.empty(N + 1, dtype=precision.INT)
        d_fwd_write_cursor[:] = d_offset
        kernels['scatter_pairs'](grid_c, (threads_per_block,),
            (d_unique_i, d_unique_j, d_offset, np.int32(num_pairs),
             d_neighbors, d_fwd_write_cursor))
        self._exclusion_csr = (d_offset, d_neighbors)

        n1 = (N + threads_per_block - 1) // threads_per_block
        d_rev_offset = cp.zeros(N + 1, dtype=precision.INT)
        kernels['rev_count']((n1,), (threads_per_block,),
            (d_offset, d_neighbors, np.int32(N), d_rev_offset))
        d_rev_offset = cp.cumsum(d_rev_offset, dtype=cp.int32).astype(precision.INT)
        max_rev = num_pairs if num_pairs > 0 else int(d_rev_offset[-1])
        d_rev_neighbors = cp.empty(max_rev, dtype=precision.INT)
        d_rev_write_cursor = d_rev_offset.copy()
        kernels['rev_fill']((n1,), (threads_per_block,),
            (d_offset, d_neighbors, d_rev_offset, np.int32(N),
             d_rev_neighbors, d_rev_write_cursor))
        self._exclusion_reverse_csr = (d_rev_offset, d_rev_neighbors)

        self._exclusion_dirty = False

    @property
    def exclusion_pairs(self):
        if self._exclusion_dirty:
            self._derive_exclusion_state()
        return self._exclusion_pairs

    @property
    def exclusion_csr(self):
        if self._exclusion_dirty:
            self._derive_exclusion_state()
        return self._exclusion_csr

    @property
    def exclusion_reverse_csr(self):
        if self._exclusion_dirty:
            self._derive_exclusion_state()
        return self._exclusion_reverse_csr

    def invalidate_exclusions(self):
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


# --- CUDA kernels (unchanged) ---

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
