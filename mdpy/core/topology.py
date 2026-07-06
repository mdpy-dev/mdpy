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
        'exclusion_offset', 'exclusion_neighbors', 'exclusion_scale',
        'masses', 'charges', 'particle_types', 'molecule_ids',
        'particle_names', 'type_names', 'chain_ids', 'molecule_types',
        '_is_joined',
    ]

    def __init__(self, builder: Builder | None = None):
        if builder is None:
            self._init_legacy()
            return
        self.num_particles = builder._num_particles
        self.masses = builder._masses.copy()
        self.charges = builder._charges.copy()
        self.particle_types = builder._particle_types.copy()
        self.molecule_ids = builder._molecule_ids.copy()
        self.particle_names = list(builder._particle_names)
        self.type_names = list(builder._type_names)
        self.chain_ids = list(builder._chain_ids)
        self.molecule_types = list(builder._molecule_types)

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

        if builder._exclusion_offset is not None:
            self.exclusion_offset = builder._exclusion_offset.copy()
            self.exclusion_neighbors = builder._exclusion_neighbors.copy()
            self.exclusion_scale = builder._exclusion_scale.copy()
        else:
            self.exclusion_offset = np.zeros(
                self.num_particles + 1, dtype=env.NUMPY_INT
            )
            self.exclusion_neighbors = np.empty(0, dtype=env.NUMPY_INT)
            self.exclusion_scale = np.empty(0, dtype=env.NUMPY_FLOAT)

    def _init_legacy(self):
        self.num_particles = 0
        self.masses = np.empty(0, dtype=env.NUMPY_FLOAT)
        self.charges = np.empty(0, dtype=env.NUMPY_FLOAT)
        self.particle_types = np.empty(0, dtype=env.NUMPY_INT)
        self.molecule_ids = np.empty(0, dtype=env.NUMPY_INT)
        self.particle_names = []
        self.type_names = []
        self.chain_ids = []
        self.molecule_types = []
        self.bond_indices = np.empty((0, 2), dtype=env.NUMPY_INT)
        self.num_bonds = 0
        self.angle_indices = np.empty((0, 3), dtype=env.NUMPY_INT)
        self.num_angles = 0
        self.dihedral_indices = np.empty((0, 4), dtype=env.NUMPY_INT)
        self.num_dihedrals = 0
        self.improper_indices = np.empty((0, 4), dtype=env.NUMPY_INT)
        self.num_impropers = 0
        self.exclusion_offset = np.zeros(1, dtype=env.NUMPY_INT)
        self.exclusion_neighbors = np.empty(0, dtype=env.NUMPY_INT)
        self.exclusion_scale = np.empty(0, dtype=env.NUMPY_FLOAT)
        self._is_joined = False

    def join(self):
        self._is_joined = True

    def __repr__(self) -> str:
        return (
            '<mdpy.core.Topology: %d particles, %d bonds, %d angles, '
            '%d dihedrals, %d impropers>'
            % (
                self.num_particles, self.num_bonds, self.num_angles,
                self.num_dihedrals, self.num_impropers,
            )
        )


_PARALLEL_CSR_KERNEL = r'''
extern "C" __global__
void parallel_csr_kernel(
    const int* __restrict__ sorted_i,
    const int num_unique,
    const int num_particles,
    int* __restrict__ offset
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        offset[num_particles] = num_unique;
    }
    if (tid >= num_unique) return;
    if (tid == 0 || sorted_i[tid] != sorted_i[tid - 1]) {
        offset[sorted_i[tid]] = tid;
    }
}
'''

_FILL_CSR_GAPS_KERNEL = r'''
extern "C" __global__
void fill_csr_gaps_kernel(
    int* __restrict__ offset,
    const int num_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    if (offset[i] >= 0) return;
    for (int j = i + 1; j <= num_particles; j++) {
        if (offset[j] >= 0) {
            offset[i] = offset[j];
            break;
        }
    }
}
'''

_PARALLEL_DEDUP_KERNEL = r'''
extern "C" __global__
void parallel_dedup_kernel(
    const int* __restrict__ sorted_i,
    const int* __restrict__ sorted_j,
    const float* __restrict__ sorted_scale,
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

_PERMUTE_PAIRS_KERNEL = r'''
extern "C" __global__
void permute_pairs_kernel(
    const int* __restrict__ old_i,
    const int* __restrict__ old_j,
    const float* __restrict__ old_scale,
    const int* __restrict__ permutation,
    const int num_pairs,
    int* __restrict__ new_i,
    int* __restrict__ new_j,
    float* __restrict__ new_scale
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_pairs) return;
    new_i[tid] = permutation[old_i[tid]];
    new_j[tid] = permutation[old_j[tid]];
    new_scale[tid] = old_scale[tid];
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
    const float* __restrict__ pair_scale,
    const int* __restrict__ offset,
    int num_pairs,
    int* __restrict__ neighbors_out,
    float* __restrict__ scale_out,
    int* __restrict__ temp_offset
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_pairs) return;
    int i = pair_i[tid];
    int pos = atomicAdd(&temp_offset[i], 1);
    neighbors_out[pos] = pair_j[tid];
    scale_out[pos] = pair_scale[tid];
}
"""

_gpu_kernels = None


def _get_gpu_kernels():
    global _gpu_kernels
    if _gpu_kernels is None:
        _gpu_kernels = {
            'parallel_csr': cp.RawKernel(_PARALLEL_CSR_KERNEL, 'parallel_csr_kernel'),
            'fill_csr_gaps': cp.RawKernel(_FILL_CSR_GAPS_KERNEL, 'fill_csr_gaps_kernel'),
            'parallel_dedup': cp.RawKernel(_PARALLEL_DEDUP_KERNEL, 'parallel_dedup_kernel'),
            'permute_pairs': cp.RawKernel(_PERMUTE_PAIRS_KERNEL, 'permute_pairs_kernel'),
            'count_row': cp.RawKernel(_COUNT_ROW_KERNEL, 'count_row_kernel'),
            'scatter_pairs': cp.RawKernel(_SCATTER_PAIRS_KERNEL, 'scatter_pairs_kernel'),
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


def build_exclusion_map_gpu(topology, scale_14=1.0):
    num_particles = topology.num_particles

    pair_i_np, pair_j_np, total_pairs, _, _, _ = _build_bond_graph_exclusion_pairs(
        topology.bond_indices, num_particles
    )
    if total_pairs == 0:
        d_offset = cp.zeros(num_particles + 1, dtype=cp.int32)
        d_neighbors = cp.empty(0, dtype=cp.int32)
        d_scale = cp.empty(0, dtype=cp.float32)
        return d_offset, d_neighbors, d_scale, cp.empty(0, dtype=cp.int32)

    kernels = _get_gpu_kernels()

    d_pair_i = cp.asarray(pair_i_np)
    d_pair_j = cp.asarray(pair_j_np)
    d_pair_scale = cp.zeros(total_pairs, dtype=cp.float32)

    scale_rank = (d_pair_scale > 0.0).astype(cp.int64)
    sort_key = (d_pair_i.astype(cp.int64) * np.int64(2000000000)
                + d_pair_j.astype(cp.int64) * np.int64(2)
                + scale_rank)
    order = cp.argsort(sort_key)
    d_pair_i = d_pair_i[order]
    d_pair_j = d_pair_j[order]
    d_pair_scale = d_pair_scale[order]

    d_flags = cp.zeros(total_pairs, dtype=cp.int32)
    tpb_dedup = 256
    grid_dedup = ((total_pairs + tpb_dedup - 1) // tpb_dedup,)
    kernels['parallel_dedup'](grid_dedup, (tpb_dedup,),
        (d_pair_i, d_pair_j, d_pair_scale,
         np.int32(total_pairs), d_flags))

    scatter_idx = cp.cumsum(d_flags) - 1
    unique_count = int(scatter_idx[total_pairs - 1]) + 1

    d_dedup_i = cp.full(unique_count, -1, dtype=cp.int32)
    d_dedup_j = cp.full(unique_count, -1, dtype=cp.int32)
    d_dedup_scale = cp.zeros(unique_count, dtype=cp.float32)

    d_dedup_i[scatter_idx] = d_pair_i
    d_dedup_j[scatter_idx] = d_pair_j
    d_dedup_scale[scatter_idx] = d_pair_scale

    d_bi_i = cp.concatenate([d_dedup_i, d_dedup_j])
    d_bi_j = cp.concatenate([d_dedup_j, d_dedup_i])
    d_bi_scale = cp.concatenate([d_dedup_scale, d_dedup_scale])
    bi_count = len(d_bi_i)

    bi_scale_rank = (d_bi_scale > 0.0).astype(cp.int64)
    bi_sort_key = (d_bi_i.astype(cp.int64) * np.int64(2000000000)
                   + d_bi_j.astype(cp.int64) * np.int64(2)
                   + bi_scale_rank)
    bi_order = cp.argsort(bi_sort_key)
    d_bi_i = d_bi_i[bi_order]
    d_bi_j = d_bi_j[bi_order]
    d_bi_scale = d_bi_scale[bi_order]

    d_bi_flags = cp.zeros(bi_count, dtype=cp.int32)
    grid_dedup2 = ((bi_count + tpb_dedup - 1) // tpb_dedup,)
    kernels['parallel_dedup'](grid_dedup2, (tpb_dedup,),
        (d_bi_i, d_bi_j, d_bi_scale,
         np.int32(bi_count), d_bi_flags))

    bi_scatter = cp.cumsum(d_bi_flags) - 1
    bi_unique_count = int(bi_scatter[bi_count - 1]) + 1

    d_unique_i = cp.full(bi_unique_count, -1, dtype=cp.int32)
    d_unique_j = cp.full(bi_unique_count, -1, dtype=cp.int32)
    d_unique_scale = cp.zeros(bi_unique_count, dtype=cp.float32)

    d_unique_i[bi_scatter] = d_bi_i
    d_unique_j[bi_scatter] = d_bi_j
    d_unique_scale[bi_scatter] = d_bi_scale
    unique_count = bi_unique_count

    d_offset = cp.full(num_particles + 1, -1, dtype=cp.int32)
    tpb_csr = 256
    grid_csr = ((unique_count + 1 + tpb_csr - 1) // tpb_csr,)
    kernels['parallel_csr'](
        grid_csr, (tpb_csr,),
        (d_unique_i, np.int32(unique_count),
         np.int32(num_particles), d_offset))

    tpb_fill = 256
    grid_fill = ((num_particles + tpb_fill - 1) // tpb_fill,)
    kernels['fill_csr_gaps'](
        grid_fill, (tpb_fill,),
        (d_offset, np.int32(num_particles)))

    return d_offset, d_unique_j, d_unique_scale, d_unique_i


def _excl_get(name, size, dtype, pool, fill=None):
    """Return a reusable buffer from the given exclusion pool. The pool is
    double-buffered: the caller selects pool A or B once per call, so a call's
    outputs (next call's inputs) never alias."""
    key = (name, dtype)
    arr = pool.get(key)
    if arr is None or arr.size < size:
        arr = cp.empty(size, dtype=dtype)
        pool[key] = arr
    arr = arr[:size]
    if fill == 0:
        cp.cuda.runtime.memsetAsync(
            arr.data.ptr, 0, size * arr.itemsize, cp.cuda.Stream.null.ptr
        )
    return arr


def permute_exclusion_pairs_gpu(d_cached_i, d_cached_j, d_cached_scale,
                                 d_composed_perm, num_particles, pool):
    num_pairs = len(d_cached_i)
    if num_pairs == 0:
        d_offset = _excl_get("offset", num_particles + 1, np.int32, pool, fill=0)
        d_neighbors = _excl_get("neighbors", 0, np.int32, pool)
        d_scale = _excl_get("scale_out", 0, np.float32, pool)
        return d_offset, d_neighbors, d_scale, d_cached_i, d_cached_j, d_cached_scale

    kernels = _get_gpu_kernels()

    d_new_i = _excl_get("new_i", num_pairs, np.int32, pool)
    d_new_j = _excl_get("new_j", num_pairs, np.int32, pool)
    d_new_scale = _excl_get("new_scale", num_pairs, np.float32, pool)

    tpb = 256
    grid = ((num_pairs + tpb - 1) // tpb,)
    kernels['permute_pairs'](grid, (tpb,),
        (d_cached_i, d_cached_j, d_cached_scale,
         d_composed_perm, np.int32(num_pairs),
         d_new_i, d_new_j, d_new_scale))

    d_count = _excl_get("count", num_particles + 1, np.int32, pool, fill=0)
    kernels['count_row'](grid, (tpb,),
        (d_new_i, np.int32(num_pairs), d_count))

    d_offset = _excl_get("offset", num_particles + 1, np.int32, pool)
    cp.cumsum(d_count, dtype=cp.int32, out=d_offset)

    d_neighbors = _excl_get("neighbors", num_pairs, np.int32, pool)
    d_scale_out = _excl_get("scale_out", num_pairs, np.float32, pool)
    d_temp = _excl_get("temp", num_particles + 1, np.int32, pool)
    d_temp[:] = d_offset

    kernels['scatter_pairs'](grid, (tpb,),
        (d_new_i, d_new_j, d_new_scale, d_offset,
         np.int32(num_pairs),
         d_neighbors, d_scale_out, d_temp))

    return d_offset, d_neighbors, d_scale_out, d_new_i, d_new_j, d_new_scale


class Builder:

    def __init__(self):
        self._num_particles = 0
        self._masses = None
        self._charges = None
        self._particle_types = None
        self._molecule_ids = None
        self._particle_names: list[str] = []
        self._type_names: list[str] = []
        self._chain_ids: list[str] = []
        self._molecule_types: list[str] = []
        self._bonds: list[list] = []
        self._angles: list[list] = []
        self._dihedrals: list[list] = []
        self._impropers: list[list] = []
        self._exclusion_offset = None
        self._exclusion_neighbors = None
        self._exclusion_scale = None

    def set_particles(
        self,
        masses: np.ndarray,
        charges: np.ndarray,
        particle_types: np.ndarray,
        molecule_ids: np.ndarray | None = None,
        particle_names: list[str] | None = None,
        type_names: list[str] | None = None,
        chain_ids: list[str] | None = None,
        molecule_types: list[str] | None = None,
    ) -> Builder:
        self._num_particles = len(masses)
        self._masses = np.asarray(masses, dtype=env.NUMPY_FLOAT)
        self._charges = np.asarray(charges, dtype=env.NUMPY_FLOAT)
        self._particle_types = np.asarray(particle_types, dtype=env.NUMPY_INT)
        if molecule_ids is not None:
            self._molecule_ids = np.asarray(molecule_ids, dtype=env.NUMPY_INT)
        else:
            self._molecule_ids = np.zeros(self._num_particles, dtype=env.NUMPY_INT)
        self._particle_names = particle_names or [''] * self._num_particles
        self._type_names = type_names or [''] * self._num_particles
        self._chain_ids = chain_ids or [''] * self._num_particles
        self._molecule_types = molecule_types or [''] * self._num_particles
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

    def build_exclusion_map(self, scale_14: float = 1.0) -> Builder:
        num_particles = self._num_particles
        exclusion_dict: dict[int, dict[int, float]] = {
            i: {} for i in range(num_particles)
        }

        def _add(pair_i: int, pair_j: int, scale: float):
            if pair_j < pair_i:
                pair_i, pair_j = pair_j, pair_i
            if pair_j not in exclusion_dict[pair_i]:
                exclusion_dict[pair_i][pair_j] = scale
            else:
                exclusion_dict[pair_i][pair_j] = min(
                    exclusion_dict[pair_i][pair_j], scale
                )

        for bond in self._bonds:
            _add(bond[0], bond[1], 0.0)

        for angle in self._angles:
            _add(angle[0], angle[2], 0.0)

        for dihedral in self._dihedrals:
            _add(dihedral[0], dihedral[3], 0.0)

        for improper in self._impropers:
            _add(improper[0], improper[3], 0.0)

        sorted_pairs = []
        for particle_index in range(num_particles):
            neighbors = sorted(exclusion_dict[particle_index].keys())
            for neighbor in neighbors:
                scale = exclusion_dict[particle_index][neighbor]
                sorted_pairs.append((particle_index, neighbor, scale))
                sorted_pairs.append((neighbor, particle_index, scale))
        sorted_pairs.sort()

        offset = np.zeros(num_particles + 1, dtype=env.NUMPY_INT)
        neighbors_array = np.empty(len(sorted_pairs), dtype=env.NUMPY_INT)
        scale_array = np.empty(len(sorted_pairs), dtype=env.NUMPY_FLOAT)

        pair_index = 0
        for particle_index in range(num_particles):
            offset[particle_index] = pair_index
            while (
                pair_index < len(sorted_pairs)
                and sorted_pairs[pair_index][0] == particle_index
            ):
                neighbors_array[pair_index] = sorted_pairs[pair_index][1]
                scale_array[pair_index] = sorted_pairs[pair_index][2]
                pair_index += 1
        offset[num_particles] = pair_index

        self._exclusion_offset = offset
        self._exclusion_neighbors = neighbors_array
        self._exclusion_scale = scale_array
        return self

    def build(self) -> tuple:
        if self._masses is None:
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
