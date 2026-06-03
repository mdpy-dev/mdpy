from __future__ import annotations

import cupy as cp
import numpy as np
from mdpy import env
from mdpy.core.radix_sort import RadixSorter, fill_constant

_pair_sorter = None


def _get_pair_sorter(max_pairs: int, num_particles: int) -> RadixSorter:
    global _pair_sorter
    if _pair_sorter is None or max_pairs > _pair_sorter._max_elements:
        max_key = num_particles * 2_000_000_000
        num_bits = max(40, (max_key).bit_length())
        num_bits = ((num_bits + 3) // 4) * 4
        _pair_sorter = RadixSorter(max_elements=max_pairs, num_bits=num_bits)
    return _pair_sorter


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
        '_legacy_particles', '_legacy_bonds', '_legacy_angles',
        '_legacy_dihedrals', '_legacy_impropers', '_is_joined',
        '_legacy_bonded_particles', '_legacy_scaling_particles',
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
        self._legacy_particles = []
        self._legacy_bonds = []
        self._legacy_angles = []
        self._legacy_dihedrals = []
        self._legacy_impropers = []
        self._is_joined = False

    @property
    def is_joined(self):
        return getattr(self, '_is_joined', True)

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


_GENERATE_PAIRS_KERNEL = r'''
extern "C" __global__
void generate_pairs_kernel(
    const int* __restrict__ bond_idx, const int num_bonds,
    const int* __restrict__ angle_idx, const int num_angles,
    const int* __restrict__ dihedral_idx, const int num_dihedrals,
    const int* __restrict__ improper_idx, const int num_impropers,
    const float scale_14,
    int* __restrict__ out_i, int* __restrict__ out_j,
    float* __restrict__ out_scale,
    const int total_pairs
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_pairs) return;

    int bond_end = num_bonds;
    int angle_end = bond_end + num_angles;
    int dihedral_end = angle_end + num_dihedrals;

    if (tid < bond_end) {
        int a = bond_idx[tid * 2];
        int b = bond_idx[tid * 2 + 1];
        out_i[tid] = min(a, b);
        out_j[tid] = max(a, b);
        out_scale[tid] = 0.0f;
    } else if (tid < angle_end) {
        int idx = tid - bond_end;
        int a = angle_idx[idx * 3];
        int c = angle_idx[idx * 3 + 2];
        out_i[tid] = min(a, c);
        out_j[tid] = max(a, c);
        out_scale[tid] = 0.0f;
    } else if (tid < dihedral_end) {
        int idx = tid - angle_end;
        int a = dihedral_idx[idx * 4];
        int d = dihedral_idx[idx * 4 + 3];
        out_i[tid] = min(a, d);
        out_j[tid] = max(a, d);
        out_scale[tid] = scale_14;
    } else {
        int idx = tid - dihedral_end;
        int a = improper_idx[idx * 4];
        int d = improper_idx[idx * 4 + 3];
        out_i[tid] = min(a, d);
        out_j[tid] = max(a, d);
        out_scale[tid] = 0.0f;
    }
}
'''

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
    int ni = permutation[old_i[tid]];
    int nj = permutation[old_j[tid]];
    if (ni < nj) {
        new_i[tid] = ni;
        new_j[tid] = nj;
    } else {
        new_i[tid] = nj;
        new_j[tid] = ni;
    }
    new_scale[tid] = old_scale[tid];
}
'''

_BUILD_SORT_KEY_KERNEL = r"""
extern "C" __global__
void build_sort_key_kernel(
    const int* __restrict__ keys_i,
    const int* __restrict__ keys_j,
    const float* __restrict__ keys_scale,
    int num_pairs,
    long long* __restrict__ sort_key
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= num_pairs) return;
    long long ki = (long long)keys_i[p] * 2000000000LL;
    long long kj = (long long)keys_j[p] * 2LL;
    long long ks = (keys_scale[p] > 0.0f) ? 1LL : 0LL;
    sort_key[p] = ki + kj + ks;
}
"""

_GATHER_THREE_KERNEL = r"""
extern "C" __global__
void gather_three_kernel(
    const int* __restrict__ src_i,
    const int* __restrict__ src_j,
    const float* __restrict__ src_s,
    const int* __restrict__ order,
    int num_pairs,
    int* __restrict__ dst_i,
    int* __restrict__ dst_j,
    float* __restrict__ dst_s
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= num_pairs) return;
    int o = order[p];
    dst_i[p] = src_i[o];
    dst_j[p] = src_j[o];
    dst_s[p] = src_s[o];
}
"""

_gpu_kernels = None


def _get_gpu_kernels():
    global _gpu_kernels
    if _gpu_kernels is None:
        _gpu_kernels = {
            'generate': cp.RawKernel(_GENERATE_PAIRS_KERNEL, 'generate_pairs_kernel'),
            'parallel_csr': cp.RawKernel(_PARALLEL_CSR_KERNEL, 'parallel_csr_kernel'),
            'fill_csr_gaps': cp.RawKernel(_FILL_CSR_GAPS_KERNEL, 'fill_csr_gaps_kernel'),
            'parallel_dedup': cp.RawKernel(_PARALLEL_DEDUP_KERNEL, 'parallel_dedup_kernel'),
            'permute_pairs': cp.RawKernel(_PERMUTE_PAIRS_KERNEL, 'permute_pairs_kernel'),
            'build_sort_key': cp.RawKernel(_BUILD_SORT_KEY_KERNEL, 'build_sort_key_kernel'),
            'gather_three': cp.RawKernel(_GATHER_THREE_KERNEL, 'gather_three_kernel'),
        }
    return _gpu_kernels


def build_exclusion_map_gpu(topology, scale_14=1.0):
    num_bonds = topology.num_bonds
    num_angles = topology.num_angles
    num_dihedrals = topology.num_dihedrals
    num_impropers = topology.num_impropers
    total_pairs = num_bonds + num_angles + num_dihedrals + num_impropers
    num_particles = topology.num_particles

    if total_pairs == 0:
        d_offset = cp.zeros(num_particles + 1, dtype=cp.int32)
        d_neighbors = cp.empty(0, dtype=cp.int32)
        d_scale = cp.empty(0, dtype=cp.float32)
        return d_offset, d_neighbors, d_scale, cp.empty(0, dtype=cp.int32)

    kernels = _get_gpu_kernels()

    d_bond_idx = cp.asarray(topology.bond_indices.ravel().astype(np.int32))
    d_angle_idx = cp.asarray(topology.angle_indices.ravel().astype(np.int32))
    d_dihedral_idx = cp.asarray(topology.dihedral_indices.ravel().astype(np.int32))
    d_improper_idx = cp.asarray(topology.improper_indices.ravel().astype(np.int32))

    d_pair_i = cp.empty(total_pairs, dtype=cp.int32)
    d_pair_j = cp.empty(total_pairs, dtype=cp.int32)
    d_pair_scale = cp.empty(total_pairs, dtype=cp.float32)

    block = 256
    grid = (total_pairs + block - 1) // block
    kernels['generate'](
        (grid,), (block,),
        (d_bond_idx, np.int32(num_bonds),
         d_angle_idx, np.int32(num_angles),
         d_dihedral_idx, np.int32(num_dihedrals),
         d_improper_idx, np.int32(num_impropers),
         np.float32(scale_14),
         d_pair_i, d_pair_j, d_pair_scale,
         np.int32(total_pairs))
    )

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

    d_unique_i = cp.full(unique_count, -1, dtype=cp.int32)
    d_unique_j = cp.full(unique_count, -1, dtype=cp.int32)
    d_unique_scale = cp.zeros(unique_count, dtype=cp.float32)

    d_unique_i[scatter_idx] = d_pair_i
    d_unique_j[scatter_idx] = d_pair_j
    d_unique_scale[scatter_idx] = d_pair_scale

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


def permute_exclusion_pairs_gpu(d_cached_i, d_cached_j, d_cached_scale,
                                 d_composed_perm, num_particles):
    num_pairs = len(d_cached_i)
    if num_pairs == 0:
        d_offset = cp.zeros(num_particles + 1, dtype=cp.int32)
        d_neighbors = cp.empty(0, dtype=cp.int32)
        d_scale = cp.empty(0, dtype=cp.float32)
        return d_offset, d_neighbors, d_scale, d_cached_i, d_cached_j, d_cached_scale

    kernels = _get_gpu_kernels()

    d_new_i = cp.empty(num_pairs, dtype=cp.int32)
    d_new_j = cp.empty(num_pairs, dtype=cp.int32)
    d_new_scale = cp.empty(num_pairs, dtype=cp.float32)

    tpb = 256
    grid = ((num_pairs + tpb - 1) // tpb,)
    kernels['permute_pairs'](grid, (tpb,),
        (d_cached_i, d_cached_j, d_cached_scale,
         d_composed_perm, np.int32(num_pairs),
         d_new_i, d_new_j, d_new_scale))

    sort_key = cp.empty(num_pairs, dtype=cp.int64)
    kernels['build_sort_key'](
        grid, (tpb,),
        (d_new_i, d_new_j, d_new_scale, np.int32(num_pairs), sort_key),
    )
    sorter = _get_pair_sorter(num_pairs, num_particles)
    order = sorter.argsort(sort_key.view(np.uint64))
    d_sorted_i = cp.empty(num_pairs, dtype=cp.int32)
    d_sorted_j = cp.empty(num_pairs, dtype=cp.int32)
    d_sorted_scale = cp.empty(num_pairs, dtype=cp.float32)
    kernels['gather_three'](
        grid, (tpb,),
        (d_new_i, d_new_j, d_new_scale, order, np.int32(num_pairs),
         d_sorted_i, d_sorted_j, d_sorted_scale),
    )
    d_new_i = d_sorted_i
    d_new_j = d_sorted_j
    d_new_scale = d_sorted_scale

    d_offset = cp.empty(num_particles + 1, dtype=cp.int32)
    fill_constant(d_offset, -1)
    grid_csr = ((num_pairs + 1 + tpb - 1) // tpb,)
    kernels['parallel_csr'](grid_csr, (tpb,),
        (d_new_i, np.int32(num_pairs),
         np.int32(num_particles), d_offset))

    grid_fill = ((num_particles + tpb - 1) // tpb,)
    kernels['fill_csr_gaps'](grid_fill, (tpb,),
        (d_offset, np.int32(num_particles)))

    return d_offset, d_new_j, d_new_scale, d_new_i, d_new_j, d_new_scale


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

    def add_angle_indices(
        self, indices: np.ndarray, parameters: np.ndarray,
    ) -> Builder:
        for row in range(indices.shape[0]):
            self._angles.append(
                [indices[row, 0], indices[row, 1], indices[row, 2],
                 parameters[row, 0], parameters[row, 1],
                 parameters[row, 2], parameters[row, 3]]
            )
        return self

    def add_dihedral_indices(
        self, indices: np.ndarray, parameters: np.ndarray,
    ) -> Builder:
        for row in range(indices.shape[0]):
            self._dihedrals.append(
                [indices[row, 0], indices[row, 1],
                 indices[row, 2], indices[row, 3],
                 parameters[row, 0], parameters[row, 1], parameters[row, 2]]
            )
        return self

    def add_improper_indices(
        self, indices: np.ndarray, parameters: np.ndarray,
    ) -> Builder:
        for row in range(indices.shape[0]):
            self._impropers.append(
                [indices[row, 0], indices[row, 1],
                 indices[row, 2], indices[row, 3],
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
            _add(dihedral[0], dihedral[3], scale_14)

        for improper in self._impropers:
            _add(improper[0], improper[3], 0.0)

        sorted_pairs = []
        for particle_index in range(num_particles):
            neighbors = sorted(exclusion_dict[particle_index].keys())
            for neighbor in neighbors:
                sorted_pairs.append(
                    (particle_index, neighbor, exclusion_dict[particle_index][neighbor])
                )

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
