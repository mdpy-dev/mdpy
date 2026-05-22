from __future__ import annotations

import numpy as np
import cupy as cp
from numba import njit
from mdpy import env

W = 32

_HALF_NEIGHBORS = [
    (0, 0, 0),
    (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (1, 1, 0), (1, 0, 1), (0, 1, 1),
    (1, -1, 0), (1, 0, -1), (0, 1, -1),
    (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
]

_PBC_WRAP_KERNEL = r"""
extern "C" __global__
void pbc_wrap_kernel(
    float* positions,
    const float* pbc_matrix,
    const float* pbc_inv,
    int number_particles
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= number_particles) return;

    float px = positions[index * 3 + 0];
    float py = positions[index * 3 + 1];
    float pz = positions[index * 3 + 2];

    float fx = px * pbc_inv[0] + py * pbc_inv[3] + pz * pbc_inv[6];
    float fy = px * pbc_inv[1] + py * pbc_inv[4] + pz * pbc_inv[7];
    float fz = px * pbc_inv[2] + py * pbc_inv[5] + pz * pbc_inv[8];

    fx = fx - floorf(fx);
    fy = fy - floorf(fy);
    fz = fz - floorf(fz);

    positions[index * 3 + 0] = fx * pbc_matrix[0] + fy * pbc_matrix[3] + fz * pbc_matrix[6];
    positions[index * 3 + 1] = fx * pbc_matrix[1] + fy * pbc_matrix[4] + fz * pbc_matrix[7];
    positions[index * 3 + 2] = fx * pbc_matrix[2] + fy * pbc_matrix[5] + fz * pbc_matrix[8];
}
"""

_MORTON_ENCODE_KERNEL = r"""
__device__ unsigned long long morton_split(unsigned int v) {
    v = v & 0x000003FFu;
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v << 8))  & 0x0300F00Fu;
    v = (v | (v << 4))  & 0x030C30C3u;
    v = (v | (v << 2))  & 0x09249249u;
    return (unsigned long long)v;
}

extern "C" __global__
void morton_encode_kernel(
    const float* positions,
    const float* pbc_matrix,
    const float* pbc_inv,
    int number_particles,
    float cell_size,
    int nx, int ny, int nz,
    unsigned long long* morton_codes,
    int* bin_coords
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= number_particles) return;

    float px = positions[index * 3 + 0];
    float py = positions[index * 3 + 1];
    float pz = positions[index * 3 + 2];

    float fx = px * pbc_inv[0] + py * pbc_inv[3] + pz * pbc_inv[6];
    float fy = px * pbc_inv[1] + py * pbc_inv[4] + pz * pbc_inv[7];
    float fz = px * pbc_inv[2] + py * pbc_inv[5] + pz * pbc_inv[8];

    fx = fx - floorf(fx);
    fy = fy - floorf(fy);
    fz = fz - floorf(fz);

    float wx = fx * pbc_matrix[0] + fy * pbc_matrix[3] + fz * pbc_matrix[6];
    float wy = fx * pbc_matrix[1] + fy * pbc_matrix[4] + fz * pbc_matrix[7];
    float wz = fx * pbc_matrix[2] + fy * pbc_matrix[5] + fz * pbc_matrix[8];

    int bx = (int)floorf(wx / cell_size);
    int by = (int)floorf(wy / cell_size);
    int bz = (int)floorf(wz / cell_size);

    if (bx < 0) bx = 0; else if (bx >= nx) bx = nx - 1;
    if (by < 0) by = 0; else if (by >= ny) by = ny - 1;
    if (bz < 0) bz = 0; else if (bz >= nz) bz = nz - 1;

    bin_coords[index * 3 + 0] = bx;
    bin_coords[index * 3 + 1] = by;
    bin_coords[index * 3 + 2] = bz;

    morton_codes[index] = morton_split(bx) | (morton_split(by) << 1) | (morton_split(bz) << 2);
}
"""

_BLOCK_CUT_KERNEL = r"""
extern "C" __global__
void block_cut_kernel(
    const int* sorted_indices,
    int number_particles,
    int num_bins_total,
    const int* bin_start_indices,
    const int* bin_block_offsets,
    const int* bin_num_blocks,
    const int* bin_coord_packed,
    int* block_atoms_out,
    int* block_bin_out,
    int block_width
) {
    int bin_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin_index >= num_bins_total) return;

    int bin_start = bin_start_indices[bin_index];
    int bin_count = (bin_index + 1 < num_bins_total)
                    ? bin_start_indices[bin_index + 1] - bin_start
                    : number_particles - bin_start;
    if (bin_count == 0) return;

    int num_full = bin_count / block_width;
    int remainder = bin_count % block_width;
    int num_blocks_in_bin = num_full + (remainder > 0 ? 1 : 0);
    int block_offset = bin_block_offsets[bin_index];

    int bx = bin_coord_packed[bin_index * 3 + 0];
    int by = bin_coord_packed[bin_index * 3 + 1];
    int bz = bin_coord_packed[bin_index * 3 + 2];

    for (int block_local = 0; block_local < num_blocks_in_bin; block_local++) {
        int global_block = block_offset + block_local;
        int atom_offset = bin_start + block_local * block_width;
        int count = (block_local < num_full) ? block_width : remainder;

        for (int slot = 0; slot < block_width; slot++) {
            if (slot < count) {
                block_atoms_out[global_block * block_width + slot] = sorted_indices[atom_offset + slot];
            } else {
                block_atoms_out[global_block * block_width + slot] = -1;
            }
        }
        block_bin_out[global_block * 3 + 0] = bx;
        block_bin_out[global_block * 3 + 1] = by;
        block_bin_out[global_block * 3 + 2] = bz;
    }
}
"""

_COMPUTE_AABB_KERNEL = r"""
extern "C" __global__
void compute_aabb_kernel(
    const float* positions,
    const int* block_atoms,
    int num_blocks,
    int block_width,
    float* aabb_min_out,
    float* aabb_max_out
) {
    int block_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_index >= num_blocks) return;

    float mn_x = 1e30f, mn_y = 1e30f, mn_z = 1e30f;
    float mx_x = -1e30f, mx_y = -1e30f, mx_z = -1e30f;

    for (int slot = 0; slot < block_width; slot++) {
        int atom_id = block_atoms[block_index * block_width + slot];
        if (atom_id < 0) continue;
        float px = positions[atom_id * 3 + 0];
        float py = positions[atom_id * 3 + 1];
        float pz = positions[atom_id * 3 + 2];
        if (px < mn_x) mn_x = px;
        if (py < mn_y) mn_y = py;
        if (pz < mn_z) mn_z = pz;
        if (px > mx_x) mx_x = px;
        if (py > mx_y) mx_y = py;
        if (pz > mx_z) mx_z = pz;
    }

    aabb_min_out[block_index * 3 + 0] = mn_x;
    aabb_min_out[block_index * 3 + 1] = mn_y;
    aabb_min_out[block_index * 3 + 2] = mn_z;
    aabb_max_out[block_index * 3 + 0] = mx_x;
    aabb_max_out[block_index * 3 + 1] = mx_y;
    aabb_max_out[block_index * 3 + 2] = mx_z;
}
"""

_FIND_TILES_KERNEL = r"""
__device__ inline float block_dist_sq(
    const float* aabb_min_a, const float* aabb_max_a,
    const float* aabb_min_b, const float* aabb_max_b,
    float box_x, float box_y, float box_z
) {
    float ca_x = 0.5f * (aabb_min_a[0] + aabb_max_a[0]);
    float ca_y = 0.5f * (aabb_min_a[1] + aabb_max_a[1]);
    float ca_z = 0.5f * (aabb_min_a[2] + aabb_max_a[2]);
    float cb_x = 0.5f * (aabb_min_b[0] + aabb_max_b[0]);
    float cb_y = 0.5f * (aabb_min_b[1] + aabb_max_b[1]);
    float cb_z = 0.5f * (aabb_min_b[2] + aabb_max_b[2]);
    float ha_x = 0.5f * (aabb_max_a[0] - aabb_min_a[0]);
    float ha_y = 0.5f * (aabb_max_a[1] - aabb_min_a[1]);
    float ha_z = 0.5f * (aabb_max_a[2] - aabb_min_a[2]);
    float hb_x = 0.5f * (aabb_max_b[0] - aabb_min_b[0]);
    float hb_y = 0.5f * (aabb_max_b[1] - aabb_min_b[1]);
    float hb_z = 0.5f * (aabb_max_b[2] - aabb_min_b[2]);

    float d = 0.0f;
    float dx = cb_x - ca_x;
    dx -= box_x * roundf(dx / box_x);
    float sep = fabsf(dx) - ha_x - hb_x;
    if (sep > 0.0f) d += sep * sep;
    float dy = cb_y - ca_y;
    dy -= box_y * roundf(dy / box_y);
    sep = fabsf(dy) - ha_y - hb_y;
    if (sep > 0.0f) d += sep * sep;
    float dz = cb_z - ca_z;
    dz -= box_z * roundf(dz / box_z);
    sep = fabsf(dz) - ha_z - hb_z;
    if (sep > 0.0f) d += sep * sep;
    return d;
}

extern "C" __global__
void find_tiles_kernel(
    const int* block_bin,
    const float* aabb_min,
    const float* aabb_max,
    int num_blocks,
    int block_width,
    float build_radius_sq,
    float box_x, float box_y, float box_z,
    int nx, int ny, int nz,
    int num_unique_bins,
    const int* bin_block_indices,
    const int* bin_block_offsets,
    const int* bin_coords_flat,
    int max_self_tiles,
    int max_cross_tiles,
    int* self_tile_indices_out,
    int* cross_tiles_i_out,
    int* cross_tiles_j_out,
    float* cross_tiles_shift_out,
    int* counters
) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int global_warp = (blockIdx.x * blockDim.x + tid) / 32;
    int total_warps = gridDim.x * (blockDim.x / 32);

    for (int block_x = global_warp; block_x < num_blocks; block_x += total_warps) {
        int bx_x = block_bin[block_x * 3 + 0];
        int bx_y = block_bin[block_x * 3 + 1];
        int bx_z = block_bin[block_x * 3 + 2];

        int self_slot = -1;
        if (lane == 0)
            self_slot = atomicAdd(&counters[0], 1);
        self_slot = __shfl_sync(0xffffffff, self_slot, 0);
        if (self_slot < max_self_tiles && lane == 0)
            self_tile_indices_out[self_slot] = block_x;

        for (int nb = 0; nb < 14; nb++) {
            int dx, dy, dz;
            switch (nb) {
                case 0: dx = 0; dy = 0; dz = 0; break;
                case 1: dx = 1; dy = 0; dz = 0; break;
                case 2: dx = 0; dy = 1; dz = 0; break;
                case 3: dx = 0; dy = 0; dz = 1; break;
                case 4: dx = 1; dy = 1; dz = 0; break;
                case 5: dx = 1; dy = 0; dz = 1; break;
                case 6: dx = 0; dy = 1; dz = 1; break;
                case 7: dx = 1; dy = -1; dz = 0; break;
                case 8: dx = 1; dy = 0; dz = -1; break;
                case 9: dx = 0; dy = 1; dz = -1; break;
                case 10: dx = 1; dy = 1; dz = 1; break;
                case 11: dx = 1; dy = 1; dz = -1; break;
                case 12: dx = 1; dy = -1; dz = 1; break;
                case 13: dx = 1; dy = -1; dz = -1; break;
            }

            int nbx = (bx_x + dx) % nx;
            if (nbx < 0) nbx += nx;
            int nby = (bx_y + dy) % ny;
            if (nby < 0) nby += ny;
            int nbz = (bx_z + dz) % nz;
            if (nbz < 0) nbz += nz;

            int nbin_idx = -1;
            for (int bi = lane; bi < num_unique_bins; bi += 32) {
                if (bin_coords_flat[bi * 3 + 0] == nbx &&
                    bin_coords_flat[bi * 3 + 1] == nby &&
                    bin_coords_flat[bi * 3 + 2] == nbz) {
                    nbin_idx = bi;
                }
            }
            for (int off = 16; off > 0; off /= 2) {
                int other = __shfl_down_sync(0xffffffff, nbin_idx, off);
                if (other >= 0) nbin_idx = other;
            }
            nbin_idx = __shfl_sync(0xffffffff, nbin_idx, 0);
            if (nbin_idx < 0) continue;

            int raw_x = bx_x + dx;
            int raw_y = bx_y + dy;
            int raw_z = bx_z + dz;
            int shift_reps_x = raw_x / nx - (raw_x % nx < 0 ? 1 : 0);
            int shift_reps_y = raw_y / ny - (raw_y % ny < 0 ? 1 : 0);
            int shift_reps_z = raw_z / nz - (raw_z % nz < 0 ? 1 : 0);
            float shift_x = (float)shift_reps_x * box_x;
            float shift_y = (float)shift_reps_y * box_y;
            float shift_z = (float)shift_reps_z * box_z;

            int nb_start = bin_block_offsets[nbin_idx];
            int nb_end = bin_block_offsets[nbin_idx + 1];
            int nb_count = nb_end - nb_start;

            int j_start;
            if (dx == 0 && dy == 0 && dz == 0) {
                int my_pos = -1;
                for (int k = 0; k < nb_count; k++) {
                    if (bin_block_indices[nb_start + k] == block_x) {
                        my_pos = k;
                        break;
                    }
                }
                my_pos = __shfl_sync(0xffffffff, my_pos, 0);
                j_start = my_pos + 1;
            } else {
                j_start = 0;
            }

            for (int j = j_start; j < nb_count; j += 32) {
                int j_idx = j + lane;
                int pass = 0;
                int block_y = -1;
                if (j_idx < nb_count) {
                    block_y = bin_block_indices[nb_start + j_idx];
                    float dsq = block_dist_sq(
                        &aabb_min[block_x * 3], &aabb_max[block_x * 3],
                        &aabb_min[block_y * 3], &aabb_max[block_y * 3],
                        box_x, box_y, box_z
                    );
                    pass = (dsq <= build_radius_sq) ? 1 : 0;
                }
                unsigned int ballot = __ballot_sync(0xffffffff, pass);

                int base_cross = -1;
                int num_pass = __popc(ballot);
                if (lane == 0 && num_pass > 0)
                    base_cross = atomicAdd(&counters[1], num_pass);
                base_cross = __shfl_sync(0xffffffff, base_cross, 0);

                if (pass && base_cross >= 0) {
                    int pos_in_ballot = __popc(ballot & ((1u << lane) - 1));
                    int my_slot = base_cross + pos_in_ballot;
                    if (my_slot < max_cross_tiles) {
                        cross_tiles_i_out[my_slot] = block_x;
                        cross_tiles_j_out[my_slot] = block_y;
                        cross_tiles_shift_out[my_slot * 3 + 0] = shift_x;
                        cross_tiles_shift_out[my_slot * 3 + 1] = shift_y;
                        cross_tiles_shift_out[my_slot * 3 + 2] = shift_z;
                    }
                }
            }
        }
    }
}
"""


_BUILD_ATOM_MAP_KERNEL = r'''
extern "C" __global__
void build_atom_map_kernel(
    const int* block_atoms,
    const int num_blocks,
    const int W,
    int* atom_to_block,
    int* atom_to_slot
) {
    int block_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_index >= num_blocks) return;
    for (int slot = 0; slot < W; slot++) {
        int atom_id = block_atoms[block_index * W + slot];
        if (atom_id >= 0) {
            atom_to_block[atom_id] = block_index;
            atom_to_slot[atom_id] = slot;
        }
    }
}
'''


def _compile_gpu_kernels():
    wrap = cp.RawKernel(_PBC_WRAP_KERNEL, 'pbc_wrap_kernel')
    morton = cp.RawKernel(_MORTON_ENCODE_KERNEL, 'morton_encode_kernel')
    cut = cp.RawKernel(_BLOCK_CUT_KERNEL, 'block_cut_kernel')
    aabb = cp.RawKernel(_COMPUTE_AABB_KERNEL, 'compute_aabb_kernel')
    find = cp.RawKernel(_FIND_TILES_KERNEL, 'find_tiles_kernel')
    atom_map = cp.RawKernel(_BUILD_ATOM_MAP_KERNEL, 'build_atom_map_kernel')
    return wrap, morton, cut, aabb, find, atom_map


@njit(cache=True)
def _build_atom_to_block_slot(block_atoms, atom_to_block, atom_to_slot, num_blocks):
    for block_index in range(num_blocks):
        for slot in range(W):
            atom_id = block_atoms[block_index, slot]
            if atom_id >= 0:
                atom_to_block[atom_id] = block_index
                atom_to_slot[atom_id] = slot

@njit(cache=True)
def _build_masks_numba(
    block_atoms, atom_to_block, atom_to_slot,
    exclusion_offset, exclusion_neighbors, exclusion_scale,
    self_tile_indices, cross_tiles_i, cross_tiles_j,
    self_exclusion_masks, self_scaling_masks,
    cross_exclusion_masks, cross_scaling_masks,
    num_self, num_cross,
):
    num_particles = len(atom_to_block)

    excl_offset = exclusion_offset
    excl_neighbors = exclusion_neighbors
    excl_scale = exclusion_scale

    reverse_offset = np.zeros(num_particles + 1, dtype=np.int32)
    for atom_a in range(num_particles):
        start = excl_offset[atom_a]
        end = excl_offset[atom_a + 1]
        for k in range(start, end):
            neighbor = excl_neighbors[k]
            reverse_offset[neighbor + 1] += 1
    for i in range(num_particles):
        reverse_offset[i + 1] += reverse_offset[i]

    total_reverse = reverse_offset[num_particles]
    reverse_neighbors = np.empty(total_reverse, dtype=np.int32)
    reverse_scale = np.empty(total_reverse, dtype=np.float32)
    temp_offset = reverse_offset.copy()

    for atom_a in range(num_particles):
        start = excl_offset[atom_a]
        end = excl_offset[atom_a + 1]
        for k in range(start, end):
            neighbor = excl_neighbors[k]
            scale = excl_scale[k]
            pos = temp_offset[neighbor]
            reverse_neighbors[pos] = atom_a
            reverse_scale[pos] = scale
            temp_offset[neighbor] += 1

    for tile_idx in range(num_self):
        block_k = self_tile_indices[tile_idx]
        for a in range(W):
            atom_a = block_atoms[block_k, a]
            if atom_a < 0:
                continue

            mask = np.uint32(1 << a)
            for b in range(a):
                mask |= np.uint32(1 << b)

            start = excl_offset[atom_a]
            end = excl_offset[atom_a + 1]
            for k in range(start, end):
                neighbor = excl_neighbors[k]
                if atom_to_block[neighbor] == block_k:
                    slot_b = atom_to_slot[neighbor]
                    if slot_b >= 0:
                        scale = excl_scale[k]
                        if scale == 0.0:
                            mask |= np.uint32(1 << slot_b)
                        else:
                            self_scaling_masks[tile_idx, a] |= np.uint32(1 << slot_b)

            rstart = reverse_offset[atom_a]
            rend = reverse_offset[atom_a + 1]
            for k in range(rstart, rend):
                neighbor = reverse_neighbors[k]
                if atom_to_block[neighbor] == block_k:
                    slot_b = atom_to_slot[neighbor]
                    if slot_b >= 0:
                        scale = reverse_scale[k]
                        if scale == 0.0:
                            mask |= np.uint32(1 << slot_b)
                        else:
                            self_scaling_masks[tile_idx, a] |= np.uint32(1 << slot_b)

            self_exclusion_masks[tile_idx, a] = mask

    for tile_idx in range(num_cross):
        bi = cross_tiles_i[tile_idx]
        bj = cross_tiles_j[tile_idx]
        for a in range(W):
            atom_a = block_atoms[bi, a]
            if atom_a < 0:
                continue

            mask = np.uint32(0)
            scale_mask = np.uint32(0)

            start = excl_offset[atom_a]
            end = excl_offset[atom_a + 1]
            for k in range(start, end):
                neighbor = excl_neighbors[k]
                if atom_to_block[neighbor] == bj:
                    slot_b = atom_to_slot[neighbor]
                    if slot_b >= 0:
                        scale = excl_scale[k]
                        if scale == 0.0:
                            mask |= np.uint32(1 << slot_b)
                        else:
                            scale_mask |= np.uint32(1 << slot_b)

            rstart = reverse_offset[atom_a]
            rend = reverse_offset[atom_a + 1]
            for k in range(rstart, rend):
                neighbor = reverse_neighbors[k]
                if atom_to_block[neighbor] == bj:
                    slot_b = atom_to_slot[neighbor]
                    if slot_b >= 0:
                        scale = reverse_scale[k]
                        if scale == 0.0:
                            mask |= np.uint32(1 << slot_b)
                        else:
                            scale_mask |= np.uint32(1 << slot_b)

            cross_exclusion_masks[tile_idx, a] = mask
            cross_scaling_masks[tile_idx, a] = scale_mask


class TileList:

    def __init__(self, cutoff: float, skin: float = 2.0):
        self.cutoff = cutoff
        self.skin = skin
        self.build_radius = cutoff + skin
        self._is_initialized = False

        self.block_atoms = None
        self.block_bin = None
        self.num_blocks = 0

        self.self_tile_indices = None
        self.num_self = 0

        self.cross_tiles_i = None
        self.cross_tiles_j = None
        self.cross_tiles_shift = None
        self.num_cross = 0

        self.self_exclusion_masks = None
        self.self_scaling_masks = None
        self.cross_exclusion_masks = None
        self.cross_scaling_masks = None

        self.d_block_atoms = None
        self.d_atom_to_block = None
        self.d_atom_to_slot = None
        self.d_self_tile_indices = None
        self.d_self_exclusion_masks = None
        self.d_self_scaling_masks = None
        self.d_cross_tiles_i = None
        self.d_cross_tiles_j = None
        self.d_cross_tiles_shift = None
        self.d_cross_exclusion_masks = None
        self.d_cross_scaling_masks = None
        self.d_positions_at_rebuild = None

        self._gpu_wrap_kernel = None
        self._gpu_morton_kernel = None
        self._gpu_cut_kernel = None
        self._gpu_aabb_kernel = None
        self._gpu_find_tiles_kernel = None
        self._gpu_atom_map_kernel = None

        self._max_self_tiles = 0
        self._max_cross_tiles = 0
        self._d_self_tile_buf = None
        self._d_cross_i_buf = None
        self._d_cross_j_buf = None
        self._d_cross_shift_buf = None
        self._d_counters = None

        self._d_pbc_matrix = None
        self._d_pbc_inv = None

    @property
    def num_interactions(self):
        return self.num_self + self.num_cross

    @property
    def tile_particles(self):
        return self.block_atoms

    def rebuild(self, positions, topology, pbc_matrix, pbc_inv):
        num_particles = topology.num_particles
        if num_particles == 0:
            self._init_empty()
            return

        pbc_matrix_2d = np.asarray(pbc_matrix).reshape(3, 3)
        pbc_inv_2d = np.asarray(pbc_inv).reshape(3, 3)
        box_lengths = np.abs(np.diag(pbc_matrix_2d))
        cell_size = self.build_radius
        nx = max(1, int(np.floor(box_lengths[0] / cell_size)))
        ny = max(1, int(np.floor(box_lengths[1] / cell_size)))
        nz = max(1, int(np.floor(box_lengths[2] / cell_size)))

        self._ensure_gpu_kernels()

        if isinstance(positions, cp.ndarray):
            d_positions_flat = positions.ravel().astype(np.float32)
        else:
            d_positions_flat = cp.asarray(positions, dtype=np.float32).ravel()

        if self._d_pbc_matrix is None:
            self._d_pbc_matrix = cp.asarray(
                np.ascontiguousarray(pbc_matrix_2d, dtype=np.float32).ravel()
            )
            self._d_pbc_inv = cp.asarray(
                np.ascontiguousarray(pbc_inv_2d, dtype=np.float32).ravel()
            )

        self._rebuild_gpu(d_positions_flat, topology, pbc_matrix_2d, pbc_inv_2d,
                          box_lengths, cell_size, nx, ny, nz, num_particles)

    def _ensure_gpu_kernels(self):
        if self._gpu_morton_kernel is not None:
            return
        (self._gpu_wrap_kernel, self._gpu_morton_kernel,
         self._gpu_cut_kernel, self._gpu_aabb_kernel,
         self._gpu_find_tiles_kernel,
         self._gpu_atom_map_kernel) = _compile_gpu_kernels()

    def _ensure_tile_buffers(self):
        max_self = max(self.num_blocks, self._max_self_tiles)
        max_cross = max(self.num_blocks * 200, self._max_cross_tiles)
        if max_cross > 5000000:
            max_cross = 5000000

        if max_self > self._max_self_tiles:
            self._max_self_tiles = max_self
            self._d_self_tile_buf = cp.zeros(max_self, dtype=np.int32)
        if max_cross > self._max_cross_tiles:
            self._max_cross_tiles = max_cross
            self._d_cross_i_buf = cp.zeros(max_cross, dtype=np.int32)
            self._d_cross_j_buf = cp.zeros(max_cross, dtype=np.int32)
            self._d_cross_shift_buf = cp.zeros(max_cross * 3, dtype=np.float32)
        if self._d_counters is None:
            self._d_counters = cp.zeros(2, dtype=np.int32)

    def _rebuild_gpu(self, d_positions_flat, topology, pbc_matrix_2d, pbc_inv_2d,
                     box_lengths, cell_size, nx, ny, nz, num_particles):

        tpb = 256

        d_wrapped = d_positions_flat.copy()
        self._gpu_wrap_kernel(
            ((num_particles + tpb - 1) // tpb,), (tpb,),
            (d_wrapped, self._d_pbc_matrix, self._d_pbc_inv, np.int32(num_particles)),
        )

        d_morton_codes = cp.empty(num_particles, dtype=np.uint64)
        d_bin_coords = cp.empty(num_particles * 3, dtype=np.int32)

        self._gpu_morton_kernel(
            ((num_particles + tpb - 1) // tpb,), (tpb,),
            (d_wrapped, self._d_pbc_matrix, self._d_pbc_inv,
             np.int32(num_particles), np.float32(cell_size),
             np.int32(nx), np.int32(ny), np.int32(nz),
             d_morton_codes, d_bin_coords),
        )

        d_sorted_indices = cp.argsort(d_morton_codes).astype(np.int32)

        sorted_bin_x = cp.asnumpy(d_bin_coords[d_sorted_indices * 3 + 0])
        sorted_bin_y = cp.asnumpy(d_bin_coords[d_sorted_indices * 3 + 1])
        sorted_bin_z = cp.asnumpy(d_bin_coords[d_sorted_indices * 3 + 2])

        change_mask = (sorted_bin_x[1:] != sorted_bin_x[:-1]) | \
                      (sorted_bin_y[1:] != sorted_bin_y[:-1]) | \
                      (sorted_bin_z[1:] != sorted_bin_z[:-1])

        bin_end = np.concatenate([np.where(change_mask)[0] + 1, [num_particles]])
        bin_start = np.concatenate([[0], bin_end[:-1]])
        num_unique_bins = len(bin_start)

        bin_coord = np.empty(num_unique_bins * 3, dtype=np.int32)
        bin_num_blocks = np.empty(num_unique_bins, dtype=np.int32)
        total_blocks = 0
        for bi in range(num_unique_bins):
            s = bin_start[bi]
            bin_coord[bi * 3 + 0] = sorted_bin_x[s]
            bin_coord[bi * 3 + 1] = sorted_bin_y[s]
            bin_coord[bi * 3 + 2] = sorted_bin_z[s]
            cnt = int(bin_end[bi]) - int(s)
            nb = cnt // W + (1 if cnt % W > 0 else 0)
            bin_num_blocks[bi] = nb
            total_blocks += nb

        bin_block_offsets = np.zeros(num_unique_bins, dtype=np.int32)
        np.cumsum(bin_num_blocks[:-1], out=bin_block_offsets[1:])

        d_block_atoms = cp.full(total_blocks * W, -1, dtype=np.int32)
        d_block_bin = cp.empty(total_blocks * 3, dtype=np.int32)

        self._gpu_cut_kernel(
            ((num_unique_bins + tpb - 1) // tpb,), (tpb,),
            (d_sorted_indices,
             np.int32(num_particles), np.int32(num_unique_bins),
             cp.asarray(bin_start.astype(np.int32)),
             cp.asarray(bin_block_offsets.astype(np.int32)),
             cp.asarray(bin_num_blocks.astype(np.int32)),
             cp.asarray(bin_coord.astype(np.int32)),
             d_block_atoms, d_block_bin, np.int32(W)),
        )

        self.block_atoms = cp.asnumpy(d_block_atoms).reshape(-1, W)
        self.block_bin = cp.asnumpy(d_block_bin).reshape(-1, 3)
        self.num_blocks = total_blocks
        self.d_block_atoms = d_block_atoms

        self.d_atom_to_block = cp.full(num_particles, -1, dtype=np.int32)
        self.d_atom_to_slot = cp.full(num_particles, -1, dtype=np.int32)
        tpb_atom = 256
        self._gpu_atom_map_kernel(
            ((total_blocks + tpb_atom - 1) // tpb_atom,), (tpb_atom,),
            (d_block_atoms, np.int32(total_blocks), np.int32(W),
             self.d_atom_to_block, self.d_atom_to_slot),
        )

        d_aabb_min = cp.empty(total_blocks * 3, dtype=np.float32)
        d_aabb_max = cp.empty(total_blocks * 3, dtype=np.float32)
        self._gpu_aabb_kernel(
            ((total_blocks + tpb - 1) // tpb,), (tpb,),
            (d_wrapped, d_block_atoms,
             np.int32(total_blocks), np.int32(W),
             d_aabb_min, d_aabb_max),
        )

        bin_block_indices = np.arange(total_blocks, dtype=np.int32)
        bin_block_offsets_ext = np.empty(num_unique_bins + 1, dtype=np.int32)
        bin_block_offsets_ext[:num_unique_bins] = bin_block_offsets
        bin_block_offsets_ext[num_unique_bins] = total_blocks

        d_bin_block_indices = cp.asarray(bin_block_indices)
        d_bin_block_offsets_ext = cp.asarray(bin_block_offsets_ext)
        d_bin_coord = cp.asarray(bin_coord.astype(np.int32))

        self._ensure_tile_buffers()
        self._d_counters[0] = 0
        self._d_counters[1] = 0

        num_sm = cp.cuda.runtime.getDeviceProperties(0)['multiProcessorCount']
        self._gpu_find_tiles_kernel(
            (4 * num_sm,), (tpb,),
            (d_block_bin, d_aabb_min, d_aabb_max,
             np.int32(total_blocks), np.int32(W),
             np.float32(self.build_radius ** 2),
             np.float32(box_lengths[0]), np.float32(box_lengths[1]), np.float32(box_lengths[2]),
             np.int32(nx), np.int32(ny), np.int32(nz),
             np.int32(num_unique_bins),
              d_bin_block_indices, d_bin_block_offsets_ext, d_bin_coord,
             np.int32(self._max_self_tiles),
             np.int32(self._max_cross_tiles),
             self._d_self_tile_buf,
             self._d_cross_i_buf, self._d_cross_j_buf, self._d_cross_shift_buf,
             self._d_counters),
        )

        counters = cp.asnumpy(self._d_counters)
        self.num_self = int(counters[0])
        self.num_cross = int(counters[1])

        self.self_tile_indices = cp.asnumpy(self._d_self_tile_buf[:self.num_self]).copy()
        self.cross_tiles_i = cp.asnumpy(self._d_cross_i_buf[:self.num_cross]).copy()
        self.cross_tiles_j = cp.asnumpy(self._d_cross_j_buf[:self.num_cross]).copy()
        self.cross_tiles_shift = cp.asnumpy(self._d_cross_shift_buf[:self.num_cross * 3]).reshape(-1, 3).copy()

        self._build_exclusion_masks(topology)
        self._upload_to_device_gpu()

        self.d_positions_at_rebuild = d_wrapped
        self._is_initialized = True

    def _build_exclusion_masks(self, topology):
        self.self_exclusion_masks = np.zeros((self.num_self, W), dtype=np.uint32)
        self.self_scaling_masks = np.zeros((self.num_self, W), dtype=np.uint32)
        self.cross_exclusion_masks = np.zeros((self.num_cross, W), dtype=np.uint32)
        self.cross_scaling_masks = np.zeros((self.num_cross, W), dtype=np.uint32)

        if self.num_blocks == 0:
            return

        num_particles = topology.num_particles
        exclusion_offset = np.ascontiguousarray(topology.exclusion_offset, dtype=np.int32)
        exclusion_neighbors = np.ascontiguousarray(topology.exclusion_neighbors, dtype=np.int32)
        exclusion_scale = np.ascontiguousarray(topology.exclusion_scale, dtype=np.float32)
        block_atoms = np.ascontiguousarray(self.block_atoms, dtype=np.int32)
        self_tile_indices = np.ascontiguousarray(self.self_tile_indices, dtype=np.int32)
        cross_tiles_i = np.ascontiguousarray(self.cross_tiles_i, dtype=np.int32)
        cross_tiles_j = np.ascontiguousarray(self.cross_tiles_j, dtype=np.int32)

        atom_to_block = np.full(num_particles, -1, dtype=np.int32)
        atom_to_slot = np.full(num_particles, -1, dtype=np.int32)
        _build_atom_to_block_slot(block_atoms, atom_to_block, atom_to_slot, self.num_blocks)

        _build_masks_numba(
            block_atoms, atom_to_block, atom_to_slot,
            exclusion_offset, exclusion_neighbors, exclusion_scale,
            self_tile_indices, cross_tiles_i, cross_tiles_j,
            self.self_exclusion_masks, self.self_scaling_masks,
            self.cross_exclusion_masks, self.cross_scaling_masks,
            self.num_self, self.num_cross,
        )

    def _upload_to_device_gpu(self):
        self.d_self_tile_indices = cp.asarray(np.ascontiguousarray(self.self_tile_indices))
        self.d_self_exclusion_masks = cp.asarray(np.ascontiguousarray(self.self_exclusion_masks.ravel()))
        self.d_self_scaling_masks = cp.asarray(np.ascontiguousarray(self.self_scaling_masks.ravel()))
        self.d_cross_tiles_i = cp.asarray(np.ascontiguousarray(self.cross_tiles_i))
        self.d_cross_tiles_j = cp.asarray(np.ascontiguousarray(self.cross_tiles_j))
        self.d_cross_tiles_shift = cp.asarray(np.ascontiguousarray(self.cross_tiles_shift.ravel()))
        self.d_cross_exclusion_masks = cp.asarray(np.ascontiguousarray(self.cross_exclusion_masks.ravel()))
        self.d_cross_scaling_masks = cp.asarray(np.ascontiguousarray(self.cross_scaling_masks.ravel()))

    def _init_empty(self):
        self.block_atoms = np.empty((0, W), dtype=np.int32)
        self.block_bin = np.empty((0, 3), dtype=np.int32)
        self.num_blocks = 0
        self.self_tile_indices = np.empty(0, dtype=np.int32)
        self.num_self = 0
        self.cross_tiles_i = np.empty(0, dtype=np.int32)
        self.cross_tiles_j = np.empty(0, dtype=np.int32)
        self.cross_tiles_shift = np.empty((0, 3), dtype=np.float32)
        self.num_cross = 0
        self.self_exclusion_masks = np.empty((0, W), dtype=np.uint32)
        self.self_scaling_masks = np.empty((0, W), dtype=np.uint32)
        self.cross_exclusion_masks = np.empty((0, W), dtype=np.uint32)
        self.cross_scaling_masks = np.empty((0, W), dtype=np.uint32)
        self.d_block_atoms = cp.asarray(np.empty(0, dtype=np.int32))
        self.d_atom_to_block = cp.full(0, -1, dtype=np.int32)
        self.d_atom_to_slot = cp.full(0, -1, dtype=np.int32)
        self.d_self_tile_indices = cp.asarray(np.empty(0, dtype=np.int32))
        self.d_self_exclusion_masks = cp.asarray(np.empty(0, dtype=np.uint32))
        self.d_self_scaling_masks = cp.asarray(np.empty(0, dtype=np.uint32))
        self.d_cross_tiles_i = cp.asarray(np.empty(0, dtype=np.int32))
        self.d_cross_tiles_j = cp.asarray(np.empty(0, dtype=np.int32))
        self.d_cross_tiles_shift = cp.asarray(np.empty(0, dtype=np.float32))
        self.d_cross_exclusion_masks = cp.asarray(np.empty(0, dtype=np.uint32))
        self.d_cross_scaling_masks = cp.asarray(np.empty(0, dtype=np.uint32))
        self.d_positions_at_rebuild = cp.zeros(0, dtype=np.float32)
        self._is_initialized = True

    def check_rebuild(self, positions) -> bool:
        if not self._is_initialized:
            return True
        if isinstance(positions, cp.ndarray):
            d_pos = positions.ravel().astype(np.float32)
        else:
            d_pos = cp.asarray(positions, dtype=np.float32).ravel()
        diff = cp.abs(d_pos - self.d_positions_at_rebuild)
        max_disp = float(cp.max(diff))
        return max_disp > self.skin / 2
