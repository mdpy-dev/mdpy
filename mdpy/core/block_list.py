from __future__ import annotations

import numpy as np
import cupy as cp
from mdpy import env

W = 32
NUM_ATOMS_SENTINEL = 0x7FFFFFFF

_MORTON_SPLIT_FUNC = r"""
__device__ unsigned long long morton_split(unsigned int v) {
    v = v & 0x000003FFu;
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v << 8))  & 0x0300F00Fu;
    v = (v | (v << 4))  & 0x030C30C3u;
    v = (v | (v << 2))  & 0x09249249u;
    return (unsigned long long)v;
}
"""

_CELL_MORTON_KERNEL = _MORTON_SPLIT_FUNC + r"""
extern "C" __global__
void cell_morton_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    const float* __restrict__ pbc_matrix,
    const float* __restrict__ pbc_inv,
    int number_particles,
    int nc_x, int nc_y, int nc_z,
    unsigned long long* __restrict__ sort_keys,
    int* __restrict__ cell_indices
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= number_particles) return;

    float px = pos_x[i], py = pos_y[i], pz = pos_z[i];
    float fx = px*pbc_inv[0] + py*pbc_inv[3] + pz*pbc_inv[6];
    float fy = px*pbc_inv[1] + py*pbc_inv[4] + pz*pbc_inv[7];
    float fz = px*pbc_inv[2] + py*pbc_inv[5] + pz*pbc_inv[8];
    fx -= floorf(fx); fy -= floorf(fy); fz -= floorf(fz);

    int cx = min((int)(fx * nc_x), nc_x - 1);
    int cy = min((int)(fy * nc_y), nc_y - 1);
    int cz = min((int)(fz * nc_z), nc_z - 1);
    cell_indices[i] = cx + cy * nc_x + cz * nc_x * nc_y;

    float lfx = fx * nc_x - cx;
    float lfy = fy * nc_y - cy;
    float lfz = fz * nc_z - cz;

    unsigned int lx = min((unsigned int)(lfx * 1024.f), 1023u);
    unsigned int ly = min((unsigned int)(lfy * 1024.f), 1023u);
    unsigned int lz = min((unsigned int)(lfz * 1024.f), 1023u);
    unsigned int wm = morton_split(lx) | (morton_split(ly) << 1) | (morton_split(lz) << 2);

    sort_keys[i] = ((unsigned long long)cell_indices[i] << 30) | wm;
}
"""

_SCATTER_PADDED_KERNEL = r"""
extern "C" __global__
void scatter_padded_kernel(
    const int* __restrict__ cell_offset,
    const int* __restrict__ cell_offset_padded,
    const int* __restrict__ cell_indices_sorted,
    int number_particles,
    int* __restrict__ block_atoms_out
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= number_particles) return;
    int cell = cell_indices_sorted[i];
    int local = i - cell_offset[cell];
    int padded_pos = cell_offset_padded[cell] + local;
    block_atoms_out[padded_pos] = i;
}
"""

_COMPUTE_BLOCK_BOUNDS_KERNEL = r"""
extern "C" __global__
void compute_block_bounds_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    const int* __restrict__ block_atoms,
    int num_blocks,
    float* __restrict__ block_center_x_out,
    float* __restrict__ block_center_y_out,
    float* __restrict__ block_center_z_out,
    float* __restrict__ block_size_x_out,
    float* __restrict__ block_size_y_out,
    float* __restrict__ block_size_z_out
) {
    int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= num_blocks) return;
    float min_x = 1e30f, min_y = 1e30f, min_z = 1e30f;
    float max_x = -1e30f, max_y = -1e30f, max_z = -1e30f;
    for (int s = 0; s < 32; s++) {
        int a = block_atoms[bi * 32 + s];
        if (a < 0) continue;
        float x = pos_x[a], y = pos_y[a], z = pos_z[a];
        min_x = fminf(min_x, x); max_x = fmaxf(max_x, x);
        min_y = fminf(min_y, y); max_y = fmaxf(max_y, y);
        min_z = fminf(min_z, z); max_z = fmaxf(max_z, z);
    }
    block_center_x_out[bi] = 0.5f * (min_x + max_x);
    block_center_y_out[bi] = 0.5f * (min_y + max_y);
    block_center_z_out[bi] = 0.5f * (min_z + max_z);
    block_size_x_out[bi]   = 0.5f * (max_x - min_x);
    block_size_y_out[bi]   = 0.5f * (max_y - min_y);
    block_size_z_out[bi]   = 0.5f * (max_z - min_z);
}
"""

_BUILD_ATOM_MAP_KERNEL = r"""
extern "C" __global__
void build_atom_map_kernel(
    const int* __restrict__ block_atoms,
    const int num_blocks,
    const int W,
    int* __restrict__ atom_to_block,
    int* __restrict__ atom_to_slot
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
"""

_FIND_INTERACTING_BLOCKS_KERNEL = r"""
extern "C" __global__ __launch_bounds__(256, 3)
void find_interacting_blocks_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    const int* __restrict__ block_atoms,
    const float* __restrict__ block_center_x,
    const float* __restrict__ block_center_y,
    const float* __restrict__ block_center_z,
    const float* __restrict__ block_size_x,
    const float* __restrict__ block_size_y,
    const float* __restrict__ block_size_z,
    const int* __restrict__ cell_block_offset,
    const int* __restrict__ cell_block_count,
    const int* __restrict__ block_to_cell,
    int nc_x, int nc_y, int nc_z,
    int num_blocks, int num_particles,
    float build_radius_sq,
    const float* __restrict__ pbc_matrix,
    int* __restrict__ tiles_out,
    int* __restrict__ interacting_atoms_out,
    float* __restrict__ shift_x_out,
    float* __restrict__ shift_y_out,
    float* __restrict__ shift_z_out,
    int* __restrict__ interaction_count,
    int max_tiles
) {
    int tgx = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int global_warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (global_warp >= num_blocks) return;

    int bx = global_warp;
    int my_cell = block_to_cell[bx];

    int cz_idx = my_cell / (nc_x * nc_y);
    int cy_idx = (my_cell % (nc_x * nc_y)) / nc_x;
    int cx_idx = my_cell % nc_x;

    __shared__ int s_atom_idx[8][32];
    __shared__ float s_pos_x[8][32];
    __shared__ float s_pos_y[8][32];
    __shared__ float s_pos_z[8][32];
    __shared__ int s_buffer[8 * 256];
    int* my_buf = s_buffer + warp_in_block * 256;
    int nBuf = 0;

    {
        int gk = block_atoms[bx * 32 + tgx];
        int valid = (gk >= 0 && gk < num_particles) ? 1 : 0;
        s_atom_idx[warp_in_block][tgx] = valid ? gk : -1;
        if (valid) {
            s_pos_x[warp_in_block][tgx] = pos_x[gk];
            s_pos_y[warp_in_block][tgx] = pos_y[gk];
            s_pos_z[warp_in_block][tgx] = pos_z[gk];
        } else {
            s_pos_x[warp_in_block][tgx] = 0.0f;
            s_pos_y[warp_in_block][tgx] = 0.0f;
            s_pos_z[warp_in_block][tgx] = 0.0f;
        }
    }

    float mcx = block_center_x[bx], mcy = block_center_y[bx], mcz = block_center_z[bx];
    float msx = block_size_x[bx],   msy = block_size_y[bx],   msz = block_size_z[bx];

    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = (cx_idx + dx + nc_x) % nc_x;
                int ny = (cy_idx + dy + nc_y) % nc_y;
                int nz = (cz_idx + dz + nc_z) % nc_z;
                int nc = nx + ny * nc_x + nz * nc_x * nc_y;
                float sx = 0.0f, sy = 0.0f, sz = 0.0f;
                {
                    int raw_nx = cx_idx + dx;
                    if (raw_nx < 0) { sx -= pbc_matrix[0]; sy -= pbc_matrix[1]; sz -= pbc_matrix[2]; }
                    else if (raw_nx >= nc_x) { sx += pbc_matrix[0]; sy += pbc_matrix[1]; sz += pbc_matrix[2]; }
                }
                {
                    int raw_ny = cy_idx + dy;
                    if (raw_ny < 0) { sx -= pbc_matrix[3]; sy -= pbc_matrix[4]; sz -= pbc_matrix[5]; }
                    else if (raw_ny >= nc_y) { sx += pbc_matrix[3]; sy += pbc_matrix[4]; sz += pbc_matrix[5]; }
                }
                {
                    int raw_nz = cz_idx + dz;
                    if (raw_nz < 0) { sx -= pbc_matrix[6]; sy -= pbc_matrix[7]; sz -= pbc_matrix[8]; }
                    else if (raw_nz >= nc_z) { sx += pbc_matrix[6]; sy += pbc_matrix[7]; sz += pbc_matrix[8]; }
                }

                int b_start = cell_block_offset[nc];
                int b_count = cell_block_count[nc];
                bool is_self = (nc == my_cell);

                for (int iter = 0; iter < (b_count + 31) / 32; iter++) {
                    int bj_local = tgx + iter * 32;
                    bool include_block = false;
                    int bj = -1;

                    if (bj_local < b_count) {
                        bj = b_start + bj_local;
                        if (bj <= bx) {
                            // skip: Newton's 3rd law — global block ID ordering
                        } else {
                            float bcx2 = block_center_x[bj];
                            float bcy2 = block_center_y[bj];
                            float bcz2 = block_center_z[bj];
                            float bsx2 = block_size_x[bj];
                            float bsy2 = block_size_y[bj];
                            float bsz2 = block_size_z[bj];

                            float ddx = (bcx2 + sx) - mcx;
                            float ddy = (bcy2 + sy) - mcy;
                            float ddz = (bcz2 + sz) - mcz;
                            ddx = fmaxf(0.0f, fabsf(ddx) - msx - bsx2);
                            ddy = fmaxf(0.0f, fabsf(ddy) - msy - bsy2);
                            ddz = fmaxf(0.0f, fabsf(ddz) - msz - bsz2);
                            include_block = (ddx*ddx + ddy*ddy + ddz*ddz < build_radius_sq);
                        }
                    }

                    unsigned int include_flags = __ballot_sync(0xffffffff, include_block);

                    while (include_flags != 0) {
                        int fi = __ffs(include_flags) - 1;
                        include_flags &= include_flags - 1;
                        int target_bj = b_start + fi + iter * 32;

                        int gj = block_atoms[target_bj * 32 + tgx];
                        int interacts = 0;

                        if (gj >= 0 && gj < num_particles) {
                            float px_j = pos_x[gj];
                            float py_j = pos_y[gj];
                            float pz_j = pos_z[gj];
                            for (int k = 0; k < 32; k++) {
                                int gk = s_atom_idx[warp_in_block][k];
                                if (gk < 0) continue;
                                float ddx = (px_j + sx) - s_pos_x[warp_in_block][k];
                                float ddy = (py_j + sy) - s_pos_y[warp_in_block][k];
                                float ddz = (pz_j + sz) - s_pos_z[warp_in_block][k];
                                if (ddx*ddx + ddy*ddy + ddz*ddz <= build_radius_sq) {
                                    interacts = 1; break;
                                }
                            }
                        }

                        unsigned int ballot = __ballot_sync(0xffffffff, interacts);
                        int rank = __popc(ballot & ((1u << tgx) - 1));
                        if (interacts) my_buf[nBuf + rank] = gj;
                        nBuf += __popc(ballot);

                        while (nBuf >= 32) {
                            int ti = 0;
                            if (tgx == 0) ti = atomicAdd(interaction_count, 1);
                            ti = __shfl_sync(0xffffffff, ti, 0);
                            if (ti < max_tiles) {
                                if (tgx < 1) {
                                    tiles_out[ti] = bx;
                                    shift_x_out[ti] = sx;
                                    shift_y_out[ti] = sy;
                                    shift_z_out[ti] = sz;
                                }
                                interacting_atoms_out[ti * 32 + tgx] = my_buf[tgx];
                            }
                            for (int s = tgx; s < nBuf - 32; s += 32)
                                my_buf[s] = my_buf[s + 32];
                            nBuf -= 32;
                        }
                    }
                }
                if (nBuf > 0) {
                    int ti = 0;
                    if (tgx == 0) ti = atomicAdd(interaction_count, 1);
                    ti = __shfl_sync(0xffffffff, ti, 0);
                    if (ti < max_tiles) {
                        if (tgx < 1) {
                            tiles_out[ti] = bx;
                            shift_x_out[ti] = sx;
                            shift_y_out[ti] = sy;
                            shift_z_out[ti] = sz;
                        }
                        interacting_atoms_out[ti * 32 + tgx] = (tgx < nBuf) ? my_buf[tgx] : -1;
                    }
                    nBuf = 0;
                }
            }
        }
    }

    {
        int gj = block_atoms[bx * 32 + tgx];
        int interacts = (gj >= 0 && gj < num_particles) ? 1 : 0;
        unsigned int ballot = __ballot_sync(0xffffffff, interacts);
        int rank = __popc(ballot & ((1u << tgx) - 1));
        if (interacts) my_buf[nBuf + rank] = gj;
        nBuf += __popc(ballot);

        while (nBuf >= 32) {
            int ti = 0;
            if (tgx == 0) ti = atomicAdd(interaction_count, 1);
            ti = __shfl_sync(0xffffffff, ti, 0);
            if (ti < max_tiles) {
                if (tgx < 1) {
                    tiles_out[ti] = bx;
                    shift_x_out[ti] = 0.0f;
                    shift_y_out[ti] = 0.0f;
                    shift_z_out[ti] = 0.0f;
                }
                interacting_atoms_out[ti * 32 + tgx] = my_buf[tgx];
            }
            for (int s = tgx; s < nBuf - 32; s += 32)
                my_buf[s] = my_buf[s + 32];
            nBuf -= 32;
        }
    }

    if (nBuf > 0) {
        int ti = 0;
        if (tgx == 0) ti = atomicAdd(interaction_count, 1);
        ti = __shfl_sync(0xffffffff, ti, 0);
        if (ti < max_tiles) {
            if (tgx < 1) {
                tiles_out[ti] = bx;
                shift_x_out[ti] = 0.0f;
                shift_y_out[ti] = 0.0f;
                shift_z_out[ti] = 0.0f;
            }
            interacting_atoms_out[ti * 32 + tgx] =
                (tgx < nBuf) ? my_buf[tgx] : 0x7FFFFFFF;
        }
    }
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
    const float* __restrict__ exclusion_scale,
    const int* __restrict__ reverse_offset,
    const int num_particles,
    int* __restrict__ reverse_neighbors,
    float* __restrict__ reverse_scale,
    int* __restrict__ temp_offset
) {
    int atom_a = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom_a >= num_particles) return;
    int start = exclusion_offset[atom_a];
    int end = exclusion_offset[atom_a + 1];
    for (int k = start; k < end; k++) {
        int neighbor = exclusion_neighbors[k];
        float scale = exclusion_scale[k];
        int pos = atomicAdd(&temp_offset[neighbor], 1);
        reverse_neighbors[pos] = atom_a;
        reverse_scale[pos] = scale;
    }
}
"""

_BUILD_MASKS_KERNEL = r"""
extern "C" __global__
void build_masks_kernel(
    const int* __restrict__ tiles,
    const int* __restrict__ interacting_atoms,
    const int* __restrict__ block_atoms,
    const int* __restrict__ atom_to_block,
    const int* __restrict__ atom_to_slot,
    const int* __restrict__ exclusion_offset,
    const int* __restrict__ exclusion_neighbors,
    const float* __restrict__ exclusion_scale,
    const int* __restrict__ reverse_offset,
    const int* __restrict__ reverse_neighbors,
    const float* __restrict__ reverse_scale,
    int num_tiles,
    int num_particles,
    unsigned int* __restrict__ exclusion_masks_out,
    unsigned int* __restrict__ scaling_masks_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tiles * 32) return;

    int tile = idx / 32;
    int slot_j = idx % 32;
    int block_x = tiles[tile];
    int atom_j = interacting_atoms[tile * 32 + slot_j];

    unsigned int excl = 0, scale = 0;

    if (atom_j >= 0 && atom_j < num_particles) {
        int bj = atom_to_block[atom_j];
        int sj = atom_to_slot[atom_j];
        if (bj == block_x) {
            excl |= ((1u << (sj + 1)) - 1);
        }

        int s = exclusion_offset[atom_j];
        int e = exclusion_offset[atom_j + 1];
        for (int k = s; k < e; k++) {
            int nb = exclusion_neighbors[k];
            if (nb < 0 || nb >= num_particles) continue;
            if (atom_to_block[nb] == block_x) {
                int sn = atom_to_slot[nb];
                if (exclusion_scale[k] == 0.0f)
                    excl |= (1u << sn);
                else
                    scale |= (1u << sn);
            }
        }

        s = reverse_offset[atom_j];
        e = reverse_offset[atom_j + 1];
        for (int k = s; k < e; k++) {
            int nb = reverse_neighbors[k];
            if (nb < 0 || nb >= num_particles) continue;
            if (atom_to_block[nb] == block_x) {
                int sn = atom_to_slot[nb];
                if (reverse_scale[k] == 0.0f)
                    excl |= (1u << sn);
                else
                    scale |= (1u << sn);
            }
        }
    }

    exclusion_masks_out[idx] = excl;
    scaling_masks_out[idx] = scale;
}
"""

_CLASSIFY_TILES_KERNEL = r"""
extern "C" __global__
void classify_tiles_kernel(
    const unsigned int* __restrict__ excl_masks,
    const unsigned int* __restrict__ scale_masks,
    const int* __restrict__ tiles,
    const int* __restrict__ interacting_atoms,
    const float* __restrict__ shift_x,
    const float* __restrict__ shift_y,
    const float* __restrict__ shift_z,
    int num_tiles,
    int* __restrict__ excl_counter,
    int* __restrict__ main_counter,
    int* __restrict__ excl_tiles_out,
    int* __restrict__ excl_int_atoms_out,
    unsigned int* __restrict__ excl_masks_out,
    unsigned int* __restrict__ excl_scale_out,
    int* __restrict__ main_tiles_out,
    int* __restrict__ main_int_atoms_out,
    float* __restrict__ excl_shift_x_out,
    float* __restrict__ excl_shift_y_out,
    float* __restrict__ excl_shift_z_out,
    float* __restrict__ main_shift_x_out,
    float* __restrict__ main_shift_y_out,
    float* __restrict__ main_shift_z_out
) {
    int tile = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile >= num_tiles) return;

    bool has_mask = false;
    for (int i = 0; i < 32; i++) {
        if (excl_masks[tile * 32 + i] != 0 || scale_masks[tile * 32 + i] != 0) {
            has_mask = true;
            break;
        }
    }

    if (has_mask) {
        int idx = atomicAdd(excl_counter, 1);
        excl_tiles_out[idx] = tiles[tile];
        for (int i = 0; i < 32; i++) {
            excl_int_atoms_out[idx * 32 + i] = interacting_atoms[tile * 32 + i];
            excl_masks_out[idx * 32 + i] = excl_masks[tile * 32 + i];
            excl_scale_out[idx * 32 + i] = scale_masks[tile * 32 + i];
        }
        excl_shift_x_out[idx] = shift_x[tile];
        excl_shift_y_out[idx] = shift_y[tile];
        excl_shift_z_out[idx] = shift_z[tile];
    } else {
        int idx = atomicAdd(main_counter, 1);
        main_tiles_out[idx] = tiles[tile];
        for (int i = 0; i < 32; i++) {
            main_int_atoms_out[idx * 32 + i] = interacting_atoms[tile * 32 + i];
        }
        main_shift_x_out[idx] = shift_x[tile];
        main_shift_y_out[idx] = shift_y[tile];
        main_shift_z_out[idx] = shift_z[tile];
    }
}
"""

_CHECK_REBUILD_KERNEL = r"""
extern "C" __global__
void check_rebuild_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    const float* __restrict__ old_pos_x,
    const float* __restrict__ old_pos_y,
    const float* __restrict__ old_pos_z,
    int num_particles,
    float threshold_sq,
    const float* __restrict__ pbc_inv,
    const float* __restrict__ pbc_matrix,
    int* __restrict__ rebuild_flag
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    float dx = pos_x[idx] - old_pos_x[idx];
    float dy = pos_y[idx] - old_pos_y[idx];
    float dz = pos_z[idx] - old_pos_z[idx];
    float fx = dx*pbc_inv[0] + dy*pbc_inv[3] + dz*pbc_inv[6];
    float fy = dx*pbc_inv[1] + dy*pbc_inv[4] + dz*pbc_inv[7];
    float fz = dx*pbc_inv[2] + dy*pbc_inv[5] + dz*pbc_inv[8];
    fx -= roundf(fx); fy -= roundf(fy); fz -= roundf(fz);
    dx = fx*pbc_matrix[0] + fy*pbc_matrix[3] + fz*pbc_matrix[6];
    dy = fx*pbc_matrix[1] + fy*pbc_matrix[4] + fz*pbc_matrix[7];
    dz = fx*pbc_matrix[2] + fy*pbc_matrix[5] + fz*pbc_matrix[8];
    if (dx*dx + dy*dy + dz*dz > threshold_sq)
        rebuild_flag[0] = 1;
}
"""

_FUSED_COPY3_KERNEL = r"""
extern "C" __global__
void fused_copy3_kernel(
    const float* __restrict__ src0,
    const float* __restrict__ src1,
    const float* __restrict__ src2,
    int num_elements,
    float* __restrict__ dst0,
    float* __restrict__ dst1,
    float* __restrict__ dst2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    dst0[idx] = src0[idx];
    dst1[idx] = src1[idx];
    dst2[idx] = src2[idx];
}
"""


def _compile_gpu_kernels():
    return {
        "cell_morton": cp.RawKernel(_CELL_MORTON_KERNEL, "cell_morton_kernel"),
        "scatter_padded": cp.RawKernel(_SCATTER_PADDED_KERNEL, "scatter_padded_kernel"),
        "compute_bounds": cp.RawKernel(
            _COMPUTE_BLOCK_BOUNDS_KERNEL, "compute_block_bounds_kernel"
        ),
        "atom_map": cp.RawKernel(_BUILD_ATOM_MAP_KERNEL, "build_atom_map_kernel"),
        "find_interacting": cp.RawKernel(
            _FIND_INTERACTING_BLOCKS_KERNEL, "find_interacting_blocks_kernel"
        ),
        "rev_count": cp.RawKernel(_BUILD_REVERSE_COUNT_KERNEL, "build_reverse_count_kernel"),
        "rev_fill": cp.RawKernel(_FILL_REVERSE_KERNEL, "fill_reverse_kernel"),
        "build_masks": cp.RawKernel(_BUILD_MASKS_KERNEL, "build_masks_kernel"),
        "classify_tiles": cp.RawKernel(_CLASSIFY_TILES_KERNEL, "classify_tiles_kernel"),
        "check_rebuild": cp.RawKernel(_CHECK_REBUILD_KERNEL, "check_rebuild_kernel"),
        "fused_copy3": cp.RawKernel(_FUSED_COPY3_KERNEL, "fused_copy3_kernel"),
    }


class BlockList:

    def __init__(self, cutoff: float, skin: float = 1.0, rebuild_check_interval: int = 10):
        self.cutoff = cutoff
        self.skin = skin
        self.build_radius = cutoff + skin
        self._is_initialized = False
        self.rebuild_check_interval = rebuild_check_interval

        self.num_blocks = 0
        self.num_tiles = 0
        self.num_particles = 0
        self._max_tiles = 0

        self.nc_x = 0
        self.nc_y = 0
        self.nc_z = 0
        self.nc_total = 0

        self.d_block_atoms = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_block_center_x = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_block_center_y = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_block_center_z = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_block_size_x = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_block_size_y = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_block_size_z = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_atom_to_block = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_atom_to_slot = cp.empty(0, dtype=env.NUMPY_INT)

        self.d_tiles = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_interacting_atoms = cp.empty(0, dtype=env.NUMPY_INT)

        self.d_cell_block_offset = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_cell_block_count = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_block_to_cell = cp.empty(0, dtype=env.NUMPY_INT)

        self.d_raw_order = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_pdb_to_sorted = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_sorted_to_pdb = cp.empty(0, dtype=env.NUMPY_INT)

        self._d_tile_buf = cp.empty(0, dtype=env.NUMPY_INT)
        self._d_interacting_buf = cp.empty(0, dtype=env.NUMPY_INT)
        self._d_tile_shift_x_buf = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self._d_tile_shift_y_buf = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self._d_tile_shift_z_buf = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self._d_counters = cp.zeros(1, dtype=env.NUMPY_INT)

        self._d_pbc_matrix = None
        self._d_pbc_inv = None
        self._kernels = None

        self._d_excl_offset = None
        self._d_excl_neighbors = None
        self._d_excl_scale = None
        self._d_reverse_offset = None
        self._d_reverse_neighbors = None
        self._d_reverse_scale = None
        self._total_exclusion_pairs = 0

        self.d_exclusion_masks = cp.empty(0, dtype=np.uint32)
        self.d_scaling_masks = cp.empty(0, dtype=np.uint32)

        self.num_main_tiles = 0
        self.num_exclusion_tiles = 0
        self.d_main_tiles = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_main_interacting_atoms = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_excl_tiles = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_excl_interacting_atoms = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_excl_exclusion_masks = cp.empty(0, dtype=np.uint32)
        self.d_excl_scaling_masks = cp.empty(0, dtype=np.uint32)

        self.d_tile_shift_x = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_tile_shift_y = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_tile_shift_z = cp.empty(0, dtype=env.NUMPY_FLOAT)

        self._d_classify_excl_counter = cp.zeros(1, dtype=env.NUMPY_INT)
        self._d_classify_main_counter = cp.zeros(1, dtype=env.NUMPY_INT)
        self._d_classify_excl_tiles = cp.empty(0, dtype=env.NUMPY_INT)
        self._d_classify_excl_int_atoms = cp.empty(0, dtype=env.NUMPY_INT)
        self._d_classify_excl_masks = cp.empty(0, dtype=np.uint32)
        self._d_classify_excl_scale = cp.empty(0, dtype=np.uint32)
        self._d_classify_main_tiles = cp.empty(0, dtype=env.NUMPY_INT)
        self._d_classify_main_int_atoms = cp.empty(0, dtype=env.NUMPY_INT)
        self._d_classify_excl_shift_x = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self._d_classify_excl_shift_y = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self._d_classify_excl_shift_z = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self._d_classify_main_shift_x = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self._d_classify_main_shift_y = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self._d_classify_main_shift_z = cp.empty(0, dtype=env.NUMPY_FLOAT)

        self.d_main_shift_x = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_main_shift_y = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_main_shift_z = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_excl_shift_x = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_excl_shift_y = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_excl_shift_z = cp.empty(0, dtype=env.NUMPY_FLOAT)

        self._exclusion_masks_np = None
        self._scaling_masks_np = None

        self.d_rebuild_flag = cp.zeros(1, dtype=env.NUMPY_INT)
        self.d_positions_at_rebuild_x = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_positions_at_rebuild_y = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_positions_at_rebuild_z = cp.empty(0, dtype=env.NUMPY_FLOAT)

        self._block_atoms_np = None
        self._tiles_np = None
        self._interacting_atoms_np = None

    @property
    def block_atoms(self):
        if self._block_atoms_np is None and self.d_block_atoms.size > 0:
            self._block_atoms_np = cp.asnumpy(self.d_block_atoms).reshape(-1, W)
        return self._block_atoms_np

    @property
    def tiles(self):
        if self._tiles_np is None and self.d_tiles.size > 0:
            self._tiles_np = cp.asnumpy(self.d_tiles[: self.num_tiles])
        return self._tiles_np

    @property
    def interacting_atoms(self):
        if self._interacting_atoms_np is None and self.d_interacting_atoms.size > 0:
            self._interacting_atoms_np = cp.asnumpy(
                self.d_interacting_atoms[: self.num_tiles * W]
            ).reshape(-1, W)
        return self._interacting_atoms_np

    @property
    def exclusion_masks(self):
        if self._exclusion_masks_np is None and self.d_exclusion_masks.size > 0:
            self._exclusion_masks_np = cp.asnumpy(
                self.d_exclusion_masks[:self.num_tiles * W]
            ).reshape(-1, W)
        return self._exclusion_masks_np

    @property
    def scaling_masks(self):
        if self._scaling_masks_np is None and self.d_scaling_masks.size > 0:
            self._scaling_masks_np = cp.asnumpy(
                self.d_scaling_masks[:self.num_tiles * W]
            ).reshape(-1, W)
        return self._scaling_masks_np

    def _invalidate_caches(self):
        self._block_atoms_np = None
        self._tiles_np = None
        self._interacting_atoms_np = None
        self._exclusion_masks_np = None
        self._scaling_masks_np = None

    def _ensure_kernels(self):
        if self._kernels is not None:
            return
        self._kernels = _compile_gpu_kernels()

    def _upload_pbc(self, pbc_matrix, pbc_inv):
        if self._d_pbc_matrix is None:
            self._d_pbc_matrix = cp.asarray(
                np.ascontiguousarray(pbc_matrix, dtype=env.NUMPY_FLOAT).ravel()
            )
            self._d_pbc_inv = cp.asarray(
                np.ascontiguousarray(pbc_inv, dtype=env.NUMPY_FLOAT).ravel()
            )

    def _compute_cell_grid(self, pbc_matrix):
        pbc_2d = np.asarray(pbc_matrix).reshape(3, 3)
        a_vec = pbc_2d[0]
        b_vec = pbc_2d[1]
        c_vec = pbc_2d[2]
        box_a = float(np.linalg.norm(a_vec))
        box_b = float(np.linalg.norm(b_vec))
        box_c = float(np.linalg.norm(c_vec))
        cell_size = self.build_radius
        self.nc_x = max(1, int(box_a / cell_size))
        self.nc_y = max(1, int(box_b / cell_size))
        self.nc_z = max(1, int(box_c / cell_size))
        self.nc_total = self.nc_x * self.nc_y * self.nc_z

    def rebuild(self, positions, topology, pbc_matrix, pbc_inv):
        """Sort particles into cell-aligned blocks. Returns (pdb_to_sorted, None)."""
        N = topology.num_particles
        if N == 0:
            self._init_empty()
            return None, None

        self._ensure_kernels()
        self._invalidate_caches()
        self.num_particles = N
        tpb = 256

        self._compute_cell_grid(pbc_matrix)
        self._upload_pbc(pbc_matrix, pbc_inv)

        if isinstance(positions, tuple):
            pos_x = positions[0]
            pos_y = positions[1]
            pos_z = positions[2]
        else:
            data = cp.asarray(
                np.ascontiguousarray(positions.ravel(), dtype=env.NUMPY_FLOAT)
            )
            pos_x = data[0::3].copy()
            pos_y = data[1::3].copy()
            pos_z = data[2::3].copy()

        # K1: Fused cell-assign + within-cell morton
        sort_keys = cp.empty(N, dtype=np.uint64)
        cell_indices = cp.empty(N, dtype=env.NUMPY_INT)
        nm = (N + tpb - 1) // tpb
        self._kernels["cell_morton"](
            (nm,), (tpb,),
            (
                pos_x, pos_y, pos_z,
                self._d_pbc_matrix, self._d_pbc_inv,
                np.int32(N),
                np.int32(self.nc_x), np.int32(self.nc_y), np.int32(self.nc_z),
                sort_keys, cell_indices,
            ),
        )

        # Sort by (cell_index, within_cell_morton)
        sorted_indices = cp.argsort(sort_keys).astype(env.NUMPY_INT)
        self.d_raw_order = sorted_indices.copy()
        self.d_pdb_to_sorted = cp.empty(N, dtype=env.NUMPY_INT)
        self.d_pdb_to_sorted[sorted_indices] = cp.arange(N, dtype=env.NUMPY_INT)
        self.d_sorted_to_pdb = sorted_indices.copy()

        pos_x = pos_x[sorted_indices]
        pos_y = pos_y[sorted_indices]
        pos_z = pos_z[sorted_indices]
        cell_indices_sorted = cell_indices[sorted_indices]
        self._sorted_positions = (pos_x, pos_y, pos_z)

        # Cell count + offset (CuPy operations)
        cell_counts = cp.bincount(cell_indices_sorted, minlength=self.nc_total)
        cell_offset = cp.cumsum(
            cp.concatenate([cp.zeros(1, dtype=env.NUMPY_INT), cell_counts])
        ).astype(env.NUMPY_INT)
        cell_block_count = ((cell_counts + 31) // 32).astype(env.NUMPY_INT)
        cell_block_offset = cp.cumsum(
            cp.concatenate([cp.zeros(1, dtype=env.NUMPY_INT), cell_block_count])
        ).astype(env.NUMPY_INT)
        num_blocks = int(cell_block_count.sum())
        self.num_blocks = num_blocks
        total_padded = int(cell_block_offset[-1]) * W

        cell_offset_padded = cell_block_offset * W

        self.d_cell_block_offset = cell_block_offset
        self.d_cell_block_count = cell_block_count

        # K2: Scatter to padded layout
        block_atoms = cp.full(total_padded, -1, dtype=env.NUMPY_INT)
        self._kernels["scatter_padded"](
            (nm,), (tpb,),
            (
                cell_offset, cell_offset_padded, cell_indices_sorted,
                np.int32(N), block_atoms,
            ),
        )
        self.d_block_atoms = block_atoms

        # Block-to-cell mapping
        cell_block_count_np = cp.asnumpy(cell_block_count)
        cell_idx_np = np.arange(self.nc_total, dtype=env.NUMPY_INT)
        self.d_block_to_cell = cp.asarray(np.repeat(cell_idx_np, cell_block_count_np))

        # K3: Compute block AABB
        self.d_block_center_x = cp.empty(num_blocks, dtype=env.NUMPY_FLOAT)
        self.d_block_center_y = cp.empty(num_blocks, dtype=env.NUMPY_FLOAT)
        self.d_block_center_z = cp.empty(num_blocks, dtype=env.NUMPY_FLOAT)
        self.d_block_size_x = cp.empty(num_blocks, dtype=env.NUMPY_FLOAT)
        self.d_block_size_y = cp.empty(num_blocks, dtype=env.NUMPY_FLOAT)
        self.d_block_size_z = cp.empty(num_blocks, dtype=env.NUMPY_FLOAT)
        nb = (num_blocks + tpb - 1) // tpb
        self._kernels["compute_bounds"](
            (nb,), (tpb,),
            (
                pos_x, pos_y, pos_z, self.d_block_atoms,
                np.int32(num_blocks),
                self.d_block_center_x, self.d_block_center_y, self.d_block_center_z,
                self.d_block_size_x, self.d_block_size_y, self.d_block_size_z,
            ),
        )

        # K4: Build atom-to-block map
        self.d_atom_to_block = cp.full(N, -1, dtype=env.NUMPY_INT)
        self.d_atom_to_slot = cp.full(N, -1, dtype=env.NUMPY_INT)
        self._kernels["atom_map"](
            (nb,), (tpb,),
            (
                self.d_block_atoms, np.int32(num_blocks), np.int32(W),
                self.d_atom_to_block, self.d_atom_to_slot,
            ),
        )

        self._d_reverse_offset = None
        self._d_reverse_neighbors = None
        self._d_reverse_scale = None

        self._is_initialized = True

        self.d_positions_at_rebuild_x, self.d_positions_at_rebuild_y, self.d_positions_at_rebuild_z = self.fused_copy3(pos_x, pos_y, pos_z)
        self.d_rebuild_flag[0] = 0

        return self.d_pdb_to_sorted, None

    def build_tiles(self, topology, pbc_matrix):
        """Find interacting block pairs using cell-based neighbor search."""
        if self.num_particles == 0 or not hasattr(self, '_sorted_positions'):
            return
        self._invalidate_caches()

        pos_x, pos_y, pos_z = self._sorted_positions
        num_blocks = self.num_blocks
        build_radius_sq = self.build_radius ** 2

        max_tiles = max(num_blocks * 100, 10000)
        if self._d_tile_buf.size < max_tiles:
            self._d_tile_buf = cp.empty(max_tiles, dtype=env.NUMPY_INT)
            self._d_interacting_buf = cp.empty(max_tiles * W, dtype=env.NUMPY_INT)
            self._d_tile_shift_x_buf = cp.empty(max_tiles, dtype=env.NUMPY_FLOAT)
            self._d_tile_shift_y_buf = cp.empty(max_tiles, dtype=env.NUMPY_FLOAT)
            self._d_tile_shift_z_buf = cp.empty(max_tiles, dtype=env.NUMPY_FLOAT)
            self._max_tiles = max_tiles
        self._d_counters[0] = 0

        tpb = 256
        grid_blocks = max((num_blocks + 7) // 8, 1)
        self._kernels["find_interacting"](
            (grid_blocks,), (tpb,),
            (
                pos_x, pos_y, pos_z,
                self.d_block_atoms,
                self.d_block_center_x, self.d_block_center_y, self.d_block_center_z,
                self.d_block_size_x, self.d_block_size_y, self.d_block_size_z,
                self.d_cell_block_offset, self.d_cell_block_count,
                self.d_block_to_cell,
                np.int32(self.nc_x), np.int32(self.nc_y), np.int32(self.nc_z),
                np.int32(num_blocks), np.int32(self.num_particles),
                np.float32(build_radius_sq),
                self._d_pbc_matrix,
                self._d_tile_buf, self._d_interacting_buf,
                self._d_tile_shift_x_buf, self._d_tile_shift_y_buf, self._d_tile_shift_z_buf,
                self._d_counters, np.int32(max_tiles),
            ),
        )

        self.num_tiles = int(self._d_counters[0])
        self.d_tiles = self._d_tile_buf
        self.d_interacting_atoms = self._d_interacting_buf
        self.d_tile_shift_x = self._d_tile_shift_x_buf
        self.d_tile_shift_y = self._d_tile_shift_y_buf
        self.d_tile_shift_z = self._d_tile_shift_z_buf

        self._build_masks_gpu(topology)
        self._extract_exclusion_tiles()

    def fused_copy3(self, src0, src1, src2):
        N = src0.size
        self._ensure_kernels()
        tpb = 256
        grid = ((N + tpb - 1) // tpb,)
        dst0 = cp.empty(N, dtype=env.NUMPY_FLOAT)
        dst1 = cp.empty(N, dtype=env.NUMPY_FLOAT)
        dst2 = cp.empty(N, dtype=env.NUMPY_FLOAT)
        self._kernels["fused_copy3"](
            grid,
            (tpb,),
            (src0, src1, src2, np.int32(N), dst0, dst1, dst2),
        )
        return dst0, dst1, dst2

    def set_gpu_exclusion(self, d_offset, d_neighbors, d_scale):
        self._d_excl_offset = d_offset
        self._d_excl_neighbors = d_neighbors
        self._d_excl_scale = d_scale
        self._d_reverse_offset = None
        self._d_reverse_neighbors = None
        self._d_reverse_scale = None
        self._total_exclusion_pairs = int(d_neighbors.shape[0])

    def _upload_exclusion(self, topology):
        if self._d_excl_offset is not None:
            return
        self._d_excl_offset = cp.asarray(
            np.ascontiguousarray(topology.exclusion_offset, dtype=env.NUMPY_INT)
        )
        self._d_excl_neighbors = cp.asarray(
            np.ascontiguousarray(topology.exclusion_neighbors, dtype=env.NUMPY_INT)
        )
        self._d_excl_scale = cp.asarray(
            np.ascontiguousarray(topology.exclusion_scale, dtype=env.NUMPY_FLOAT)
        )

    def _build_masks_gpu(self, topology):
        if self.num_tiles == 0:
            self.d_exclusion_masks = cp.empty(0, dtype=np.uint32)
            self.d_scaling_masks = cp.empty(0, dtype=np.uint32)
            return

        self._upload_exclusion(topology)
        N = self.num_particles
        tpb = 256

        if self._d_reverse_offset is None:
            d_rev_offset = cp.zeros(N + 1, dtype=env.NUMPY_INT)
            n1 = (N + tpb - 1) // tpb
            self._kernels["rev_count"](
                (n1,),
                (tpb,),
                (
                    self._d_excl_offset,
                    self._d_excl_neighbors,
                    np.int32(N),
                    d_rev_offset,
                ),
            )

            d_rev_offset = cp.cumsum(d_rev_offset, dtype=env.NUMPY_INT).astype(env.NUMPY_INT)
            max_rev = (
                self._total_exclusion_pairs
                if self._total_exclusion_pairs > 0
                else int(d_rev_offset[-1])
            )
            d_rev_neighbors = cp.empty(max_rev, dtype=env.NUMPY_INT)
            d_rev_scale = cp.empty(max_rev, dtype=env.NUMPY_FLOAT)
            d_temp = d_rev_offset.copy()

            self._kernels["rev_fill"](
                (n1,),
                (tpb,),
                (
                    self._d_excl_offset,
                    self._d_excl_neighbors,
                    self._d_excl_scale,
                    d_rev_offset,
                    np.int32(N),
                    d_rev_neighbors,
                    d_rev_scale,
                    d_temp,
                ),
            )

            self._d_reverse_offset = d_rev_offset
            self._d_reverse_neighbors = d_rev_neighbors
            self._d_reverse_scale = d_rev_scale

        total_work = self.num_tiles * W
        grid = ((total_work + tpb - 1) // tpb,)
        self.d_exclusion_masks = cp.empty(total_work, dtype=np.uint32)
        self.d_scaling_masks = cp.empty(total_work, dtype=np.uint32)
        self._kernels["build_masks"](
            grid,
            (tpb,),
            (
                self.d_tiles,
                self.d_interacting_atoms,
                self.d_block_atoms,
                self.d_atom_to_block,
                self.d_atom_to_slot,
                self._d_excl_offset,
                self._d_excl_neighbors,
                self._d_excl_scale,
                self._d_reverse_offset,
                self._d_reverse_neighbors,
                self._d_reverse_scale,
                np.int32(self.num_tiles),
                np.int32(N),
                self.d_exclusion_masks,
                self.d_scaling_masks,
            ),
        )

    def _extract_exclusion_tiles(self):
        if self.num_tiles == 0:
            self.num_exclusion_tiles = 0
            self.num_main_tiles = 0
            self.d_excl_tiles = cp.empty(0, dtype=env.NUMPY_INT)
            self.d_excl_interacting_atoms = cp.empty(0, dtype=env.NUMPY_INT)
            self.d_excl_exclusion_masks = cp.empty(0, dtype=np.uint32)
            self.d_excl_scaling_masks = cp.empty(0, dtype=np.uint32)
            self.d_main_tiles = cp.empty(0, dtype=env.NUMPY_INT)
            self.d_main_interacting_atoms = cp.empty(0, dtype=env.NUMPY_INT)
            self.d_tile_shift_x = cp.empty(0, dtype=env.NUMPY_FLOAT)
            self.d_tile_shift_y = cp.empty(0, dtype=env.NUMPY_FLOAT)
            self.d_tile_shift_z = cp.empty(0, dtype=env.NUMPY_FLOAT)
            self.d_main_shift_x = cp.empty(0, dtype=env.NUMPY_FLOAT)
            self.d_main_shift_y = cp.empty(0, dtype=env.NUMPY_FLOAT)
            self.d_main_shift_z = cp.empty(0, dtype=env.NUMPY_FLOAT)
            self.d_excl_shift_x = cp.empty(0, dtype=env.NUMPY_FLOAT)
            self.d_excl_shift_y = cp.empty(0, dtype=env.NUMPY_FLOAT)
            self.d_excl_shift_z = cp.empty(0, dtype=env.NUMPY_FLOAT)
            return

        nt = self.num_tiles
        max_tiles = nt

        if self._d_classify_excl_tiles.size < max_tiles:
            self._d_classify_excl_tiles = cp.empty(max_tiles, dtype=env.NUMPY_INT)
            self._d_classify_excl_int_atoms = cp.empty(
                max_tiles * W, dtype=env.NUMPY_INT
            )
            self._d_classify_excl_masks = cp.empty(max_tiles * W, dtype=np.uint32)
            self._d_classify_excl_scale = cp.empty(max_tiles * W, dtype=np.uint32)
            self._d_classify_main_tiles = cp.empty(max_tiles, dtype=env.NUMPY_INT)
            self._d_classify_main_int_atoms = cp.empty(
                max_tiles * W, dtype=env.NUMPY_INT
            )
            self._d_classify_excl_shift_x = cp.empty(max_tiles, dtype=env.NUMPY_FLOAT)
            self._d_classify_excl_shift_y = cp.empty(max_tiles, dtype=env.NUMPY_FLOAT)
            self._d_classify_excl_shift_z = cp.empty(max_tiles, dtype=env.NUMPY_FLOAT)
            self._d_classify_main_shift_x = cp.empty(max_tiles, dtype=env.NUMPY_FLOAT)
            self._d_classify_main_shift_y = cp.empty(max_tiles, dtype=env.NUMPY_FLOAT)
            self._d_classify_main_shift_z = cp.empty(max_tiles, dtype=env.NUMPY_FLOAT)

        self._d_classify_excl_counter[0] = 0
        self._d_classify_main_counter[0] = 0

        tpb = 256
        grid = ((nt + tpb - 1) // tpb,)
        self._kernels["classify_tiles"](
            grid,
            (tpb,),
            (
                self.d_exclusion_masks,
                self.d_scaling_masks,
                self.d_tiles,
                self.d_interacting_atoms,
                self.d_tile_shift_x,
                self.d_tile_shift_y,
                self.d_tile_shift_z,
                np.int32(nt),
                self._d_classify_excl_counter,
                self._d_classify_main_counter,
                self._d_classify_excl_tiles,
                self._d_classify_excl_int_atoms,
                self._d_classify_excl_masks,
                self._d_classify_excl_scale,
                self._d_classify_main_tiles,
                self._d_classify_main_int_atoms,
                self._d_classify_excl_shift_x,
                self._d_classify_excl_shift_y,
                self._d_classify_excl_shift_z,
                self._d_classify_main_shift_x,
                self._d_classify_main_shift_y,
                self._d_classify_main_shift_z,
            ),
        )

        self.num_exclusion_tiles = int(self._d_classify_excl_counter[0])
        self.num_main_tiles = int(self._d_classify_main_counter[0])

        self.d_excl_tiles = self._d_classify_excl_tiles
        self.d_excl_interacting_atoms = self._d_classify_excl_int_atoms
        self.d_excl_exclusion_masks = self._d_classify_excl_masks
        self.d_excl_scaling_masks = self._d_classify_excl_scale
        self.d_main_tiles = self._d_classify_main_tiles
        self.d_main_interacting_atoms = self._d_classify_main_int_atoms
        self.d_excl_shift_x = self._d_classify_excl_shift_x
        self.d_excl_shift_y = self._d_classify_excl_shift_y
        self.d_excl_shift_z = self._d_classify_excl_shift_z
        self.d_main_shift_x = self._d_classify_main_shift_x
        self.d_main_shift_y = self._d_classify_main_shift_y
        self.d_main_shift_z = self._d_classify_main_shift_z

    def check_rebuild(self, positions) -> bool:
        if not self._is_initialized:
            return True
        if self.d_positions_at_rebuild_x.size == 0:
            return True

        if isinstance(positions, tuple):
            pos_x, pos_y, pos_z = positions
        else:
            data = cp.asarray(
                np.ascontiguousarray(positions.ravel(), dtype=env.NUMPY_FLOAT)
            )
            pos_x = cp.ascontiguousarray(data[0::3])
            pos_y = cp.ascontiguousarray(data[1::3])
            pos_z = cp.ascontiguousarray(data[2::3])

        self.d_rebuild_flag[0] = 0
        threshold_sq = (self.skin * 0.5) ** 2
        tpb = 256
        grid = ((self.num_particles + tpb - 1) // tpb,)
        self._kernels["check_rebuild"](
            grid,
            (tpb,),
            (
                pos_x,
                pos_y,
                pos_z,
                self.d_positions_at_rebuild_x,
                self.d_positions_at_rebuild_y,
                self.d_positions_at_rebuild_z,
                np.int32(self.num_particles),
                np.float32(threshold_sq),
                self._d_pbc_inv,
                self._d_pbc_matrix,
                self.d_rebuild_flag,
            ),
        )
        flag = int(self.d_rebuild_flag[0])
        return flag == 1

    def check_rebuild_async(self, positions) -> bool:
        if not self._is_initialized:
            return True
        if self.d_positions_at_rebuild_x.size == 0:
            return True

        if isinstance(positions, tuple):
            pos_x, pos_y, pos_z = positions
        else:
            data = cp.asarray(
                np.ascontiguousarray(positions.ravel(), dtype=env.NUMPY_FLOAT)
            )
            pos_x = cp.ascontiguousarray(data[0::3])
            pos_y = cp.ascontiguousarray(data[1::3])
            pos_z = cp.ascontiguousarray(data[2::3])

        self._ensure_kernels()
        threshold_sq = (self.skin * 0.5) ** 2
        tpb = 256
        grid = ((self.num_particles + tpb - 1) // tpb,)
        self._kernels["check_rebuild"](
            grid,
            (tpb,),
            (
                pos_x,
                pos_y,
                pos_z,
                self.d_positions_at_rebuild_x,
                self.d_positions_at_rebuild_y,
                self.d_positions_at_rebuild_z,
                np.int32(self.num_particles),
                np.float32(threshold_sq),
                self._d_pbc_inv,
                self._d_pbc_matrix,
                self.d_rebuild_flag,
            ),
        )
        return False

    def _init_empty(self):
        self.num_blocks = 0
        self.num_tiles = 0
        self.num_particles = 0
        self.d_block_atoms = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_block_center_x = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_block_center_y = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_block_center_z = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_block_size_x = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_block_size_y = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_block_size_z = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_atom_to_block = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_atom_to_slot = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_tiles = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_interacting_atoms = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_cell_block_offset = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_cell_block_count = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_block_to_cell = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_raw_order = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_pdb_to_sorted = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_sorted_to_pdb = cp.empty(0, dtype=env.NUMPY_INT)
        self._sorted_positions = None
        self.d_exclusion_masks = cp.empty(0, dtype=np.uint32)
        self.d_scaling_masks = cp.empty(0, dtype=np.uint32)
        self.num_exclusion_tiles = 0
        self.num_main_tiles = 0
        self.d_excl_tiles = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_excl_interacting_atoms = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_excl_exclusion_masks = cp.empty(0, dtype=np.uint32)
        self.d_excl_scaling_masks = cp.empty(0, dtype=np.uint32)
        self.d_main_tiles = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_main_interacting_atoms = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_tile_shift_x = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_tile_shift_y = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_tile_shift_z = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_main_shift_x = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_main_shift_y = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_main_shift_z = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_excl_shift_x = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_excl_shift_y = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_excl_shift_z = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_positions_at_rebuild_x = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_positions_at_rebuild_y = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_positions_at_rebuild_z = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self._invalidate_caches()
