from __future__ import annotations

import numpy as np
import cupy as cp
from mdpy import env

W = 32
NUM_ATOMS_SENTINEL = 0x7FFFFFFF

_PBC_WRAP_KERNEL = r"""
extern "C" __global__
void pbc_wrap_kernel(
    float* __restrict__ pos_x,
    float* __restrict__ pos_y,
    float* __restrict__ pos_z,
    const float* __restrict__ pbc_matrix,
    const float* __restrict__ pbc_inv,
    int number_particles
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= number_particles) return;

    float px = pos_x[index];
    float py = pos_y[index];
    float pz = pos_z[index];

    float fx = px * pbc_inv[0] + py * pbc_inv[3] + pz * pbc_inv[6];
    float fy = px * pbc_inv[1] + py * pbc_inv[4] + pz * pbc_inv[7];
    float fz = px * pbc_inv[2] + py * pbc_inv[5] + pz * pbc_inv[8];

    fx = fx - floorf(fx);
    fy = fy - floorf(fy);
    fz = fz - floorf(fz);

    pos_x[index] = fx * pbc_matrix[0] + fy * pbc_matrix[3] + fz * pbc_matrix[6];
    pos_y[index] = fx * pbc_matrix[1] + fy * pbc_matrix[4] + fz * pbc_matrix[7];
    pos_z[index] = fx * pbc_matrix[2] + fy * pbc_matrix[5] + fz * pbc_matrix[8];
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
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    const float* __restrict__ pbc_matrix,
    const float* __restrict__ pbc_inv,
    int number_particles,
    float box_x, float box_y, float box_z,
    unsigned long long* __restrict__ morton_codes
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= number_particles) return;

    float px = pos_x[index];
    float py = pos_y[index];
    float pz = pos_z[index];

    float fx = px * pbc_inv[0] + py * pbc_inv[3] + pz * pbc_inv[6];
    float fy = px * pbc_inv[1] + py * pbc_inv[4] + pz * pbc_inv[7];
    float fz = px * pbc_inv[2] + py * pbc_inv[5] + pz * pbc_inv[8];

    fx = fx - floorf(fx);
    fy = fy - floorf(fy);
    fz = fz - floorf(fz);

    float wx = fx * pbc_matrix[0] + fy * pbc_matrix[3] + fz * pbc_matrix[6];
    float wy = fx * pbc_matrix[1] + fy * pbc_matrix[4] + fz * pbc_matrix[7];
    float wz = fx * pbc_matrix[2] + fy * pbc_matrix[5] + fz * pbc_matrix[8];

    unsigned int ix = min((unsigned int)(wx / box_x * 1023.0f), 1023u);
    unsigned int iy = min((unsigned int)(wy / box_y * 1023.0f), 1023u);
    unsigned int iz = min((unsigned int)(wz / box_z * 1023.0f), 1023u);

    morton_codes[index] = morton_split(ix) | (morton_split(iy) << 1) | (morton_split(iz) << 2);
}
"""

_FORM_BLOCKS_KERNEL = r"""
extern "C" __global__
void form_blocks_kernel(
    const int* __restrict__ sorted_indices,
    int number_particles,
    int num_blocks,
    int* __restrict__ block_atoms_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks * 32) return;
    if (idx < number_particles) {
        block_atoms_out[idx] = sorted_indices[idx];
    } else {
        block_atoms_out[idx] = -1;
    }
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
    float* __restrict__ block_center_out,
    float* __restrict__ block_size_out
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
    block_center_out[bi*3]   = 0.5f*(min_x + max_x);
    block_center_out[bi*3+1] = 0.5f*(min_y + max_y);
    block_center_out[bi*3+2] = 0.5f*(min_z + max_z);
    block_size_out[bi*3]   = 0.5f*(max_x - min_x);
    block_size_out[bi*3+1] = 0.5f*(max_y - min_y);
    block_size_out[bi*3+2] = 0.5f*(max_z - min_z);
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

_COMPUTE_LARGE_BLOCK_BOUNDS_KERNEL = r"""
extern "C" __global__
void compute_large_block_bounds_kernel(
    const float* __restrict__ block_center,
    const float* __restrict__ block_size,
    int num_blocks,
    int num_large_blocks,
    float box_x, float box_y, float box_z,
    float inv_box_x, float inv_box_y, float inv_box_z,
    float* __restrict__ large_block_center_out,
    float* __restrict__ large_block_size_out
) {
    int lb = blockIdx.x * blockDim.x + threadIdx.x;
    if (lb >= num_large_blocks) return;

    int start = lb * 32;
    int end = min(start + 32, num_blocks);

    float min_x = 1e30f, min_y = 1e30f, min_z = 1e30f;
    float max_x = -1e30f, max_y = -1e30f, max_z = -1e30f;

    for (int b = start; b < end; b++) {
        float bcx = block_center[b*3], bcy = block_center[b*3+1], bcz = block_center[b*3+2];
        float bsx = block_size[b*3],   bsy = block_size[b*3+1],   bsz = block_size[b*3+2];
        if (b > start) {
            float ref_x = block_center[start*3], ref_y = block_center[start*3+1], ref_z = block_center[start*3+2];
            float dx = bcx - ref_x, dy = bcy - ref_y, dz = bcz - ref_z;
            dx -= box_x * roundf(dx * inv_box_x);
            dy -= box_y * roundf(dy * inv_box_y);
            dz -= box_z * roundf(dz * inv_box_z);
            bcx = ref_x + dx; bcy = ref_y + dy; bcz = ref_z + dz;
        }
        min_x = fminf(min_x, bcx - bsx); min_y = fminf(min_y, bcy - bsy); min_z = fminf(min_z, bcz - bsz);
        max_x = fmaxf(max_x, bcx + bsx); max_y = fmaxf(max_y, bcy + bsy); max_z = fmaxf(max_z, bcz + bsz);
    }

    large_block_center_out[lb*3]   = 0.5f * (min_x + max_x);
    large_block_center_out[lb*3+1] = 0.5f * (min_y + max_y);
    large_block_center_out[lb*3+2] = 0.5f * (min_z + max_z);
    large_block_size_out[lb*3]     = 0.5f * (max_x - min_x);
    large_block_size_out[lb*3+1]   = 0.5f * (max_y - min_y);
    large_block_size_out[lb*3+2]   = 0.5f * (max_z - min_z);
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
    float box_x, float box_y, float box_z,
    float inv_box_x, float inv_box_y, float inv_box_z,
    int* __restrict__ rebuild_flag
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    float dx = pos_x[idx] - old_pos_x[idx];
    float dy = pos_y[idx] - old_pos_y[idx];
    float dz = pos_z[idx] - old_pos_z[idx];
    dx -= box_x * roundf(dx * inv_box_x);
    dy -= box_y * roundf(dy * inv_box_y);
    dz -= box_z * roundf(dz * inv_box_z);
    if (dx*dx + dy*dy + dz*dz > threshold_sq)
        rebuild_flag[0] = 1;
}
"""

_GATHER_SORTED_KERNEL = r"""
extern "C" __global__
void gather_sorted_kernel(
    const float* __restrict__ src,
    const int* __restrict__ block_atoms,
    int total_slots,
    int num_particles,
    float* __restrict__ dst
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_slots) return;
    int atom_id = block_atoms[idx];
    float val = 0.0f;
    if (atom_id >= 0 && atom_id < num_particles) {
        val = src[atom_id];
    }
    dst[idx] = val;
}
"""

_GATHER_SORTED_KERNEL_2COMP = r"""
extern "C" __global__
void gather_sorted_kernel_2comp(
    const float* __restrict__ src,
    const int* __restrict__ block_atoms,
    int total_slots,
    int num_particles,
    float* __restrict__ dst
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_slots) return;
    int atom_id = block_atoms[idx];
    if (atom_id >= 0 && atom_id < num_particles) {
        dst[idx * 2 + 0] = src[atom_id * 2 + 0];
        dst[idx * 2 + 1] = src[atom_id * 2 + 1];
    } else {
        dst[idx * 2 + 0] = 0.0f;
        dst[idx * 2 + 1] = 0.0f;
    }
}
"""

_GATHER_SORTED_KERNEL_4COMP = r"""
extern "C" __global__
void gather_sorted_kernel_4comp(
    const float* __restrict__ src,
    const int* __restrict__ block_atoms,
    int total_slots,
    int num_particles,
    float* __restrict__ dst
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_slots) return;
    int atom_id = block_atoms[idx];
    if (atom_id >= 0 && atom_id < num_particles) {
        dst[idx * 4 + 0] = src[atom_id * 4 + 0];
        dst[idx * 4 + 1] = src[atom_id * 4 + 1];
        dst[idx * 4 + 2] = src[atom_id * 4 + 2];
        dst[idx * 4 + 3] = src[atom_id * 4 + 3];
    } else {
        dst[idx * 4 + 0] = 0.0f;
        dst[idx * 4 + 1] = 0.0f;
        dst[idx * 4 + 2] = 0.0f;
        dst[idx * 4 + 3] = 0.0f;
    }
}
"""

_PERMUTE_ARRAY_KERNEL = r"""
extern "C" __global__
void permute_array_kernel(
    const float* __restrict__ src,
    const int* __restrict__ permutation,
    int num_particles,
    float* __restrict__ dst
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    dst[idx] = src[permutation[idx]];
}
"""

_PERMUTE_INT_ARRAY_KERNEL = r"""
extern "C" __global__
void permute_int_array_kernel(
    const int* __restrict__ src,
    const int* __restrict__ permutation,
    int num_particles,
    int* __restrict__ dst
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    dst[idx] = src[permutation[idx]];
}
"""

_PERMUTE_ARRAY_2COMP_KERNEL = r"""
extern "C" __global__
void permute_array_2comp_kernel(
    const float* __restrict__ src,
    const int* __restrict__ permutation,
    int num_particles,
    float* __restrict__ dst
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    int src_idx = permutation[idx];
    dst[idx * 2 + 0] = src[src_idx * 2 + 0];
    dst[idx * 2 + 1] = src[src_idx * 2 + 1];
}
"""

_INVERSE_PERMUTE_KERNEL = r"""
extern "C" __global__
void inverse_permute_kernel(
    const float* __restrict__ sorted_src,
    const int* __restrict__ sorted_to_pdb,
    int num_particles,
    float* __restrict__ pdb_dst
) {
    int sorted_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sorted_idx >= num_particles) return;
    int pdb_idx = sorted_to_pdb[sorted_idx];
    pdb_dst[pdb_idx] = sorted_src[sorted_idx];
}
"""

_FIND_INTERACTING_BLOCKS_KERNEL = r"""
extern "C" __global__ __launch_bounds__(256, 3)
void find_interacting_blocks_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    const int* __restrict__ block_atoms,
    const float* __restrict__ block_center,
    const float* __restrict__ block_size,
    int num_blocks,
    int num_particles,
    float build_radius_sq,
    float box_x, float box_y, float box_z,
    float inv_box_x, float inv_box_y, float inv_box_z,
    const float* __restrict__ large_block_center,
    const float* __restrict__ large_block_size,
    int* __restrict__ tiles_out,
    int* __restrict__ interacting_atoms_out,
    int* __restrict__ interaction_count,
    int max_tiles
) {
    int tgx = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int global_warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    if (global_warp >= num_blocks) return;

    __shared__ int buffer[8 * 256];
    int* my_buf = buffer + warp_in_block * 256;
    int nBuf = 0;

    int bx = global_warp;
    float cx = block_center[bx*3], cy = block_center[bx*3+1], cz = block_center[bx*3+2];
    float sx = block_size[bx*3],   sy = block_size[bx*3+1],   sz = block_size[bx*3+2];
    int my_large_block = bx >> 5;
    int num_large_blocks = (num_blocks + 31) >> 5;

    for (int lb = my_large_block; lb < num_large_blocks; lb++) {
        bool lb_pass;
        if (lb == my_large_block) {
            lb_pass = true;
        } else {
            float lcx = large_block_center[lb*3];
            float lcy = large_block_center[lb*3+1];
            float lcz = large_block_center[lb*3+2];
            float lsx = large_block_size[lb*3];
            float lsy = large_block_size[lb*3+1];
            float lsz = large_block_size[lb*3+2];

            float dx = lcx - cx, dy = lcy - cy, dz = lcz - cz;
            dx -= box_x * roundf(dx * inv_box_x);
            dy -= box_y * roundf(dy * inv_box_y);
            dz -= box_z * roundf(dz * inv_box_z);
            dx = fmaxf(0.0f, fabsf(dx) - sx - lsx);
            dy = fmaxf(0.0f, fabsf(dy) - sy - lsy);
            dz = fmaxf(0.0f, fabsf(dz) - sz - lsz);
            lb_pass = (dx*dx + dy*dy + dz*dz < build_radius_sq);
        }

        if (!lb_pass) continue;

        int block2Base = lb << 5;
        {
            int block2 = block2Base + tgx;
            bool include_block = false;

            if (block2 < num_blocks && block2 > bx) {
                float bcx2 = block_center[block2*3];
                float bcy2 = block_center[block2*3+1];
                float bcz2 = block_center[block2*3+2];
                float bsx2 = block_size[block2*3];
                float bsy2 = block_size[block2*3+1];
                float bsz2 = block_size[block2*3+2];

                float dx = bcx2 - cx, dy = bcy2 - cy, dz = bcz2 - cz;
                dx -= box_x * roundf(dx * inv_box_x);
                dy -= box_y * roundf(dy * inv_box_y);
                dz -= box_z * roundf(dz * inv_box_z);
                dx = fmaxf(0.0f, fabsf(dx) - sx - bsx2);
                dy = fmaxf(0.0f, fabsf(dy) - sy - bsy2);
                dz = fmaxf(0.0f, fabsf(dz) - sz - bsz2);
                include_block = (dx*dx + dy*dy + dz*dz < build_radius_sq);
            }

            unsigned int include_flags = __ballot_sync(0xffffffff, include_block);

            while (include_flags != 0) {
                int i = __ffs(include_flags) - 1;
                include_flags &= include_flags - 1;
                int by = block2Base + i;

                int gj = block_atoms[by * 32 + tgx];
                int interacts = 0;

                if (gj >= 0 && gj < num_particles) {
                    float px_j = pos_x[gj];
                    float py_j = pos_y[gj];
                    float pz_j = pos_z[gj];
                    for (int k = 0; k < 32; k++) {
                        int gk = block_atoms[bx * 32 + k];
                        if (gk < 0) continue;
                        float ddx = px_j - pos_x[gk];
                        float ddy = py_j - pos_y[gk];
                        float ddz = pz_j - pos_z[gk];
                        ddx -= box_x * roundf(ddx * inv_box_x);
                        ddy -= box_y * roundf(ddy * inv_box_y);
                        ddz -= box_z * roundf(ddz * inv_box_z);
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
                        if (tgx < 1) tiles_out[ti] = bx;
                        interacting_atoms_out[ti * 32 + tgx] = my_buf[tgx];
                    }
                    for (int s = tgx; s < nBuf - 32; s += 32)
                        my_buf[s] = my_buf[s + 32];
                    nBuf -= 32;
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
                if (tgx < 1) tiles_out[ti] = bx;
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
            if (tgx < 1) tiles_out[ti] = bx;
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
    int num_tiles,
    int* __restrict__ excl_counter,
    int* __restrict__ main_counter,
    int* __restrict__ excl_tiles_out,
    int* __restrict__ excl_int_atoms_out,
    unsigned int* __restrict__ excl_masks_out,
    unsigned int* __restrict__ excl_scale_out,
    int* __restrict__ main_tiles_out,
    int* __restrict__ main_int_atoms_out
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
    } else {
        int idx = atomicAdd(main_counter, 1);
        main_tiles_out[idx] = tiles[tile];
        for (int i = 0; i < 32; i++) {
            main_int_atoms_out[idx * 32 + i] = interacting_atoms[tile * 32 + i];
        }
    }
}
"""


def _compile_gpu_kernels():
    return {
        "wrap": cp.RawKernel(_PBC_WRAP_KERNEL, "pbc_wrap_kernel"),
        "morton": cp.RawKernel(_MORTON_ENCODE_KERNEL, "morton_encode_kernel"),
        "form_blocks": cp.RawKernel(_FORM_BLOCKS_KERNEL, "form_blocks_kernel"),
        "compute_bounds": cp.RawKernel(
            _COMPUTE_BLOCK_BOUNDS_KERNEL, "compute_block_bounds_kernel"
        ),
        "atom_map": cp.RawKernel(_BUILD_ATOM_MAP_KERNEL, "build_atom_map_kernel"),
        "find_interacting": cp.RawKernel(
            _FIND_INTERACTING_BLOCKS_KERNEL, "find_interacting_blocks_kernel"
        ),
        "rev_count": cp.RawKernel(
            _BUILD_REVERSE_COUNT_KERNEL, "build_reverse_count_kernel"
        ),
        "rev_fill": cp.RawKernel(_FILL_REVERSE_KERNEL, "fill_reverse_kernel"),
        "build_masks": cp.RawKernel(_BUILD_MASKS_KERNEL, "build_masks_kernel"),
        "check_rebuild": cp.RawKernel(_CHECK_REBUILD_KERNEL, "check_rebuild_kernel"),
        "gather_sorted": cp.RawKernel(_GATHER_SORTED_KERNEL, "gather_sorted_kernel"),
        "gather_sorted_2comp": cp.RawKernel(
            _GATHER_SORTED_KERNEL_2COMP, "gather_sorted_kernel_2comp"
        ),
        "gather_sorted_4comp": cp.RawKernel(
            _GATHER_SORTED_KERNEL_4COMP, "gather_sorted_kernel_4comp"
        ),
        "large_block_bounds": cp.RawKernel(
            _COMPUTE_LARGE_BLOCK_BOUNDS_KERNEL, "compute_large_block_bounds_kernel"
        ),
        "permute": cp.RawKernel(_PERMUTE_ARRAY_KERNEL, "permute_array_kernel"),
        "permute_int": cp.RawKernel(
            _PERMUTE_INT_ARRAY_KERNEL, "permute_int_array_kernel"
        ),
        "permute_2comp": cp.RawKernel(
            _PERMUTE_ARRAY_2COMP_KERNEL, "permute_array_2comp_kernel"
        ),
        "inverse_permute": cp.RawKernel(
            _INVERSE_PERMUTE_KERNEL, "inverse_permute_kernel"
        ),
        "classify_tiles": cp.RawKernel(_CLASSIFY_TILES_KERNEL, "classify_tiles_kernel"),
    }


class TileList:

    def __init__(
        self, cutoff: float, skin: float = 1.0, rebuild_check_interval: int = 10
    ):
        self.cutoff = cutoff
        self.skin = skin
        self.build_radius = cutoff + skin
        self._is_initialized = False
        self.rebuild_check_interval = rebuild_check_interval

        self.num_blocks = 0
        self.num_tiles = 0
        self.num_particles = 0
        self._max_tiles = 0

        self.d_block_atoms = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_block_center = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_block_size = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_atom_to_block = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_atom_to_slot = cp.empty(0, dtype=env.NUMPY_INT)

        self.d_tiles = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_interacting_atoms = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_exclusion_masks = cp.empty(0, dtype=np.uint32)
        self.d_scaling_masks = cp.empty(0, dtype=np.uint32)

        self.d_positions_at_rebuild_x = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_positions_at_rebuild_y = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_positions_at_rebuild_z = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self._d_counters = cp.zeros(1, dtype=env.NUMPY_INT)
        self.d_rebuild_flag = cp.zeros(1, dtype=env.NUMPY_INT)
        self.num_large_blocks = 0
        self.d_large_block_center = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_large_block_size = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self._d_tile_buf = cp.empty(0, dtype=env.NUMPY_INT)
        self._d_interacting_buf = cp.empty(0, dtype=env.NUMPY_INT)
        self._d_pbc_matrix = None
        self._d_pbc_inv = None

        self.d_sorted_pos_x = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_sorted_pos_y = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_sorted_pos_z = cp.empty(0, dtype=env.NUMPY_FLOAT)

        self._d_excl_offset = None
        self._d_excl_neighbors = None
        self._d_excl_scale = None
        self._d_reverse_offset = None
        self._d_reverse_neighbors = None
        self._d_reverse_scale = None
        self._total_exclusion_pairs = 0

        self.d_sorted_to_pdb = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_pdb_to_sorted = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_raw_order = cp.empty(0, dtype=env.NUMPY_INT)
        self._sorted_positions = None

        self._block_atoms_np = None
        self._block_center_np = None
        self._tiles_np = None
        self._interacting_atoms_np = None
        self._exclusion_masks_np = None
        self._scaling_masks_np = None

        self._d_classify_excl_counter = cp.zeros(1, dtype=env.NUMPY_INT)
        self._d_classify_main_counter = cp.zeros(1, dtype=env.NUMPY_INT)
        self._d_classify_excl_tiles = cp.empty(0, dtype=env.NUMPY_INT)
        self._d_classify_excl_int_atoms = cp.empty(0, dtype=env.NUMPY_INT)
        self._d_classify_excl_masks = cp.empty(0, dtype=np.uint32)
        self._d_classify_excl_scale = cp.empty(0, dtype=np.uint32)
        self._d_classify_main_tiles = cp.empty(0, dtype=env.NUMPY_INT)
        self._d_classify_main_int_atoms = cp.empty(0, dtype=env.NUMPY_INT)

        self._kernels = None

    @property
    def block_atoms(self):
        if self._block_atoms_np is None and self.d_block_atoms.size > 0:
            self._block_atoms_np = cp.asnumpy(self.d_block_atoms).reshape(-1, W)
        return self._block_atoms_np

    @property
    def block_center(self):
        if self._block_center_np is None and self.d_block_center.size > 0:
            self._block_center_np = cp.asnumpy(self.d_block_center).reshape(-1, 3)
        return self._block_center_np

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
                self.d_exclusion_masks[: self.num_tiles * W]
            ).reshape(-1, W)
        return self._exclusion_masks_np

    @property
    def scaling_masks(self):
        if self._scaling_masks_np is None and self.d_scaling_masks.size > 0:
            self._scaling_masks_np = cp.asnumpy(
                self.d_scaling_masks[: self.num_tiles * W]
            ).reshape(-1, W)
        return self._scaling_masks_np

    def _invalidate_caches(self):
        self._block_atoms_np = None
        self._block_center_np = None
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

    def update_sorted_positions(self, d_pos_x, d_pos_y, d_pos_z):
        if self.num_blocks == 0:
            return
        self._ensure_kernels()
        total_slots = self.num_blocks * W
        tpb = 256
        grid = ((total_slots + tpb - 1) // tpb,)
        for src, attr in [
            (d_pos_x, "d_sorted_pos_x"),
            (d_pos_y, "d_sorted_pos_y"),
            (d_pos_z, "d_sorted_pos_z"),
        ]:
            if getattr(self, attr).size != total_slots:
                setattr(self, attr, cp.empty(total_slots, dtype=env.NUMPY_FLOAT))
            self._kernels["gather_sorted"](
                grid,
                (tpb,),
                (
                    src,
                    self.d_block_atoms,
                    np.int32(total_slots),
                    np.int32(self.num_particles),
                    getattr(self, attr),
                ),
            )

    def gather_sorted_params(self, param_arrays):
        if self.num_blocks == 0:
            return
        self._ensure_kernels()
        total_slots = self.num_blocks * W
        tpb = 256
        grid = ((total_slots + tpb - 1) // tpb,)
        for name, d_arr in param_arrays.items():
            num_components = (
                d_arr.shape[0] // self.num_particles if self.num_particles > 0 else 1
            )
            if num_components == 1 or d_arr.shape[0] == total_slots:
                sorted_arr = cp.empty(total_slots, dtype=env.NUMPY_FLOAT)
                self._kernels["gather_sorted"](
                    grid,
                    (tpb,),
                    (
                        d_arr,
                        self.d_block_atoms,
                        np.int32(total_slots),
                        np.int32(self.num_particles),
                        sorted_arr,
                    ),
                )
            elif num_components == 2:
                sorted_arr = cp.empty(total_slots * 2, dtype=env.NUMPY_FLOAT)
                self._kernels["gather_sorted_2comp"](
                    grid,
                    (tpb,),
                    (
                        d_arr,
                        self.d_block_atoms,
                        np.int32(total_slots),
                        np.int32(self.num_particles),
                        sorted_arr,
                    ),
                )
            elif num_components == 4:
                sorted_arr = cp.empty(total_slots * 4, dtype=env.NUMPY_FLOAT)
                self._kernels["gather_sorted_4comp"](
                    grid,
                    (tpb,),
                    (
                        d_arr,
                        self.d_block_atoms,
                        np.int32(total_slots),
                        np.int32(self.num_particles),
                        sorted_arr,
                    ),
                )
            else:
                raise ValueError(f"Unsupported number of components: {num_components}")
            setattr(self, f"d_sorted_{name}", sorted_arr)

    def permute_to_sorted(
        self, permutation, arrays_float, arrays_int=None, arrays_2comp=None
    ):
        if self.num_particles == 0:
            return
        self._ensure_kernels()
        N = self.num_particles
        tpb = 256
        grid = ((N + tpb - 1) // tpb,)
        for name, src in arrays_float.items():
            dst = cp.empty_like(src)
            self._kernels["permute"](grid, (tpb,), (src, permutation, np.int32(N), dst))
            arrays_float[name] = dst
        if arrays_int:
            for name, src in arrays_int.items():
                dst = cp.empty_like(src)
                self._kernels["permute_int"](
                    grid, (tpb,), (src, permutation, np.int32(N), dst)
                )
                arrays_int[name] = dst
        if arrays_2comp:
            for name, src in arrays_2comp.items():
                dst = cp.empty_like(src)
                self._kernels["permute_2comp"](
                    grid, (tpb,), (src, permutation, np.int32(N), dst)
                )
                arrays_2comp[name] = dst

    def permute_from_sorted(self, sorted_to_pdb, sorted_array):
        if self.num_particles == 0:
            return sorted_array
        self._ensure_kernels()
        N = self.num_particles
        tpb = 256
        grid = ((N + tpb - 1) // tpb,)
        pdb_array = cp.empty_like(sorted_array)
        self._kernels["inverse_permute"](
            grid, (tpb,), (sorted_array, sorted_to_pdb, np.int32(N), pdb_array)
        )
        return pdb_array

    def _rebuild_core(self, positions, topology, pbc_matrix, pbc_inv):
        N = topology.num_particles
        self.num_particles = N
        tpb = 256

        if isinstance(positions, tuple):
            pos_x = cp.asarray(positions[0], dtype=env.NUMPY_FLOAT).copy()
            pos_y = cp.asarray(positions[1], dtype=env.NUMPY_FLOAT).copy()
            pos_z = cp.asarray(positions[2], dtype=env.NUMPY_FLOAT).copy()
        else:
            data = cp.asarray(
                np.ascontiguousarray(positions.ravel(), dtype=env.NUMPY_FLOAT)
            )
            pos_x = data[0::3].copy()
            pos_y = data[1::3].copy()
            pos_z = data[2::3].copy()
        self._upload_pbc(pbc_matrix, pbc_inv)

        n3 = (N + tpb - 1) // tpb
        self._kernels["wrap"](
            (n3,),
            (tpb,),
            (pos_x, pos_y, pos_z, self._d_pbc_matrix, self._d_pbc_inv, np.int32(N)),
        )

        morton_codes = cp.empty(N, dtype=np.uint64)
        pbc_2d = np.asarray(pbc_matrix).reshape(3, 3)
        box_x = abs(float(pbc_2d[0, 0]))
        box_y = abs(float(pbc_2d[1, 1]))
        box_z = abs(float(pbc_2d[2, 2]))
        self._box_x = box_x
        self._box_y = box_y
        self._box_z = box_z
        self._inv_box_x = 1.0 / box_x
        self._inv_box_y = 1.0 / box_y
        self._inv_box_z = 1.0 / box_z

        nm = (N + tpb - 1) // tpb
        self._kernels["morton"](
            (nm,),
            (tpb,),
            (
                pos_x,
                pos_y,
                pos_z,
                self._d_pbc_matrix,
                self._d_pbc_inv,
                np.int32(N),
                np.float32(box_x),
                np.float32(box_y),
                np.float32(box_z),
                morton_codes,
            ),
        )

        sorted_indices = cp.argsort(morton_codes).astype(env.NUMPY_INT)

        raw_order = sorted_indices.copy()
        raw_inv = cp.empty(N, dtype=env.NUMPY_INT)
        raw_inv[sorted_indices] = cp.arange(N, dtype=env.NUMPY_INT)
        self.d_raw_order = raw_order
        self.d_pdb_to_sorted = raw_inv

        pos_x = pos_x[sorted_indices]
        pos_y = pos_y[sorted_indices]
        pos_z = pos_z[sorted_indices]

        num_blocks = (N + W - 1) // W
        self.num_blocks = num_blocks
        total_slots = num_blocks * W
        block_atoms = cp.full(total_slots, -1, dtype=env.NUMPY_INT)
        block_atoms[:N] = cp.arange(N, dtype=env.NUMPY_INT)
        self.d_block_atoms = block_atoms

        self.d_block_center = cp.empty(num_blocks * 3, dtype=env.NUMPY_FLOAT)
        self.d_block_size = cp.empty(num_blocks * 3, dtype=env.NUMPY_FLOAT)
        nb = (num_blocks + tpb - 1) // tpb
        self._kernels["compute_bounds"](
            (nb,),
            (tpb,),
            (
                pos_x,
                pos_y,
                pos_z,
                self.d_block_atoms,
                np.int32(num_blocks),
                self.d_block_center,
                self.d_block_size,
            ),
        )

        num_large_blocks = (num_blocks + 31) // 32
        self.num_large_blocks = num_large_blocks
        self.d_large_block_center = cp.empty(
            num_large_blocks * 3, dtype=env.NUMPY_FLOAT
        )
        self.d_large_block_size = cp.empty(num_large_blocks * 3, dtype=env.NUMPY_FLOAT)
        nlb = (num_large_blocks + tpb - 1) // tpb
        self._kernels["large_block_bounds"](
            (nlb,),
            (tpb,),
            (
                self.d_block_center,
                self.d_block_size,
                np.int32(num_blocks),
                np.int32(num_large_blocks),
                np.float32(box_x),
                np.float32(box_y),
                np.float32(box_z),
                np.float32(1.0 / box_x),
                np.float32(1.0 / box_y),
                np.float32(1.0 / box_z),
                self.d_large_block_center,
                self.d_large_block_size,
            ),
        )

        self.d_atom_to_block = cp.full(N, -1, dtype=env.NUMPY_INT)
        self.d_atom_to_slot = cp.full(N, -1, dtype=env.NUMPY_INT)
        self._kernels["atom_map"](
            (nb,),
            (tpb,),
            (
                self.d_block_atoms,
                np.int32(num_blocks),
                np.int32(W),
                self.d_atom_to_block,
                self.d_atom_to_slot,
            ),
        )

        return (pos_x, pos_y, pos_z)

    def _find_interacting_blocks(self, positions_soa, pbc_matrix):
        pos_x, pos_y, pos_z = positions_soa
        num_blocks = self.num_blocks
        pbc_2d = np.asarray(pbc_matrix).reshape(3, 3)
        box_x = abs(float(pbc_2d[0, 0]))
        box_y = abs(float(pbc_2d[1, 1]))
        box_z = abs(float(pbc_2d[2, 2]))
        build_radius = self.build_radius
        build_radius_sq = build_radius**2
        inv_box_x = 1.0 / box_x
        inv_box_y = 1.0 / box_y
        inv_box_z = 1.0 / box_z

        max_tiles = max(num_blocks * 100, 10000)
        if self._d_tile_buf.size < max_tiles:
            self._d_tile_buf = cp.empty(max_tiles, dtype=env.NUMPY_INT)
            self._d_interacting_buf = cp.empty(max_tiles * W, dtype=env.NUMPY_INT)
            self._max_tiles = max_tiles
        self._d_counters[0] = 0

        tpb = 256
        grid_blocks = max((num_blocks + 7) // 8, 1)
        self._kernels["find_interacting"](
            (grid_blocks,),
            (tpb,),
            (
                pos_x,
                pos_y,
                pos_z,
                self.d_block_atoms,
                self.d_block_center,
                self.d_block_size,
                np.int32(num_blocks),
                np.int32(self.num_particles),
                np.float32(build_radius_sq),
                np.float32(box_x),
                np.float32(box_y),
                np.float32(box_z),
                np.float32(inv_box_x),
                np.float32(inv_box_y),
                np.float32(inv_box_z),
                self.d_large_block_center,
                self.d_large_block_size,
                self._d_tile_buf,
                self._d_interacting_buf,
                self._d_counters,
                np.int32(max_tiles),
            ),
        )

        self.num_tiles = int(self._d_counters[0])
        self.d_tiles = self._d_tile_buf
        self.d_interacting_atoms = self._d_interacting_buf

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

            d_rev_offset = cp.cumsum(d_rev_offset, dtype=env.NUMPY_INT)
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
                np.int32(nt),
                self._d_classify_excl_counter,
                self._d_classify_main_counter,
                self._d_classify_excl_tiles,
                self._d_classify_excl_int_atoms,
                self._d_classify_excl_masks,
                self._d_classify_excl_scale,
                self._d_classify_main_tiles,
                self._d_classify_main_int_atoms,
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

    def rebuild(self, positions, topology, pbc_matrix, pbc_inv):
        N = topology.num_particles
        if N == 0:
            self._init_empty()
            return None, None

        self._ensure_kernels()
        self._invalidate_caches()

        positions_soa = self._rebuild_core(positions, topology, pbc_matrix, pbc_inv)
        self._sorted_positions = positions_soa

        pos_x, pos_y, pos_z = positions_soa
        self.d_positions_at_rebuild_x = pos_x.copy()
        self.d_positions_at_rebuild_y = pos_y.copy()
        self.d_positions_at_rebuild_z = pos_z.copy()

        total_slots = self.num_blocks * W
        self.d_sorted_pos_x = cp.zeros(total_slots, dtype=env.NUMPY_FLOAT)
        self.d_sorted_pos_y = cp.zeros(total_slots, dtype=env.NUMPY_FLOAT)
        self.d_sorted_pos_z = cp.zeros(total_slots, dtype=env.NUMPY_FLOAT)
        self.d_sorted_pos_x[:N] = pos_x
        self.d_sorted_pos_y[:N] = pos_y
        self.d_sorted_pos_z[:N] = pos_z

        self.d_rebuild_flag[0] = 0
        self._is_initialized = True

        raw_order = self.d_raw_order
        if self.d_sorted_to_pdb.size == N:
            self.d_sorted_to_pdb = self.d_sorted_to_pdb[raw_order]
        else:
            self.d_sorted_to_pdb = raw_order.copy()

        self._d_reverse_offset = None
        self._d_reverse_neighbors = None
        self._d_reverse_scale = None

        return self.d_pdb_to_sorted, None

    def build_tiles(self, topology, pbc_matrix):
        if self.num_particles == 0 or self._sorted_positions is None:
            return
        self._find_interacting_blocks(self._sorted_positions, pbc_matrix)
        self._build_masks_gpu(topology)
        self._extract_exclusion_tiles()

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
                np.float32(self._box_x),
                np.float32(self._box_y),
                np.float32(self._box_z),
                np.float32(self._inv_box_x),
                np.float32(self._inv_box_y),
                np.float32(self._inv_box_z),
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
                np.float32(self._box_x),
                np.float32(self._box_y),
                np.float32(self._box_z),
                np.float32(self._inv_box_x),
                np.float32(self._inv_box_y),
                np.float32(self._inv_box_z),
                self.d_rebuild_flag,
            ),
        )
        return False

    def _init_empty(self):
        self.num_blocks = 0
        self.num_tiles = 0
        self.num_particles = 0
        self.d_block_atoms = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_block_center = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_block_size = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_atom_to_block = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_atom_to_slot = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_tiles = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_interacting_atoms = cp.empty(0, dtype=env.NUMPY_INT)
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
        self.d_positions_at_rebuild_x = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_positions_at_rebuild_y = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_positions_at_rebuild_z = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_sorted_pos_x = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_sorted_pos_y = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_sorted_pos_z = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_sorted_to_pdb = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_pdb_to_sorted = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_raw_order = cp.empty(0, dtype=env.NUMPY_INT)
        self._sorted_positions = None
        self._invalidate_caches()
