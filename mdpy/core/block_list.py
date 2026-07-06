from __future__ import annotations

import ctypes
import math

import numpy as np
import cupy as cp
from mdpy import env
from mdpy.core.hilbert import HILBERT_ENCODE_KERNEL
from mdpy.core.radix_sort import fill_constant

BLOCK_SIZE = 32
NUM_ATOMS_SENTINEL = 0x7FFFFFFF

# Thread count for the single-block parallel prefix-sum kernels
# (cell_prefix_sum_kernel, composite_prefix_sum_kernel). Must be a power of
# two and must match the #define SCAN_BLOCK in _BLOCK_SCAN_PREAMBLE.
SCAN_BLOCK = 1024

_BLOCK_SCAN_PREAMBLE = r"""
#define SCAN_BLOCK 1024

// Single-block exclusive prefix sum. Contract:
//   - gridDim.x == 1, blockDim.x == SCAN_BLOCK.
//   - Each of the SCAN_BLOCK threads owns a contiguous tile of the n inputs.
//   - Kogge-Stone inclusive scan over per-tile partial sums (in s_part, a
//     shared int[SCAN_BLOCK]) fixes cross-tile carries.
// Writes out[0..n-1] = exclusive prefix sums and *total_out = sum(in[0..n-1]).
__device__ __forceinline__
void scan_block_excl(const int* __restrict__ in, int* __restrict__ out,
                     int n, int* __restrict__ s_part,
                     int* __restrict__ total_out) {
    const int tid = threadIdx.x;
    const int B = SCAN_BLOCK;
    const int n_per = (n + B - 1) / B;
    const int lo = tid * n_per;
    const int hi = (lo + n_per < n) ? (lo + n_per) : n;

    int my_sum = 0;
    for (int j = lo; j < hi; ++j) my_sum += in[j];
    s_part[tid] = my_sum;
    __syncthreads();

    for (int stride = 1; stride < B; stride <<= 1) {
        int v = (tid >= stride) ? s_part[tid - stride] : 0;
        __syncthreads();
        s_part[tid] += v;
        __syncthreads();
    }

    const int total = s_part[B - 1];
    const int my_base = (tid == 0) ? 0 : s_part[tid - 1];
    __syncthreads();

    int acc = my_base;
    for (int j = lo; j < hi; ++j) {
        out[j] = acc;
        acc += in[j];
    }
    __syncthreads();
    if (tid == 0) *total_out = total;
}
"""

_CELL_ASSIGN_KERNEL = HILBERT_ENCODE_KERNEL + r"""
extern "C" __global__
void cell_assign_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    const float* __restrict__ pbc_matrix,
    const float* __restrict__ pbc_inv,
    int number_particles,
    int nc_x, int nc_y, int nc_z,
    int hilbert_L,
    int* __restrict__ cell_counts,
    int* __restrict__ composite_counts,
    unsigned long long* __restrict__ sort_keys,
    int* __restrict__ cell_indices
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= number_particles) return;

    float px = pos_x[i], py = pos_y[i], pz = pos_z[i];
    float fx = px*pbc_inv[0] + py*pbc_inv[3] + pz*pbc_inv[6];
    float fy = px*pbc_inv[1] + py*pbc_inv[4] + pz*pbc_inv[7];
    float fz = px*pbc_inv[2] + py*pbc_inv[5] + pz*pbc_inv[8];

    int cx = (int)(fx * nc_x);
    cx = max(0, min(cx, nc_x - 1));
    int cy = (int)(fy * nc_y);
    cy = max(0, min(cy, nc_y - 1));
    int cz = (int)(fz * nc_z);
    cz = max(0, min(cz, nc_z - 1));
    int cell_index = cx + cy * nc_x + cz * nc_x * nc_y;

    atomicAdd(&cell_counts[cell_index], 1);
    cell_indices[i] = cell_index;

    float lfx = fx * nc_x - cx;
    lfx = fmaxf(0.0f, fminf(lfx, 1.0f - 1e-6f));
    float lfy = fy * nc_y - cy;
    lfy = fmaxf(0.0f, fminf(lfy, 1.0f - 1e-6f));
    float lfz = fz * nc_z - cz;
    lfz = fmaxf(0.0f, fminf(lfz, 1.0f - 1e-6f));

    /* intra-cell Hilbert key: 2^L sub-cells per axis, 3*L-bit key.
     * L is density-derived (see _compute_cell_grid). */
    const int L = hilbert_L;
    unsigned int n = (unsigned int)(1 << L);
    unsigned int lx = min((unsigned int)(lfx * (float)n), n - 1u);
    unsigned int ly = min((unsigned int)(lfy * (float)n), n - 1u);
    unsigned int lz = min((unsigned int)(lfz * (float)n), n - 1u);
    unsigned int h = hilbert_encode(lx, ly, lz, L);

    unsigned long long ckey = ((unsigned long long)cell_index << (3 * L)) | (unsigned long long)h;
    sort_keys[i] = ckey;
    atomicAdd(&composite_counts[(int)ckey], 1);
}
"""

_BLOCK_META_KERNEL = r"""
extern "C" __global__
void block_meta_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    const int* __restrict__ block_atoms,
    const int* __restrict__ d_num_blocks, int num_particles,
    float* __restrict__ center_x, float* __restrict__ center_y, float* __restrict__ center_z,
    float* __restrict__ size_x, float* __restrict__ size_y, float* __restrict__ size_z,
    int* __restrict__ atom_to_block,
    int* __restrict__ atom_to_slot
) {
    __shared__ int s_num_blocks;
    if (threadIdx.x == 0) s_num_blocks = d_num_blocks[0];
    __syncthreads();
    int num_blocks = s_num_blocks;
    int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= num_blocks) return;
    float minx=1e30f,miny=1e30f,minz=1e30f,maxx=-1e30f,maxy=-1e30f,maxz=-1e30f;
    for (int s = 0; s < 32; s++) {
        int a = block_atoms[bi * 32 + s];
        if (a < 0) continue;
        float x = pos_x[a], y = pos_y[a], z = pos_z[a];
        minx = fminf(minx, x); maxx = fmaxf(maxx, x);
        miny = fminf(miny, y); maxy = fmaxf(maxy, y);
        minz = fminf(minz, z); maxz = fmaxf(maxz, z);
        atom_to_block[a] = bi;
        atom_to_slot[a] = s;
    }
    center_x[bi] = 0.5f*(minx+maxx); center_y[bi] = 0.5f*(miny+maxy); center_z[bi] = 0.5f*(minz+maxz);
    size_x[bi]   = 0.5f*(maxx-minx); size_y[bi]   = 0.5f*(maxy-miny); size_z[bi]   = 0.5f*(maxz-minz);
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
    const int* __restrict__ d_num_blocks, int num_particles, int K,
    float build_radius_sq,
    const float* __restrict__ pbc_matrix,
    int* __restrict__ block_pairs_out,
    int* __restrict__ interacting_atoms_out,
    float* __restrict__ shift_x_out,
    float* __restrict__ shift_y_out,
    float* __restrict__ shift_z_out,
    int* __restrict__ interaction_count,
    int max_block_pairs
) {
    __shared__ int s_num_blocks;
    if (threadIdx.x == 0) s_num_blocks = d_num_blocks[0];
    __syncthreads();
    int num_blocks = s_num_blocks;
    int tgx = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int global_warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (global_warp >= num_blocks * K) return;

    int bx = global_warp / K;
    int cell_subset = global_warp % K;
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

    // Shift-grouped packing: track the previous cell's shift so we can
    // drain the buffer only when the PBC shift actually changes.  This
    // lets atoms from same-shift neighbour cells fill tiles together.
    float prev_sx = 0.0f, prev_sy = 0.0f, prev_sz = 0.0f;
    bool has_prev_shift = false;

    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int cell_id = (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1);
                if (K > 1 && cell_id % K != cell_subset) continue;
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

                // Drain-on-shift-change: if this cell's shift differs
                // from the previous cell's, flush the buffer first so
                // that all atoms in the buffer share one shift value.
                if (has_prev_shift && (sx != prev_sx || sy != prev_sy || sz != prev_sz)) {
                    if (nBuf > 0) {
                        int ti_drain = 0;
                        if (tgx == 0) ti_drain = atomicAdd(interaction_count, 1);
                        ti_drain = __shfl_sync(0xffffffff, ti_drain, 0);
                        if (ti_drain < max_block_pairs) {
                            if (tgx < 1) {
                                block_pairs_out[ti_drain] = bx;
                                shift_x_out[ti_drain] = prev_sx;
                                shift_y_out[ti_drain] = prev_sy;
                                shift_z_out[ti_drain] = prev_sz;
                            }
                            interacting_atoms_out[ti_drain * 32 + tgx] =
                                (tgx < nBuf) ? my_buf[tgx] : -1;
                        }
                        nBuf = 0;
                    }
                }
                prev_sx = sx; prev_sy = sy; prev_sz = sz;
                has_prev_shift = true;

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
                            if (ti < max_block_pairs) {
                                if (tgx < 1) {
                                    block_pairs_out[ti] = bx;
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
            }
        }
    }

    // Final drain: flush remaining atoms from the last shift group.
    // Empties the buffer before the self-cell path (which uses zero shift).
    if (nBuf > 0) {
        int ti_final = 0;
        if (tgx == 0) ti_final = atomicAdd(interaction_count, 1);
        ti_final = __shfl_sync(0xffffffff, ti_final, 0);
        if (ti_final < max_block_pairs) {
            if (tgx < 1) {
                block_pairs_out[ti_final] = bx;
                shift_x_out[ti_final] = prev_sx;
                shift_y_out[ti_final] = prev_sy;
                shift_z_out[ti_final] = prev_sz;
            }
            interacting_atoms_out[ti_final * 32 + tgx] =
                (tgx < nBuf) ? my_buf[tgx] : -1;
        }
        nBuf = 0;
    }

    if (cell_subset == 0) {
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
            if (ti < max_block_pairs) {
                if (tgx < 1) {
                    block_pairs_out[ti] = bx;
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

    if (cell_subset == 0 && nBuf > 0) {
        int ti = 0;
        if (tgx == 0) ti = atomicAdd(interaction_count, 1);
        ti = __shfl_sync(0xffffffff, ti, 0);
        if (ti < max_block_pairs) {
            if (tgx < 1) {
                block_pairs_out[ti] = bx;
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
    const int* __restrict__ block_pairs,
    const int* __restrict__ interacting_atoms,
    const int* __restrict__ block_atoms,
    const int* __restrict__ atom_to_block,
    const int* __restrict__ atom_to_slot,
    const int* __restrict__ exclusion_offset,
    const int* __restrict__ exclusion_neighbors,
    const int* __restrict__ reverse_offset,
    const int* __restrict__ reverse_neighbors,
    const int* __restrict__ d_block_pair_count,
    int num_particles,
    unsigned int* __restrict__ exclusion_masks_out
) {
    __shared__ int s_num_pairs;
    if (threadIdx.x == 0) s_num_pairs = d_block_pair_count[0];
    __syncthreads();
    int num_block_pairs = s_num_pairs;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_block_pairs * 32) return;

    int pair_idx = idx / 32;
    int slot_j = idx % 32;
    int block_x = block_pairs[pair_idx];
    int atom_j = interacting_atoms[pair_idx * 32 + slot_j];

    unsigned int excl = 0;

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
                excl |= (1u << sn);
            }
        }

        s = reverse_offset[atom_j];
        e = reverse_offset[atom_j + 1];
        for (int k = s; k < e; k++) {
            int nb = reverse_neighbors[k];
            if (nb < 0 || nb >= num_particles) continue;
            if (atom_to_block[nb] == block_x) {
                int sn = atom_to_slot[nb];
                excl |= (1u << sn);
            }
        }
    }

    exclusion_masks_out[idx] = excl;
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
    int* __restrict__ rebuild_flag
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    float dx = pos_x[idx] - old_pos_x[idx];
    float dy = pos_y[idx] - old_pos_y[idx];
    float dz = pos_z[idx] - old_pos_z[idx];
    if (dx*dx + dy*dy + dz*dz > threshold_sq)
        rebuild_flag[0] = 1;
}
"""

_CELL_PREFIX_SUM_KERNEL = _BLOCK_SCAN_PREAMBLE + r"""
extern "C" __global__
void cell_prefix_sum_kernel(
    const int* __restrict__ cell_counts,
    int nc_total,
    int* __restrict__ cell_offset,
    int* __restrict__ cell_block_offset,
    int* __restrict__ cell_block_count,
    int* __restrict__ cell_offset_padded,
    int* __restrict__ block_to_cell,
    int* __restrict__ num_blocks_out,
    int* __restrict__ total_padded_out
) {
    const int tid = threadIdx.x;
    const int B = SCAN_BLOCK;
    __shared__ int s_part[SCAN_BLOCK];
    __shared__ int s_total_atoms;
    __shared__ int s_total_blocks;

    // Scan 1: cell_counts -> cell_offset (exclusive) + total atom count.
    scan_block_excl(cell_counts, cell_offset, nc_total, s_part, &s_total_atoms);
    __syncthreads();

    // Element-wise: cell_block_count[c] = ceil(cell_counts[c] / 32).
    const int n_per = (nc_total + B - 1) / B;
    const int lo = tid * n_per;
    const int hi = (lo + n_per < nc_total) ? (lo + n_per) : nc_total;
    for (int j = lo; j < hi; ++j)
        cell_block_count[j] = (cell_counts[j] + 31) / 32;
    __syncthreads();

    // Scan 2: cell_block_count -> cell_block_offset (exclusive) + total blocks.
    scan_block_excl(cell_block_count, cell_block_offset, nc_total, s_part,
                    &s_total_blocks);
    __syncthreads();

    // Element-wise padded offsets.
    for (int j = lo; j < hi; ++j)
        cell_offset_padded[j] = cell_block_offset[j] * 32;

    // Sentinel slots + scalar totals (single writer, thread 0).
    if (tid == 0) {
        cell_offset[nc_total] = s_total_atoms;
        cell_block_offset[nc_total] = s_total_blocks;
        cell_offset_padded[nc_total] = s_total_blocks * 32;
        *num_blocks_out = s_total_blocks;
        *total_padded_out = s_total_blocks * 32;
    }
    __syncthreads();  // cell_block_offset[] fully visible before binary search

    // Scatter-fill block_to_cell[b] = owning cell. Each output block index b
    // is mapped to its cell by an upper-bound search over the monotonic
    // cell_block_offset (largest c with cell_block_offset[c] <= b).
    const int b_per = (s_total_blocks + B - 1) / B;
    const int blo = tid * b_per;
    const int bhi = (blo + b_per < s_total_blocks) ? (blo + b_per) : s_total_blocks;
    for (int b = blo; b < bhi; ++b) {
        int l = 0, r = nc_total;
        while (l < r) {
            int m = (l + r + 1) >> 1;
            if (cell_block_offset[m] <= b) l = m; else r = m - 1;
        }
        block_to_cell[b] = l;
    }
}
"""

_COMPOSITE_PREFIX_SUM_KERNEL = _BLOCK_SCAN_PREAMBLE + r"""
extern "C" __global__
void composite_prefix_sum_kernel(
    const int* __restrict__ composite_counts,
    int K,
    int* __restrict__ composite_offset
) {
    __shared__ int s_part[SCAN_BLOCK];
    __shared__ int s_total;
    scan_block_excl(composite_counts, composite_offset, K, s_part, &s_total);
    if (threadIdx.x == 0) composite_offset[K] = s_total;
}
"""

_COUNTING_SCATTER_KERNEL = r"""
extern "C" __global__
void counting_scatter_kernel(
    const int* __restrict__ cell_indices,
    const unsigned long long* __restrict__ sort_keys,
    const int* __restrict__ composite_offset,
    int* __restrict__ composite_cursor,
    const int* __restrict__ cell_offset,
    const int* __restrict__ cell_offset_padded,
    const float* __restrict__ src_x,
    const float* __restrict__ src_y,
    const float* __restrict__ src_z,
    int number_particles,
    float* __restrict__ dst_x, float* __restrict__ dst_y, float* __restrict__ dst_z,
    int* __restrict__ block_atoms,
    int* __restrict__ raw_order,
    int* __restrict__ pdb_to_sorted,
    int* __restrict__ sorted_to_pdb,
    int* __restrict__ cell_indices_sorted
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= number_particles) return;
    int cell = cell_indices[i];
    int ckey = (int)sort_keys[i];
    int local = atomicAdd(&composite_cursor[ckey], 1);
    int slot = composite_offset[ckey] + local;
    int intra = slot - cell_offset[cell];
    int padded = cell_offset_padded[cell] + intra;
    dst_x[slot] = src_x[i]; dst_y[slot] = src_y[i]; dst_z[slot] = src_z[i];
    block_atoms[padded] = slot;
    raw_order[slot] = i;
    pdb_to_sorted[i] = slot;
    sorted_to_pdb[slot] = i;
    cell_indices_sorted[slot] = cell;
}
"""

_CONDITIONAL_FILL_INT = r"""
extern "C" __global__
void conditional_fill_int_kernel(
    int* __restrict__ buf,
    int n,
    int fill_value
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = fill_value;
}
"""

_CAPTURE_SNAPSHOT_KERNEL = r"""
extern "C" __global__
void capture_snapshot_kernel(
    const float* __restrict__ src_x,
    const float* __restrict__ src_y,
    const float* __restrict__ src_z,
    float* __restrict__ dst_x,
    float* __restrict__ dst_y,
    float* __restrict__ dst_z,
    int num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    dst_x[idx] = src_x[idx];
    dst_y[idx] = src_y[idx];
    dst_z[idx] = src_z[idx];
}
"""


def _compile_gpu_kernels():
    return {
        "cell_assign": cp.RawKernel(_CELL_ASSIGN_KERNEL, "cell_assign_kernel"),
        "block_meta": cp.RawKernel(_BLOCK_META_KERNEL, "block_meta_kernel"),
        "find_interacting": cp.RawKernel(
            _FIND_INTERACTING_BLOCKS_KERNEL, "find_interacting_blocks_kernel"
        ),
        "rev_count": cp.RawKernel(_BUILD_REVERSE_COUNT_KERNEL, "build_reverse_count_kernel"),
        "rev_fill": cp.RawKernel(_FILL_REVERSE_KERNEL, "fill_reverse_kernel"),
        "build_masks": cp.RawKernel(_BUILD_MASKS_KERNEL, "build_masks_kernel"),
        "check_rebuild": cp.RawKernel(_CHECK_REBUILD_KERNEL, "check_rebuild_kernel"),
        "counting_scatter": cp.RawKernel(_COUNTING_SCATTER_KERNEL, "counting_scatter_kernel"),
        "cell_prefix_sum": cp.RawKernel(_CELL_PREFIX_SUM_KERNEL, "cell_prefix_sum_kernel"),
        "composite_prefix_sum": cp.RawKernel(_COMPOSITE_PREFIX_SUM_KERNEL, "composite_prefix_sum_kernel"),
        "conditional_fill": cp.RawKernel(_CONDITIONAL_FILL_INT, "conditional_fill_int_kernel"),
        "capture_snapshot": cp.RawKernel(_CAPTURE_SNAPSHOT_KERNEL, "capture_snapshot_kernel"),
    }


class BlockList:

    def __init__(self, cutoff: float, skin: float = 1.0, rebuild_check_interval: int = 10):
        self.cutoff = cutoff
        self.skin = skin
        self.build_radius = cutoff + skin
        self._is_initialized = False
        self.rebuild_check_interval = rebuild_check_interval

        self.num_blocks = 0
        self.num_block_pairs = 0
        self.num_particles = 0
        self._max_block_pairs = 0
        self.max_blocks = 0
        self.max_total_padded = 0

        self.nc_x = 0
        self.nc_y = 0
        self.nc_z = 0
        self.nc_total = 0
        self._hilbert_levels = 2
        self._hilbert_bits = 6

        self.d_block_atoms = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_block_center_x = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_block_center_y = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_block_center_z = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_block_size_x = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_block_size_y = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_block_size_z = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_atom_to_block = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_atom_to_slot = cp.empty(0, dtype=env.NUMPY_INT)

        self.d_block_pairs = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_interacting_atoms = cp.empty(0, dtype=env.NUMPY_INT)

        self.d_cell_block_offset = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_cell_block_count = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_block_to_cell = cp.empty(0, dtype=env.NUMPY_INT)

        self.d_raw_order = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_pdb_to_sorted = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_sorted_to_pdb = cp.empty(0, dtype=env.NUMPY_INT)

        self._d_block_pair_buf = cp.empty(0, dtype=env.NUMPY_INT)
        self._d_interacting_buf = cp.empty(0, dtype=env.NUMPY_INT)
        self._d_block_pair_shift_x_buf = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self._d_block_pair_shift_y_buf = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self._d_block_pair_shift_z_buf = cp.empty(0, dtype=env.NUMPY_FLOAT)
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
        self._d_exclusion_masks_buf = cp.empty(0, dtype=np.uint32)

        self.num_main_block_pairs = 0
        self.num_exclusion_block_pairs = 0
        self.d_main_block_pairs = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_main_interacting_atoms = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_excl_block_pairs = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_excl_interacting_atoms = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_excl_exclusion_masks = cp.empty(0, dtype=np.uint32)

        self.d_block_pair_shift_x = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_block_pair_shift_y = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_block_pair_shift_z = cp.empty(0, dtype=env.NUMPY_FLOAT)


        self.d_main_shift_x = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_main_shift_y = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_main_shift_z = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_excl_shift_x = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_excl_shift_y = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_excl_shift_z = cp.empty(0, dtype=env.NUMPY_FLOAT)

        self._exclusion_masks_np = None

        self._subgrid_dx = 0
        self._subgrid_dy = 0
        self._subgrid_dz = 0
        self._subgrid_total = 0

        self.d_rebuild_flag = cp.zeros(1, dtype=env.NUMPY_INT)
        self.d_positions_at_rebuild_x = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_positions_at_rebuild_y = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_positions_at_rebuild_z = cp.empty(0, dtype=env.NUMPY_FLOAT)

        self._pinned_int_buf = cp.cuda.alloc_pinned_memory(4)
        self._pinned_int_view = (ctypes.c_int32 * 1).from_address(
            self._pinned_int_buf.ptr
        )

        self._block_atoms_np = None
        self._block_pairs_np = None
        self._interacting_atoms_np = None

        self._pool = {}

    def _pool_get(self, name, size, dtype, fill=None):
        """Return a reusable buffer of the given size. Allocates on first call
        or when size grows; otherwise returns the cached array. If fill is not
        None, fill the buffer (memsetAsync for 0, fill_constant for others)."""
        key = (name, dtype)
        arr = self._pool.get(key)
        if arr is None or arr.size < size:
            arr = cp.empty(size, dtype=dtype)
            self._pool[key] = arr
        arr = arr[:size]
        if fill is not None:
            if fill == 0:
                cp.cuda.runtime.memsetAsync(
                    arr.data.ptr, 0, size * arr.itemsize, cp.cuda.Stream.null.ptr
                )
            else:
                fill_constant(arr, fill)
        return arr

    def set_cutoff(self, cutoff):
        self._cutoff = float(cutoff)

    @property
    def block_atoms(self):
        if self._block_atoms_np is None and self.d_block_atoms.size > 0:
            self._block_atoms_np = cp.asnumpy(self.d_block_atoms).reshape(-1, BLOCK_SIZE)
        return self._block_atoms_np

    @property
    def block_pairs(self):
        if self._block_pairs_np is None and self.d_block_pairs.size > 0:
            n = int(self._d_counters[0].get())
            self._block_pairs_np = cp.asnumpy(self.d_block_pairs[:n])
        return self._block_pairs_np

    @property
    def interacting_atoms(self):
        if self._interacting_atoms_np is None and self.d_interacting_atoms.size > 0:
            n = int(self._d_counters[0].get())
            self._interacting_atoms_np = cp.asnumpy(
                self.d_interacting_atoms[: n * BLOCK_SIZE]
            ).reshape(-1, BLOCK_SIZE)
        return self._interacting_atoms_np

    @property
    def exclusion_masks(self):
        if self._exclusion_masks_np is None and self.d_exclusion_masks.size > 0:
            n = int(self._d_counters[0].get())
            self._exclusion_masks_np = cp.asnumpy(
                self.d_exclusion_masks[: n * BLOCK_SIZE]
            ).reshape(-1, BLOCK_SIZE)
        return self._exclusion_masks_np

    def _invalidate_caches(self):
        self._block_atoms_np = None
        self._block_pairs_np = None
        self._interacting_atoms_np = None
        self._exclusion_masks_np = None

    def compute_pme_subgrid_dims(self, grid_x, grid_y, grid_z, order):
        self._subgrid_dx = -(-grid_x // self.nc_x) + 2 * order
        self._subgrid_dy = -(-grid_y // self.nc_y) + 2 * order
        self._subgrid_dz = -(-grid_z // self.nc_z) + 2 * order
        self._subgrid_total = self._subgrid_dx * self._subgrid_dy * self._subgrid_dz

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

    def _read_device_int(self, d_int32):
        cp.cuda.runtime.memcpyAsync(
            self._pinned_int_buf.ptr,
            d_int32.data.ptr,
            4,
            cp.cuda.runtime.memcpyDeviceToHost,
            cp.cuda.Stream.null.ptr,
        )
        cp.cuda.Stream.null.synchronize()
        return int(self._pinned_int_view[0])

    def _compute_cell_grid(self, pbc_matrix, num_particles):
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

        # Upper bounds for pre-allocation: num_blocks = sum of ceil(count_c/32)
        # over all cells.  Since ceil(x) < x+1, num_blocks < N/32 + nc_total,
        # so ceil(N/32) + nc_total is a safe integer upper bound (strict >).
        self.max_blocks = (num_particles + BLOCK_SIZE - 1) // BLOCK_SIZE + self.nc_total
        self.max_total_padded = self.max_blocks * BLOCK_SIZE

        # Density-derived Hilbert level L. Targets ~4 atoms per sub-cell:
        # atoms_per_cell / (2^L)^3 ~= 4  ->  B_raw = round(log2(apc/4)), L =
        # (B_raw+2)//3 clamped to [1, 4]. Also cap K = nc_total * 2^(3L) at
        # 1e6 so the counting-sort bucket arrays stay small.
        atoms_per_cell = num_particles / self.nc_total
        b_raw = int(round(math.log2(max(1.0, atoms_per_cell / 4.0))))
        L = max(1, min(4, (b_raw + 2) // 3))
        while L > 1 and self.nc_total * (1 << (3 * L)) > 1_000_000:
            L -= 1
        self._hilbert_levels = L
        self._hilbert_bits = 3 * L

    def rebuild(self, positions, topology, pbc_matrix, pbc_inv, *, force=False):
        """Sort particles into cell-aligned blocks. Returns (pdb_to_sorted, None).

        When force=False (default), d_rebuild_flag is left untouched — the
        conditional rebuild path in update_neighbor_list reads the flag set
        by check_rebuild_async. When force=True, flag is set to 1 so
        cell_assign runs unconditionally (first build, minimize, forced
        rebuild).

        Note: this method no longer captures the rebuild-baseline snapshot.
        Call capture_snapshot() after PBC wrapping to set the displacement
        check baseline. System._do_rebuild does this automatically.
        """
        N = topology.num_particles
        if N == 0:
            self._init_empty()
            return None, None

        prev_sorted_to_pdb = self.d_sorted_to_pdb if self.d_sorted_to_pdb.size == N else None

        self._ensure_kernels()
        self._invalidate_caches()
        self.num_particles = N
        tpb = 256

        self._compute_cell_grid(pbc_matrix, N)
        self._upload_pbc(pbc_matrix, pbc_inv)

        if force:
            self.d_rebuild_flag[0] = 1

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

        # K1: Fused cell-assign (cell_index + Hilbert key + atomic cell_counts
        # and composite_counts). The composite key (cell << (3*L)) | hilbert
        # indexes a finer bucket grid so counting_scatter can preserve the
        # intra-cell Hilbert ordering instead of scattering in arrival order.
        hilbert_levels = self._hilbert_levels
        composite_buckets = self.nc_total * (1 << (3 * hilbert_levels))
        sort_keys = self._pool_get("sort_keys", N, np.uint64)
        cell_indices = self._pool_get("cell_indices", N, env.NUMPY_INT)
        d_cell_counts = self._pool_get("cell_counts", self.nc_total, env.NUMPY_INT, fill=0)
        d_composite_counts = self._pool_get("composite_counts", composite_buckets, env.NUMPY_INT, fill=0)
        nm = (N + tpb - 1) // tpb
        self._kernels["cell_assign"](
            (nm,), (tpb,),
            (
                pos_x, pos_y, pos_z,
                self._d_pbc_matrix, self._d_pbc_inv,
                np.int32(N),
                np.int32(self.nc_x), np.int32(self.nc_y), np.int32(self.nc_z),
                np.int32(hilbert_levels),
                d_cell_counts, d_composite_counts, sort_keys, cell_indices,
            ),
        )
        self._d_cell_counts = d_cell_counts
        self._d_cell_indices = cell_indices

        # K2: prefix sum over cell_counts -> cell_offset, padded block layout,
        # block_to_cell, num_blocks. Runs BEFORE the scatter because
        # counting_scatter needs cell_offset and cell_offset_padded. prefix_sum
        # consumes only d_cell_counts (not the sorted data), so it is safe to
        # run immediately after cell_assign.
        cell_offset = self._pool_get("cell_offset", self.nc_total + 1, env.NUMPY_INT)
        cell_block_offset = self._pool_get("cell_block_offset", self.nc_total + 1, env.NUMPY_INT)
        cell_block_count = self._pool_get("cell_block_count", self.nc_total, env.NUMPY_INT)
        cell_offset_padded = self._pool_get("cell_offset_padded", self.nc_total + 1, env.NUMPY_INT)
        d_num_blocks = self._pool_get("num_blocks", 1, env.NUMPY_INT)
        d_total_padded = self._pool_get("total_padded", 1, env.NUMPY_INT)
        # num_blocks is unknown until the prefix sum writes it; num_blocks <= N
        # (each block holds >=1 atom), so N is a safe upper bound. Sliced below.
        block_to_cell = self._pool_get("block_to_cell", N, env.NUMPY_INT)
        self._kernels["cell_prefix_sum"](
            (1,), (SCAN_BLOCK,),
            (
                d_cell_counts, np.int32(self.nc_total),
                cell_offset, cell_block_offset, cell_block_count,
                cell_offset_padded, block_to_cell, d_num_blocks, d_total_padded,
            ),
        )
        self.num_blocks = self.max_blocks
        self._d_num_blocks = d_num_blocks

        self.d_cell_block_offset = cell_block_offset
        self.d_cell_block_count = cell_block_count

        # K2b: prefix sum over composite_counts (cell x Hilbert buckets) so
        # counting_scatter scatters atoms in Hilbert order within each cell.
        # The composite buckets for a cell are contiguous (cell c occupies keys
        # [c*2^(3L), (c+1)*2^(3L))), so composite_offset aligns with cell_offset.
        composite_offset = self._pool_get("composite_offset", composite_buckets + 1, env.NUMPY_INT)
        self._kernels["composite_prefix_sum"](
            (1,), (SCAN_BLOCK,),
            (d_composite_counts, np.int32(composite_buckets), composite_offset),
        )

        # K3: counting-sort scatter. Each atom claims a unique slot in its
        # composite (cell x Hilbert) bucket via atomicAdd on a per-bucket
        # cursor, then writes every output: sorted positions, block_atoms,
        # order maps, and cell_indices_sorted. Scattering on composite buckets
        # preserves the intra-cell
        # Hilbert ordering, keeping blocks Hilbert-compact -> tight AABBs.
        # block_atoms must be pre-filled with -1 (padding) before launch.
        block_atoms = self._pool_get("block_atoms", self.max_total_padded, env.NUMPY_INT)
        n_ba = max((self.max_total_padded + tpb - 1) // tpb, 1)
        self._kernels["conditional_fill"](
            (n_ba,), (tpb,),
            (block_atoms,
             np.int32(self.max_total_padded), np.int32(-1)),
        )
        composite_cursor = self._pool_get("composite_cursor", composite_buckets, env.NUMPY_INT, fill=0)
        sorted_pos_x = self._pool_get("sorted_pos_x", N, env.NUMPY_FLOAT)
        sorted_pos_y = self._pool_get("sorted_pos_y", N, env.NUMPY_FLOAT)
        sorted_pos_z = self._pool_get("sorted_pos_z", N, env.NUMPY_FLOAT)
        raw_order = self._pool_get("raw_order", N, env.NUMPY_INT)
        pdb_to_sorted = self._pool_get("pdb_to_sorted", N, env.NUMPY_INT)
        # sorted_to_pdb is NOT pooled: line 808 captures prev_sorted_to_pdb =
        # self.d_sorted_to_pdb, and counting_scatter below overwrites the
        # buffer. If pooled, prev_sorted_to_pdb would alias the same buffer
        # and be corrupted before line 971 uses it.
        sorted_to_pdb = cp.empty(N, dtype=env.NUMPY_INT)
        cell_indices_sorted = self._pool_get("cell_indices_sorted", N, env.NUMPY_INT)
        self._kernels["counting_scatter"](
            (nm,), (tpb,),
            (
                cell_indices, sort_keys, composite_offset, composite_cursor,
                cell_offset, cell_offset_padded,
                pos_x, pos_y, pos_z, np.int32(N),
                sorted_pos_x, sorted_pos_y, sorted_pos_z,
                block_atoms, raw_order, pdb_to_sorted, sorted_to_pdb,
                cell_indices_sorted,
            ),
        )
        self.d_block_atoms = block_atoms
        self.d_raw_order = raw_order
        self.d_pdb_to_sorted = pdb_to_sorted
        self.d_sorted_to_pdb = sorted_to_pdb
        self._d_cell_indices_sorted = cell_indices_sorted
        pos_x, pos_y, pos_z = sorted_pos_x, sorted_pos_y, sorted_pos_z
        self._sorted_positions = (pos_x, pos_y, pos_z)

        self.d_block_to_cell = block_to_cell[:self.max_blocks]

        nb = (self.max_blocks + tpb - 1) // tpb
        self.d_block_center_x = self._pool_get("block_center_x", self.max_blocks, env.NUMPY_FLOAT)
        self.d_block_center_y = self._pool_get("block_center_y", self.max_blocks, env.NUMPY_FLOAT)
        self.d_block_center_z = self._pool_get("block_center_z", self.max_blocks, env.NUMPY_FLOAT)
        self.d_block_size_x = self._pool_get("block_size_x", self.max_blocks, env.NUMPY_FLOAT)
        self.d_block_size_y = self._pool_get("block_size_y", self.max_blocks, env.NUMPY_FLOAT)
        self.d_block_size_z = self._pool_get("block_size_z", self.max_blocks, env.NUMPY_FLOAT)
        # K4: compute block AABB bounds and atom_to_block/slot reverse map in a
        # single per-block pass. Every real atom is in exactly one block at one
        # slot, so all N entries are written here -> no -1 pre-fill needed.
        self.d_atom_to_block = self._pool_get("atom_to_block", N, env.NUMPY_INT)
        self.d_atom_to_slot = self._pool_get("atom_to_slot", N, env.NUMPY_INT)
        self._kernels["block_meta"](
            (nb,), (tpb,),
            (
                pos_x, pos_y, pos_z, self.d_block_atoms,
                d_num_blocks, np.int32(N),
                self.d_block_center_x, self.d_block_center_y, self.d_block_center_z,
                self.d_block_size_x, self.d_block_size_y, self.d_block_size_z,
                self.d_atom_to_block, self.d_atom_to_slot,
            ),
        )

        self._d_reverse_offset = None
        self._d_reverse_neighbors = None
        self._d_reverse_scale = None

        self._is_initialized = True

        raw_order = self.d_raw_order
        if prev_sorted_to_pdb is not None:
            self.d_sorted_to_pdb = prev_sorted_to_pdb[raw_order]

        return self.d_pdb_to_sorted, None

    def build_block_pairs(self, topology, pbc_matrix):
        """Find interacting block pairs using cell-based neighbor search."""
        if self.num_particles == 0 or not hasattr(self, '_sorted_positions'):
            return
        self._invalidate_caches()

        pos_x, pos_y, pos_z = self._sorted_positions
        num_blocks = self.num_blocks
        # Cell-subset decomposition: split the 27-cell scan into K subsets
        # to fill the GPU. Target ~2 full waves (80 SMs x 4 blocks/SM = 320/wave).
        target_total_warps = 640 * 8
        cell_subsets = max(1, min(8, (target_total_warps + self.max_blocks - 1) // self.max_blocks))
        build_radius_sq = self.build_radius ** 2

        max_block_pairs = max(num_blocks * 100, 10000)
        if self._d_block_pair_buf.size < max_block_pairs:
            self._d_block_pair_buf = cp.empty(max_block_pairs, dtype=env.NUMPY_INT)
            self._d_interacting_buf = cp.empty(max_block_pairs * BLOCK_SIZE, dtype=env.NUMPY_INT)
            self._d_block_pair_shift_x_buf = cp.empty(max_block_pairs, dtype=env.NUMPY_FLOAT)
            self._d_block_pair_shift_y_buf = cp.empty(max_block_pairs, dtype=env.NUMPY_FLOAT)
            self._d_block_pair_shift_z_buf = cp.empty(max_block_pairs, dtype=env.NUMPY_FLOAT)
            self._max_block_pairs = max_block_pairs
        self._d_counters[0] = 0

        tpb = 256
        grid_blocks = max((self.max_blocks * cell_subsets + 7) // 8, 1)
        d_num_blocks = self._pool.get(("num_blocks", env.NUMPY_INT))
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
                d_num_blocks, np.int32(self.num_particles),
                np.int32(cell_subsets),
                np.float32(build_radius_sq),
                self._d_pbc_matrix,
                self._d_block_pair_buf, self._d_interacting_buf,
                self._d_block_pair_shift_x_buf, self._d_block_pair_shift_y_buf, self._d_block_pair_shift_z_buf,
                self._d_counters, np.int32(max_block_pairs),
            ),
        )

        self.num_block_pairs = self._max_block_pairs
        self.num_cell_subsets = cell_subsets
        self.d_block_pairs = self._d_block_pair_buf
        self.d_interacting_atoms = self._d_interacting_buf
        self.d_block_pair_shift_x = self._d_block_pair_shift_x_buf
        self.d_block_pair_shift_y = self._d_block_pair_shift_y_buf
        self.d_block_pair_shift_z = self._d_block_pair_shift_z_buf

        self._build_masks_gpu(topology)
        self._extract_exclusion_block_pairs()

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
        if self.num_block_pairs == 0:
            self.d_exclusion_masks = cp.empty(0, dtype=np.uint32)
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

        max_total_work = self._max_block_pairs * BLOCK_SIZE
        grid = ((max_total_work + tpb - 1) // tpb,)
        if self._d_exclusion_masks_buf.size < max_total_work:
            self._d_exclusion_masks_buf = cp.empty(max_total_work, dtype=np.uint32)
        self.d_exclusion_masks = self._d_exclusion_masks_buf
        self._kernels["build_masks"](
            grid,
            (tpb,),
            (
                self.d_block_pairs,
                self.d_interacting_atoms,
                self.d_block_atoms,
                self.d_atom_to_block,
                self.d_atom_to_slot,
                self._d_excl_offset,
                self._d_excl_neighbors,
                self._d_reverse_offset,
                self._d_reverse_neighbors,
                self._d_counters,
                np.int32(N),
                self.d_exclusion_masks,
            ),
        )

    def _extract_exclusion_block_pairs(self):
        # Phase 2: unified mask path. The exclusion kernel handles ALL pairs
        # (a zero mask = no exclusion = full force). Publish the unified arrays
        # under the main_* names; set excl count to 0 so only one kernel launches.
        if self.num_block_pairs == 0:
            self.num_exclusion_block_pairs = 0
            self.num_main_block_pairs = 0
            return
        self.num_main_block_pairs = self.num_block_pairs
        self.num_exclusion_block_pairs = 0
        # main arrays alias the raw find_interacting output
        self.d_main_block_pairs = self.d_block_pairs
        self.d_main_interacting_atoms = self.d_interacting_atoms
        self.d_main_shift_x = self.d_block_pair_shift_x
        self.d_main_shift_y = self.d_block_pair_shift_y
        self.d_main_shift_z = self.d_block_pair_shift_z
        # masks: the exclusion kernel reads d_excl_exclusion_masks for all pairs
        self.d_excl_block_pairs = self.d_block_pairs
        self.d_excl_interacting_atoms = self.d_interacting_atoms
        self.d_excl_exclusion_masks = self.d_exclusion_masks
        self.d_excl_shift_x = self.d_block_pair_shift_x
        self.d_excl_shift_y = self.d_block_pair_shift_y
        self.d_excl_shift_z = self.d_block_pair_shift_z

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
                self.d_rebuild_flag,
            ),
        )
        flag = self._read_device_int(self.d_rebuild_flag)
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
                self.d_rebuild_flag,
            ),
        )
        return False

    def capture_snapshot(self, positions_soa):
        """Capture current positions as the rebuild-baseline snapshot.

        Must be called AFTER pbc wrapping so the snapshot is in the same
        PBC image as subsequent positions. This ensures check_rebuild
        measures true cumulative drift, not wrap-artifact coordinate jumps.
        """
        pos_x, pos_y, pos_z = positions_soa
        N = self.num_particles
        snap_x = self._pool_get("snap_x", N, env.NUMPY_FLOAT)
        snap_y = self._pool_get("snap_y", N, env.NUMPY_FLOAT)
        snap_z = self._pool_get("snap_z", N, env.NUMPY_FLOAT)
        tpb = 256
        grid = ((N + tpb - 1) // tpb,)
        self._kernels["capture_snapshot"](
            grid, (tpb,),
            (pos_x, pos_y, pos_z,
             snap_x, snap_y, snap_z, np.int32(N)),
        )
        self.d_positions_at_rebuild_x = snap_x
        self.d_positions_at_rebuild_y = snap_y
        self.d_positions_at_rebuild_z = snap_z

    def read_flag_sync(self):
        """Read d_rebuild_flag with a GPU sync. Returns 0 or 1."""
        return self._read_device_int(self.d_rebuild_flag)

    def reset_flag(self):
        """Reset d_rebuild_flag to 0. Call after reading and acting on it."""
        self.d_rebuild_flag[0] = 0

    def _init_empty(self):
        self.num_blocks = 0
        self.num_block_pairs = 0
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
        self.d_block_pairs = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_interacting_atoms = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_cell_block_offset = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_cell_block_count = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_block_to_cell = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_raw_order = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_pdb_to_sorted = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_sorted_to_pdb = cp.empty(0, dtype=env.NUMPY_INT)
        self._sorted_positions = None
        self.d_exclusion_masks = cp.empty(0, dtype=np.uint32)
        self.num_exclusion_block_pairs = 0
        self.num_main_block_pairs = 0
        self.d_excl_block_pairs = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_excl_interacting_atoms = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_excl_exclusion_masks = cp.empty(0, dtype=np.uint32)
        self.d_main_block_pairs = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_main_interacting_atoms = cp.empty(0, dtype=env.NUMPY_INT)
        self.d_block_pair_shift_x = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_block_pair_shift_y = cp.empty(0, dtype=env.NUMPY_FLOAT)
        self.d_block_pair_shift_z = cp.empty(0, dtype=env.NUMPY_FLOAT)
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
