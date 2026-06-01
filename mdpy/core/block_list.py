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
    float box_x, float box_y, float box_z,
    float inv_box_x, float inv_box_y, float inv_box_z,
    int* __restrict__ tiles_out,
    int* __restrict__ interacting_atoms_out,
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

                int b_start = cell_block_offset[nc];
                int b_count = cell_block_count[nc];
                bool is_self = (nc == my_cell);

                for (int iter = 0; iter < (b_count + 31) / 32; iter++) {
                    int bj_local = tgx + iter * 32;
                    bool include_block = false;
                    int bj = -1;

                    if (bj_local < b_count) {
                        bj = b_start + bj_local;
                        if (is_self && bj <= bx) {
                            // skip: Newton's 3rd within same cell
                        } else {
                            float bcx2 = block_center_x[bj];
                            float bcy2 = block_center_y[bj];
                            float bcz2 = block_center_z[bj];
                            float bsx2 = block_size_x[bj];
                            float bsy2 = block_size_y[bj];
                            float bsz2 = block_size_z[bj];

                            float ddx = bcx2 - mcx;
                            float ddy = bcy2 - mcy;
                            float ddz = bcz2 - mcz;
                            ddx -= box_x * roundf(ddx * inv_box_x);
                            ddy -= box_y * roundf(ddy * inv_box_y);
                            ddz -= box_z * roundf(ddz * inv_box_z);
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
                                float ddx = px_j - s_pos_x[warp_in_block][k];
                                float ddy = py_j - s_pos_y[warp_in_block][k];
                                float ddz = pz_j - s_pos_z[warp_in_block][k];
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
    }


class BlockList:

    def __init__(self, cutoff: float, skin: float = 1.0):
        self.cutoff = cutoff
        self.skin = skin
        self.build_radius = cutoff + skin
        self._is_initialized = False

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
        self._d_counters = cp.zeros(1, dtype=env.NUMPY_INT)

        self._d_pbc_matrix = None
        self._d_pbc_inv = None
        self._kernels = None

        self._block_atoms_np = None
        self._tiles_np = None
        self._interacting_atoms_np = None

        self._box_x = 0.0
        self._box_y = 0.0
        self._box_z = 0.0
        self._inv_box_x = 0.0
        self._inv_box_y = 0.0
        self._inv_box_z = 0.0

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

    def _invalidate_caches(self):
        self._block_atoms_np = None
        self._tiles_np = None
        self._interacting_atoms_np = None

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
        box_x = abs(float(pbc_2d[0, 0]))
        box_y = abs(float(pbc_2d[1, 1]))
        box_z = abs(float(pbc_2d[2, 2]))
        self._box_x = box_x
        self._box_y = box_y
        self._box_z = box_z
        self._inv_box_x = 1.0 / box_x
        self._inv_box_y = 1.0 / box_y
        self._inv_box_z = 1.0 / box_z
        cell_size = self.build_radius
        self.nc_x = max(1, int(box_x / cell_size))
        self.nc_y = max(1, int(box_y / cell_size))
        self.nc_z = max(1, int(box_z / cell_size))
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

        self._is_initialized = True
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
                np.float32(self._box_x), np.float32(self._box_y), np.float32(self._box_z),
                np.float32(self._inv_box_x), np.float32(self._inv_box_y), np.float32(self._inv_box_z),
                self._d_tile_buf, self._d_interacting_buf,
                self._d_counters, np.int32(max_tiles),
            ),
        )

        self.num_tiles = int(self._d_counters[0])
        self.d_tiles = self._d_tile_buf
        self.d_interacting_atoms = self._d_interacting_buf

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
