from __future__ import annotations

import numpy as np
import cupy as cp
from mdpy import env

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

_PERMUTE_STATE_ARRAYS_KERNEL = r"""
extern "C" __global__
void permute_state_arrays_kernel(
    const float* __restrict__ src0, const float* __restrict__ src1,
    const float* __restrict__ src2, const float* __restrict__ src3,
    const float* __restrict__ src4, const float* __restrict__ src5,
    const float* __restrict__ src6, const float* __restrict__ src7,
    const float* __restrict__ src8, const float* __restrict__ src9,
    const float* __restrict__ src10, const float* __restrict__ src11,
    const float* __restrict__ src12, const float* __restrict__ src13,
    const int* __restrict__ permutation,
    int num_particles,
    float* __restrict__ dst0, float* __restrict__ dst1,
    float* __restrict__ dst2, float* __restrict__ dst3,
    float* __restrict__ dst4, float* __restrict__ dst5,
    float* __restrict__ dst6, float* __restrict__ dst7,
    float* __restrict__ dst8, float* __restrict__ dst9,
    float* __restrict__ dst10, float* __restrict__ dst11,
    float* __restrict__ dst12, float* __restrict__ dst13
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    int src_idx = permutation[idx];
    dst0[idx] = src0[src_idx];
    dst1[idx] = src1[src_idx];
    dst2[idx] = src2[src_idx];
    dst3[idx] = src3[src_idx];
    dst4[idx] = src4[src_idx];
    dst5[idx] = src5[src_idx];
    dst6[idx] = src6[src_idx];
    dst7[idx] = src7[src_idx];
    dst8[idx] = src8[src_idx];
    dst9[idx] = src9[src_idx];
    dst10[idx] = src10[src_idx];
    dst11[idx] = src11[src_idx];
    dst12[idx] = src12[src_idx];
    dst13[idx] = src13[src_idx];
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


_PBC_WRAP_INPLACE_KERNEL = r"""
extern "C" __global__
void pbc_wrap_inplace_kernel(
    float* __restrict__ pos_x,
    float* __restrict__ pos_y,
    float* __restrict__ pos_z,
    const float* __restrict__ pbc_matrix,
    const float* __restrict__ pbc_inv,
    int num_particles
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_particles) return;
    float px = pos_x[index];
    float py = pos_y[index];
    float pz = pos_z[index];
    float fx = px*pbc_inv[0] + py*pbc_inv[3] + pz*pbc_inv[6];
    float fy = px*pbc_inv[1] + py*pbc_inv[4] + pz*pbc_inv[7];
    float fz = px*pbc_inv[2] + py*pbc_inv[5] + pz*pbc_inv[8];
    fx -= floorf(fx); fy -= floorf(fy); fz -= floorf(fz);
    pos_x[index] = fx*pbc_matrix[0] + fy*pbc_matrix[3] + fz*pbc_matrix[6];
    pos_y[index] = fx*pbc_matrix[1] + fy*pbc_matrix[4] + fz*pbc_matrix[7];
    pos_z[index] = fx*pbc_matrix[2] + fy*pbc_matrix[5] + fz*pbc_matrix[8];
}
"""

_WRAP_CORRECT_KERNEL = r"""
extern "C" __global__
void wrap_correct_kernel(
    float* __restrict__ pos_x,
    float* __restrict__ pos_y,
    float* __restrict__ pos_z,
    float* __restrict__ prev_pos_x,
    float* __restrict__ prev_pos_y,
    float* __restrict__ prev_pos_z,
    const float* __restrict__ pbc_matrix,
    const float* __restrict__ pbc_inv,
    int num_particles
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_particles) return;
    float px = pos_x[index];
    float py = pos_y[index];
    float pz = pos_z[index];
    float fx = px*pbc_inv[0] + py*pbc_inv[3] + pz*pbc_inv[6];
    float fy = px*pbc_inv[1] + py*pbc_inv[4] + pz*pbc_inv[7];
    float fz = px*pbc_inv[2] + py*pbc_inv[5] + pz*pbc_inv[8];
    float fx_w = fx - floorf(fx);
    float fy_w = fy - floorf(fy);
    float fz_w = fz - floorf(fz);
    float dx = (fx_w - fx)*pbc_matrix[0] + (fy_w - fy)*pbc_matrix[3] + (fz_w - fz)*pbc_matrix[6];
    float dy = (fx_w - fx)*pbc_matrix[1] + (fy_w - fy)*pbc_matrix[4] + (fz_w - fz)*pbc_matrix[7];
    float dz = (fx_w - fx)*pbc_matrix[2] + (fy_w - fy)*pbc_matrix[5] + (fz_w - fz)*pbc_matrix[8];
    pos_x[index] += dx;
    pos_y[index] += dy;
    pos_z[index] += dz;
    prev_pos_x[index] += dx;
    prev_pos_y[index] += dy;
    prev_pos_z[index] += dz;
}
"""

_ZERO_FORCES_KERNEL = r"""
extern "C" __global__
void zero_forces_kernel(
    float* __restrict__ fx,
    float* __restrict__ fy,
    float* __restrict__ fz,
    int num_particles
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_particles) return;
    fx[index] = 0.0f;
    fy[index] = 0.0f;
    fz[index] = 0.0f;
}
"""


class GPUContext:

    def __init__(self):
        self.num_particles = 0

        # d_positions_x/y/z: particle positions. May drift to [-skin, L+skin)
        # between rebuilds. Wrapped back to [0, L) during rebuild.
        self.d_positions_x = None
        self.d_positions_y = None
        self.d_positions_z = None

        self.d_velocities_x = None
        self.d_velocities_y = None
        self.d_velocities_z = None

        self.d_forces_x = None
        self.d_forces_y = None
        self.d_forces_z = None

        self.d_prev_positions_x = None
        self.d_prev_positions_y = None
        self.d_prev_positions_z = None

        self.d_masses = None
        self.d_types = None
        self.d_charges = None
        self.d_energy = None

        self.d_pbc_matrix = None
        self.d_pbc_inv = None

        self._box_x = 0.0
        self._box_y = 0.0
        self._box_z = 0.0
        self._inv_box_x = 0.0
        self._inv_box_y = 0.0
        self._inv_box_z = 0.0

        self._permutation_kernels = None
        self._zero_forces_kernel = None
        self._wrap_kernel = None
        self._wrap_correct_kernel = None

        # Double-buffered pool for permute_state_arrays. The fused permutation
        # kernel reads src[perm[i]] and writes dst[i] in one launch — if src and
        # dst aliased the same memory the gather would corrupt. Alternating
        # between pool_A and pool_B each rebuild keeps src != dst.
        self._perm_pool_A = None
        self._perm_pool_B = None
        self._perm_pool_N = 0
        self._perm_flip = False

    def _ensure_wrap_kernel(self):
        if self._wrap_kernel is not None:
            return
        self._wrap_kernel = cp.RawKernel(
            _PBC_WRAP_INPLACE_KERNEL, "pbc_wrap_inplace_kernel"
        )

    def _ensure_wrap_correct_kernel(self):
        if self._wrap_correct_kernel is not None:
            return
        self._wrap_correct_kernel = cp.RawKernel(
            _WRAP_CORRECT_KERNEL, "wrap_correct_kernel"
        )

    def wrap_positions_with_prev_correction(self):
        self._ensure_wrap_correct_kernel()
        N = self.num_particles
        tpb = 256
        grid = ((N + tpb - 1) // tpb,)
        self._wrap_correct_kernel(
            grid,
            (tpb,),
            (
                self.d_positions_x,
                self.d_positions_y,
                self.d_positions_z,
                self.d_prev_positions_x,
                self.d_prev_positions_y,
                self.d_prev_positions_z,
                self.d_pbc_matrix,
                self.d_pbc_inv,
                np.int32(N),
            ),
        )

    def _wrap_positions_inplace(self):
        self._ensure_wrap_kernel()
        N = self.num_particles
        tpb = 256
        grid = ((N + tpb - 1) // tpb,)
        self._wrap_kernel(
            grid,
            (tpb,),
            (
                self.d_positions_x,
                self.d_positions_y,
                self.d_positions_z,
                self.d_pbc_matrix,
                self.d_pbc_inv,
                np.int32(N),
            ),
        )

    def _ensure_zero_forces_kernel(self):
        if self._zero_forces_kernel is not None:
            return
        self._zero_forces_kernel = cp.RawKernel(
            _ZERO_FORCES_KERNEL, "zero_forces_kernel"
        )

    def _ensure_permutation_kernels(self):
        if self._permutation_kernels is not None:
            return
        self._permutation_kernels = {
            "permute": cp.RawKernel(_PERMUTE_ARRAY_KERNEL, "permute_array_kernel"),
            "permute_int": cp.RawKernel(
                _PERMUTE_INT_ARRAY_KERNEL, "permute_int_array_kernel"
            ),
            "permute_2comp": cp.RawKernel(
                _PERMUTE_ARRAY_2COMP_KERNEL, "permute_array_2comp_kernel"
            ),
            "permute_state_arrays": cp.RawKernel(
                _PERMUTE_STATE_ARRAYS_KERNEL, "permute_state_arrays_kernel"
            ),
            "inverse_permute": cp.RawKernel(
                _INVERSE_PERMUTE_KERNEL, "inverse_permute_kernel"
            ),
        }

    def _ensure_perm_pool(self, N):
        """Allocate 14 float32 buffers of size N, once. Double-buffered."""
        if self._perm_pool_A is not None and self._perm_pool_N >= N:
            return
        self._perm_pool_A = [cp.empty(N, dtype=cp.float32) for _ in range(14)]
        self._perm_pool_B = [cp.empty(N, dtype=cp.float32) for _ in range(14)]
        self._perm_pool_N = N
        self._perm_flip = False

    def permute_to_sorted(
        self, permutation, arrays_float, arrays_int=None, arrays_2comp=None,
    ):
        N = permutation.size
        if N == 0:
            return
        self._ensure_permutation_kernels()
        tpb = 256
        grid = ((N + tpb - 1) // tpb,)
        for name, src in arrays_float.items():
            dst = cp.empty_like(src)
            self._permutation_kernels["permute"](
                grid, (tpb,),
                (src, permutation, np.int32(N), dst)
            )
            arrays_float[name] = dst
        if arrays_int:
            for name, src in arrays_int.items():
                dst = cp.empty_like(src)
                self._permutation_kernels["permute_int"](
                    grid, (tpb,),
                    (src, permutation, np.int32(N), dst)
                )
                arrays_int[name] = dst
        if arrays_2comp:
            for name, src in arrays_2comp.items():
                dst = cp.empty_like(src)
                self._permutation_kernels["permute_2comp"](
                    grid, (tpb,),
                    (src, permutation, np.int32(N), dst)
                )
                arrays_2comp[name] = dst

    def permute_to_sorted_inplace(
        self, permutation, src, dst
    ):
        """Permute float32 src into pre-allocated dst: dst[i] = src[permutation[i]].
        No allocation. Caller ensures dst is float32 with size >= src.size."""
        N = permutation.size
        if N == 0:
            return
        self._ensure_permutation_kernels()
        tpb = 256
        grid = ((N + tpb - 1) // tpb,)
        self._permutation_kernels["permute"](
            grid, (tpb,),
            (src, permutation, np.int32(N), dst)
        )

    def permute_state_arrays(self, permutation, name_array_pairs):
        assert (
            len(name_array_pairs) == 14
        ), f"permute_state_arrays requires 14 arrays, got {len(name_array_pairs)}"
        name_list = [p[0] for p in name_array_pairs]
        src_list = [p[1] for p in name_array_pairs]
        N = permutation.size
        if N == 0:
            return list(zip(name_list, src_list))
        self._ensure_permutation_kernels()
        tpb = 256
        grid = ((N + tpb - 1) // tpb,)
        self._ensure_perm_pool(N)
        pool = self._perm_pool_B if self._perm_flip else self._perm_pool_A
        self._perm_flip = not self._perm_flip
        dst_list = [pool[i][:N] for i in range(14)]
        self._permutation_kernels["permute_state_arrays"](
            grid,
            (tpb,),
            tuple(src_list + [permutation, np.int32(N)] + dst_list),
        )
        return list(zip(name_list, dst_list))

    def permute_from_sorted(self, sorted_to_pdb, sorted_array):
        N = sorted_array.size
        if N == 0:
            return sorted_array
        self._ensure_permutation_kernels()
        tpb = 256
        grid = ((N + tpb - 1) // tpb,)
        pdb_array = cp.empty_like(sorted_array)
        self._permutation_kernels["inverse_permute"](
            grid, (tpb,), (sorted_array, sorted_to_pdb, np.int32(N), pdb_array)
        )
        return pdb_array

    def initialize(self, topology, pbc_matrix):
        self.num_particles = topology.num_particles
        number = self.num_particles

        float_dtype = np.float32
        int_dtype = np.int32

        self.d_positions_x = cp.zeros(number, dtype=float_dtype)
        self.d_positions_y = cp.zeros(number, dtype=float_dtype)
        self.d_positions_z = cp.zeros(number, dtype=float_dtype)

        self.d_velocities_x = cp.zeros(number, dtype=float_dtype)
        self.d_velocities_y = cp.zeros(number, dtype=float_dtype)
        self.d_velocities_z = cp.zeros(number, dtype=float_dtype)

        self.d_forces_x = cp.zeros(number, dtype=float_dtype)
        self.d_forces_y = cp.zeros(number, dtype=float_dtype)
        self.d_forces_z = cp.zeros(number, dtype=float_dtype)

        self.d_prev_positions_x = cp.zeros(number, dtype=float_dtype)
        self.d_prev_positions_y = cp.zeros(number, dtype=float_dtype)
        self.d_prev_positions_z = cp.zeros(number, dtype=float_dtype)

        self.d_masses = cp.asarray(topology.masses.astype(float_dtype))
        self.d_types = cp.asarray(topology.particle_type_indices.astype(int_dtype))
        self.d_charges = cp.asarray(topology.charges.astype(float_dtype))
        self.d_energy = cp.zeros(1, dtype=float_dtype)
        self.d_energy_accumulator = None

        pbc_flat = np.ascontiguousarray(pbc_matrix, dtype=float_dtype).ravel()
        self.d_pbc_matrix = cp.asarray(pbc_flat)
        pbc_inv = np.linalg.inv(pbc_flat.reshape(3, 3))
        self.d_pbc_inv = cp.asarray(
            np.ascontiguousarray(pbc_inv, dtype=float_dtype).ravel()
        )

        pbc_2d = pbc_matrix.reshape(3, 3)
        box_x = abs(float(pbc_2d[0, 0]))
        box_y = abs(float(pbc_2d[1, 1]))
        box_z = abs(float(pbc_2d[2, 2]))
        self.set_box_dims(box_x, box_y, box_z)

    def upload_pbc(self, pbc_matrix):
        """Overwrite device PBC buffers with new pbc_matrix.

        Immediately updates d_pbc_matrix, d_pbc_inv, and box_dims.
        Does NOT trigger neighbor list rebuild.
        """
        pbc_flat = np.ascontiguousarray(
            np.asarray(pbc_matrix, dtype=np.float32)
        ).ravel()
        self.d_pbc_matrix[:] = cp.asarray(pbc_flat)
        pbc_inv = np.linalg.inv(pbc_flat.reshape(3, 3))
        self.d_pbc_inv[:] = cp.asarray(
            np.ascontiguousarray(pbc_inv, dtype=np.float32).ravel()
        )
        pbc_2d = np.asarray(pbc_matrix).reshape(3, 3)
        box_x = abs(float(pbc_2d[0, 0]))
        box_y = abs(float(pbc_2d[1, 1]))
        box_z = abs(float(pbc_2d[2, 2]))
        self.set_box_dims(box_x, box_y, box_z)

    def upload_positions(self, positions):
        data = np.ascontiguousarray(np.asarray(positions, dtype=np.float32))
        self.d_positions_x[:] = cp.asarray(data[:, 0])
        self.d_positions_y[:] = cp.asarray(data[:, 1])
        self.d_positions_z[:] = cp.asarray(data[:, 2])
        self._wrap_positions_inplace()

    def upload_velocities(self, velocities):
        data = np.ascontiguousarray(np.asarray(velocities, dtype=np.float32))
        self.d_velocities_x[:] = cp.asarray(data[:, 0])
        self.d_velocities_y[:] = cp.asarray(data[:, 1])
        self.d_velocities_z[:] = cp.asarray(data[:, 2])

    def upload_prev_positions(self, positions):
        data = np.ascontiguousarray(np.asarray(positions, dtype=np.float32))
        self.d_prev_positions_x[:] = cp.asarray(data[:, 0])
        self.d_prev_positions_y[:] = cp.asarray(data[:, 1])
        self.d_prev_positions_z[:] = cp.asarray(data[:, 2])

    def download_positions(self):
        return np.stack(
            [
                self.d_positions_x.get(),
                self.d_positions_y.get(),
                self.d_positions_z.get(),
            ],
            axis=1,
        )

    def download_velocities(self):
        return np.stack(
            [
                self.d_velocities_x.get(),
                self.d_velocities_y.get(),
                self.d_velocities_z.get(),
            ],
            axis=1,
        )

    def download_forces(self):
        return np.stack(
            [self.d_forces_x.get(), self.d_forces_y.get(), self.d_forces_z.get()],
            axis=1,
        )

    def zero_forces(self):
        self._ensure_zero_forces_kernel()
        N = self.num_particles
        tpb = 256
        grid = ((N + tpb - 1) // tpb,)
        self._zero_forces_kernel(
            grid,
            (tpb,),
            (
                self.d_forces_x,
                self.d_forces_y,
                self.d_forces_z,
                np.int32(N),
            ),
        )

    def zero_energy(self):
        self.d_energy[:] = 0

    def allocate_energy_accumulator(self, num_terms):
        self.d_energy_accumulator = cp.zeros(num_terms, dtype=np.float32)

    def accumulate_energy(self, term_index):
        self.d_energy_accumulator[term_index] = self.d_energy[0]

    def set_box_dims(self, box_x, box_y, box_z):
        self._box_x = float(box_x)
        self._box_y = float(box_y)
        self._box_z = float(box_z)
        self._inv_box_x = 1.0 / self._box_x
        self._inv_box_y = 1.0 / self._box_y
        self._inv_box_z = 1.0 / self._box_z

    @property
    def box_x(self):
        return self._box_x

    @property
    def box_y(self):
        return self._box_y

    @property
    def box_z(self):
        return self._box_z

    @property
    def inv_box_x(self):
        return self._inv_box_x

    @property
    def inv_box_y(self):
        return self._inv_box_y

    @property
    def inv_box_z(self):
        return self._inv_box_z
