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
    const float* __restrict__ src12,
    const int* __restrict__ permutation,
    int num_particles,
    float* __restrict__ dst0, float* __restrict__ dst1,
    float* __restrict__ dst2, float* __restrict__ dst3,
    float* __restrict__ dst4, float* __restrict__ dst5,
    float* __restrict__ dst6, float* __restrict__ dst7,
    float* __restrict__ dst8, float* __restrict__ dst9,
    float* __restrict__ dst10, float* __restrict__ dst11,
    float* __restrict__ dst12
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

_PBC_WRAP_KERNEL = r"""
extern "C" __global__
void pbc_wrap_kernel(
    const float* __restrict__ src_x,
    const float* __restrict__ src_y,
    const float* __restrict__ src_z,
    const float* __restrict__ pbc_matrix,
    const float* __restrict__ pbc_inv,
    int number_particles,
    float* __restrict__ dst_x,
    float* __restrict__ dst_y,
    float* __restrict__ dst_z
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= number_particles) return;

    float px = src_x[index];
    float py = src_y[index];
    float pz = src_z[index];

    float fx = px * pbc_inv[0] + py * pbc_inv[3] + pz * pbc_inv[6];
    float fy = px * pbc_inv[1] + py * pbc_inv[4] + pz * pbc_inv[7];
    float fz = px * pbc_inv[2] + py * pbc_inv[5] + pz * pbc_inv[8];

    fx = fx - floorf(fx);
    fy = fy - floorf(fy);
    fz = fz - floorf(fz);

    dst_x[index] = fx * pbc_matrix[0] + fy * pbc_matrix[3] + fz * pbc_matrix[6];
    dst_y[index] = fx * pbc_matrix[1] + fy * pbc_matrix[4] + fz * pbc_matrix[7];
    dst_z[index] = fx * pbc_matrix[2] + fy * pbc_matrix[5] + fz * pbc_matrix[8];
}
"""


class GPUContext:

    def __init__(self):
        self.number_particles = 0

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
        self.d_energy = None

        self.d_pbc_matrix = None
        self.d_pbc_inv = None

        self.d_box_dims = None

        self._box_x = 0.0
        self._box_y = 0.0
        self._box_z = 0.0
        self._inv_box_x = 0.0
        self._inv_box_y = 0.0
        self._inv_box_z = 0.0

        self.d_wrapped_positions_x = None
        self.d_wrapped_positions_y = None
        self.d_wrapped_positions_z = None

        self._wrap_kernel = None

        self._permutation_kernels = None

    def _ensure_wrap_kernel(self):
        if self._wrap_kernel is not None:
            return
        self._wrap_kernel = cp.RawKernel(_PBC_WRAP_KERNEL, "pbc_wrap_kernel")

    def refresh_wrapped_positions(self):
        if self.number_particles == 0:
            return
        self._ensure_wrap_kernel()
        if self.d_wrapped_positions_x is None:
            self.d_wrapped_positions_x = cp.empty(self.number_particles, dtype=np.float32)
            self.d_wrapped_positions_y = cp.empty(self.number_particles, dtype=np.float32)
            self.d_wrapped_positions_z = cp.empty(self.number_particles, dtype=np.float32)
        N = self.number_particles
        tpb = 256
        grid = ((N + tpb - 1) // tpb,)
        self._wrap_kernel(
            grid, (tpb,),
            (
                self.d_positions_x, self.d_positions_y, self.d_positions_z,
                self.d_pbc_matrix, self.d_pbc_inv,
                np.int32(N),
                self.d_wrapped_positions_x,
                self.d_wrapped_positions_y,
                self.d_wrapped_positions_z,
            ),
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

    def permute_to_sorted(
        self, permutation, arrays_float, arrays_int=None, arrays_2comp=None
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
                grid, (tpb,), (src, permutation, np.int32(N), dst)
            )
            arrays_float[name] = dst
        if arrays_int:
            for name, src in arrays_int.items():
                dst = cp.empty_like(src)
                self._permutation_kernels["permute_int"](
                    grid, (tpb,), (src, permutation, np.int32(N), dst)
                )
                arrays_int[name] = dst
        if arrays_2comp:
            for name, src in arrays_2comp.items():
                dst = cp.empty_like(src)
                self._permutation_kernels["permute_2comp"](
                    grid, (tpb,), (src, permutation, np.int32(N), dst)
                )
                arrays_2comp[name] = dst

    def permute_state_arrays(self, permutation, name_array_pairs):
        assert (
            len(name_array_pairs) == 13
        ), f"permute_state_arrays requires 13 arrays, got {len(name_array_pairs)}"
        name_list = [p[0] for p in name_array_pairs]
        src_list = [p[1] for p in name_array_pairs]
        N = permutation.size
        if N == 0:
            return list(zip(name_list, src_list))
        self._ensure_permutation_kernels()
        tpb = 256
        grid = ((N + tpb - 1) // tpb,)
        dst_list = [cp.empty_like(src) for src in src_list]
        self._permutation_kernels["permute_state_arrays"](
            grid,
            (tpb,),
            (
                src_list[0],
                src_list[1],
                src_list[2],
                src_list[3],
                src_list[4],
                src_list[5],
                src_list[6],
                src_list[7],
                src_list[8],
                src_list[9],
                src_list[10],
                src_list[11],
                src_list[12],
                permutation,
                np.int32(N),
                dst_list[0],
                dst_list[1],
                dst_list[2],
                dst_list[3],
                dst_list[4],
                dst_list[5],
                dst_list[6],
                dst_list[7],
                dst_list[8],
                dst_list[9],
                dst_list[10],
                dst_list[11],
                dst_list[12],
            ),
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
        self.number_particles = topology.num_particles
        number = self.number_particles

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
        self.d_types = cp.asarray(topology.particle_types.astype(int_dtype))
        self.d_energy = cp.zeros(1, dtype=float_dtype)
        self.d_energy_accumulator = None

        pbc_flat = np.ascontiguousarray(pbc_matrix, dtype=float_dtype).ravel()
        self.d_pbc_matrix = cp.asarray(pbc_flat)
        pbc_inv = np.linalg.inv(pbc_flat.reshape(3, 3))
        self.d_pbc_inv = cp.asarray(
            np.ascontiguousarray(pbc_inv, dtype=float_dtype).ravel()
        )

        self.d_box_dims = cp.zeros(6, dtype=float_dtype)

        pbc_2d = pbc_matrix.reshape(3, 3)
        box_x = abs(float(pbc_2d[0, 0]))
        box_y = abs(float(pbc_2d[1, 1]))
        box_z = abs(float(pbc_2d[2, 2]))
        self.set_box_dims(box_x, box_y, box_z)

    def upload_positions(self, particle_table):
        data = np.ascontiguousarray(particle_table.positions.astype(np.float32))
        self.d_positions_x[:] = cp.asarray(data[:, 0])
        self.d_positions_y[:] = cp.asarray(data[:, 1])
        self.d_positions_z[:] = cp.asarray(data[:, 2])

    def upload_velocities(self, particle_table):
        data = np.ascontiguousarray(particle_table.velocities.astype(np.float32))
        self.d_velocities_x[:] = cp.asarray(data[:, 0])
        self.d_velocities_y[:] = cp.asarray(data[:, 1])
        self.d_velocities_z[:] = cp.asarray(data[:, 2])

    def download_positions(self, particle_table):
        pos = np.stack(
            [
                self.d_positions_x.get(),
                self.d_positions_y.get(),
                self.d_positions_z.get(),
            ],
            axis=1,
        )
        particle_table.positions[:] = pos

    def download_velocities(self, particle_table):
        vel = np.stack(
            [
                self.d_velocities_x.get(),
                self.d_velocities_y.get(),
                self.d_velocities_z.get(),
            ],
            axis=1,
        )
        particle_table.velocities[:] = vel

    def download_forces(self, particle_table):
        frc = np.stack(
            [
                self.d_forces_x.get(),
                self.d_forces_y.get(),
                self.d_forces_z.get(),
            ],
            axis=1,
        )
        particle_table.forces[:] = frc

    def zero_forces(self):
        self.d_forces_x[:] = 0
        self.d_forces_y[:] = 0
        self.d_forces_z[:] = 0

    def zero_energy(self):
        self.d_energy[:] = 0

    def allocate_energy_accumulator(self, num_terms):
        self.d_energy_accumulator = cp.zeros(num_terms, dtype=np.float32)

    def accumulate_energy(self, term_index):
        self.d_energy_accumulator[term_index] = self.d_energy[0]

    def get_positions_2d(self):
        return (self.d_positions_x, self.d_positions_y, self.d_positions_z)

    def set_box_dims(self, box_x, box_y, box_z):
        self._box_x = float(box_x)
        self._box_y = float(box_y)
        self._box_z = float(box_z)
        self._inv_box_x = 1.0 / self._box_x
        self._inv_box_y = 1.0 / self._box_y
        self._inv_box_z = 1.0 / self._box_z
        self.d_box_dims[:] = cp.array(
            [
                self._box_x,
                self._box_y,
                self._box_z,
                self._inv_box_x,
                self._inv_box_y,
                self._inv_box_z,
            ],
            dtype=np.float32,
        )
