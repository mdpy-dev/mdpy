from __future__ import annotations

import cupy as cp
import numpy as np

_VERLET_INIT_KERNEL = r"""
extern "C" __global__
void verlet_init_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    const float* __restrict__ vel_x,
    const float* __restrict__ vel_y,
    const float* __restrict__ vel_z,
    const float* __restrict__ f_x,
    const float* __restrict__ f_y,
    const float* __restrict__ f_z,
    const float* __restrict__ masses,
    float* __restrict__ prev_pos_x,
    float* __restrict__ prev_pos_y,
    float* __restrict__ prev_pos_z,
    float dt, float dt_sq, int number_particles
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= number_particles) return;
    float mass = masses[index];
    if (mass <= 0.0f) return;
    float inv_mass = 1.0f / mass;
    float half_inv = 0.5f * inv_mass * dt_sq;
    prev_pos_x[index] = pos_x[index] - vel_x[index]*dt + f_x[index]*half_inv;
    prev_pos_y[index] = pos_y[index] - vel_y[index]*dt + f_y[index]*half_inv;
    prev_pos_z[index] = pos_z[index] - vel_z[index]*dt + f_z[index]*half_inv;
}
"""

_VERLET_KERNEL = r"""
extern "C" __global__
void verlet_kernel(
    float* __restrict__ pos_x,
    float* __restrict__ pos_y,
    float* __restrict__ pos_z,
    float* __restrict__ prev_pos_x,
    float* __restrict__ prev_pos_y,
    float* __restrict__ prev_pos_z,
    const float* __restrict__ f_x,
    const float* __restrict__ f_y,
    const float* __restrict__ f_z,
    const float* __restrict__ masses,
    float* __restrict__ vel_x,
    float* __restrict__ vel_y,
    float* __restrict__ vel_z,
    const float* __restrict__ pbc_matrix,
    const float* __restrict__ pbc_inv,
    float dt, float dt_sq, int number_particles
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= number_particles) return;
    float mass = masses[index];
    if (mass <= 0.0f) return;

    float inv_mass = 1.0f / mass;
    float ax = f_x[index] * inv_mass;
    float ay = f_y[index] * inv_mass;
    float az = f_z[index] * inv_mass;

    float cx = pos_x[index];
    float cy = pos_y[index];
    float cz = pos_z[index];

    float px = prev_pos_x[index];
    float py = prev_pos_y[index];
    float pz = prev_pos_z[index];

    float nx = 2.0f*cx - px + ax*dt_sq;
    float ny = 2.0f*cy - py + ay*dt_sq;
    float nz = 2.0f*cz - pz + az*dt_sq;

    float inv_2dt = 0.5f / dt;
    vel_x[index] = (nx - px) * inv_2dt;
    vel_y[index] = (ny - py) * inv_2dt;
    vel_z[index] = (nz - pz) * inv_2dt;

    float sx = nx*pbc_inv[0] + ny*pbc_inv[3] + nz*pbc_inv[6];
    float sy = nx*pbc_inv[1] + ny*pbc_inv[4] + nz*pbc_inv[7];
    float sz = nx*pbc_inv[2] + ny*pbc_inv[5] + nz*pbc_inv[8];
    sx -= roundf(sx); sy -= roundf(sy); sz -= roundf(sz);
    nx = sx*pbc_matrix[0] + sy*pbc_matrix[3] + sz*pbc_matrix[6];
    ny = sx*pbc_matrix[1] + sy*pbc_matrix[4] + sz*pbc_matrix[7];
    nz = sx*pbc_matrix[2] + sy*pbc_matrix[5] + sz*pbc_matrix[8];

    prev_pos_x[index] = cx;
    prev_pos_y[index] = cy;
    prev_pos_z[index] = cz;
    pos_x[index] = nx;
    pos_y[index] = ny;
    pos_z[index] = nz;
}
"""

_kernels = {
    'init': cp.RawKernel(_VERLET_INIT_KERNEL, 'verlet_init_kernel'),
    'step': cp.RawKernel(_VERLET_KERNEL, 'verlet_kernel'),
}


class VerletIntegrator:

    def __init__(self, time_step):
        self.dt = float(time_step)
        self.dt_sq = self.dt * self.dt
        self._initialized = False

    def step(self, gpu_context):
        number = gpu_context.number_particles
        block = 256
        grid = (number + block - 1) // block

        if not self._initialized:
            _kernels['init']((grid,), (block,), (
                gpu_context.d_positions_x, gpu_context.d_positions_y, gpu_context.d_positions_z,
                gpu_context.d_velocities_x, gpu_context.d_velocities_y, gpu_context.d_velocities_z,
                gpu_context.d_forces_x, gpu_context.d_forces_y, gpu_context.d_forces_z,
                gpu_context.d_masses,
                gpu_context.d_prev_positions_x, gpu_context.d_prev_positions_y, gpu_context.d_prev_positions_z,
                np.float32(self.dt),
                np.float32(self.dt_sq),
                np.int32(number),
            ))
            self._initialized = True

        _kernels['step']((grid,), (block,), (
            gpu_context.d_positions_x, gpu_context.d_positions_y, gpu_context.d_positions_z,
            gpu_context.d_prev_positions_x, gpu_context.d_prev_positions_y, gpu_context.d_prev_positions_z,
            gpu_context.d_forces_x, gpu_context.d_forces_y, gpu_context.d_forces_z,
            gpu_context.d_masses,
            gpu_context.d_velocities_x, gpu_context.d_velocities_y, gpu_context.d_velocities_z,
            gpu_context.d_pbc_matrix, gpu_context.d_pbc_inv,
            np.float32(self.dt),
            np.float32(self.dt_sq),
            np.int32(number),
        ))
