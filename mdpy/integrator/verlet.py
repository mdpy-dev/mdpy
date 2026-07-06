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
    float time_step, float time_step_squared, int num_particles
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_particles) return;
    float mass = masses[index];
    if (mass <= 0.0f) return;
    float inv_mass = 1.0f / mass;
    float half_inv = 0.5f * inv_mass * time_step_squared;
    prev_pos_x[index] = pos_x[index] - vel_x[index]*time_step + f_x[index]*half_inv;
    prev_pos_y[index] = pos_y[index] - vel_y[index]*time_step + f_y[index]*half_inv;
    prev_pos_z[index] = pos_z[index] - vel_z[index]*time_step + f_z[index]*half_inv;
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
    float time_step, float time_step_squared, int num_particles
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_particles) return;
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

    float nx = 2.0f*cx - px + ax*time_step_squared;
    float ny = 2.0f*cy - py + ay*time_step_squared;
    float nz = 2.0f*cz - pz + az*time_step_squared;

    float dx = nx - px;
    float dy = ny - py;
    float dz = nz - pz;
    float inv_two_time_step = 0.5f / time_step;
    vel_x[index] = dx * inv_two_time_step;
    vel_y[index] = dy * inv_two_time_step;
    vel_z[index] = dz * inv_two_time_step;

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
        self.time_step = float(time_step)
        self.time_step_squared = self.time_step * self.time_step
        self._initialized = False

    def step(self, system):
        gpu = system.gpu
        number = gpu.num_particles
        block = 256
        grid = (number + block - 1) // block

        if not self._initialized:
            _kernels['init']((grid,), (block,), (
                gpu.d_positions_x, gpu.d_positions_y, gpu.d_positions_z,
                gpu.d_velocities_x, gpu.d_velocities_y, gpu.d_velocities_z,
                gpu.d_forces_x, gpu.d_forces_y, gpu.d_forces_z,
                gpu.d_masses,
                gpu.d_prev_positions_x, gpu.d_prev_positions_y, gpu.d_prev_positions_z,
                np.float32(self.time_step),
                np.float32(self.time_step_squared),
                np.int32(number),
            ))
            self._initialized = True

        _kernels['step']((grid,), (block,), (
            gpu.d_positions_x, gpu.d_positions_y, gpu.d_positions_z,
            gpu.d_prev_positions_x, gpu.d_prev_positions_y, gpu.d_prev_positions_z,
            gpu.d_forces_x, gpu.d_forces_y, gpu.d_forces_z,
            gpu.d_masses,
            gpu.d_velocities_x, gpu.d_velocities_y, gpu.d_velocities_z,
            np.float32(self.time_step),
            np.float32(self.time_step_squared),
            np.int32(number),
        ))
