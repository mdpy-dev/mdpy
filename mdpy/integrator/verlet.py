from __future__ import annotations

import cupy as cp
import numpy as np

_VERLET_INIT_KERNEL = r"""
extern "C" __global__
void verlet_init_kernel(
    const float* __restrict__ positions,
    const float* __restrict__ velocities,
    const float* __restrict__ forces,
    const float* __restrict__ masses,
    float* __restrict__ prev_positions,
    float dt, float dt_sq, int number_particles
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= number_particles) return;
    float m = masses[index];
    if (m <= 0.0f) return;
    float inv_mass = 1.0f / m;
    prev_positions[index * 3 + 0] = positions[index * 3 + 0] - velocities[index * 3 + 0] * dt + 0.5f * forces[index * 3 + 0] * inv_mass * dt_sq;
    prev_positions[index * 3 + 1] = positions[index * 3 + 1] - velocities[index * 3 + 1] * dt + 0.5f * forces[index * 3 + 1] * inv_mass * dt_sq;
    prev_positions[index * 3 + 2] = positions[index * 3 + 2] - velocities[index * 3 + 2] * dt + 0.5f * forces[index * 3 + 2] * inv_mass * dt_sq;
}
"""

_VERLET_KERNEL = r"""
extern "C" __global__
void verlet_kernel(
    float* __restrict__ positions,
    float* __restrict__ prev_positions,
    const float* __restrict__ forces,
    const float* __restrict__ masses,
    float* __restrict__ velocities,
    const float* __restrict__ pbc_matrix,
    const float* __restrict__ pbc_inv,
    float dt, float dt_sq, int number_particles
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= number_particles) return;
    float m = masses[index];
    if (m <= 0.0f) return;
    float inv_mass = 1.0f / m;
    float ax = forces[index * 3 + 0] * inv_mass;
    float ay = forces[index * 3 + 1] * inv_mass;
    float az = forces[index * 3 + 2] * inv_mass;

    float cx = positions[index * 3 + 0];
    float cy = positions[index * 3 + 1];
    float cz = positions[index * 3 + 2];

    float nx = 2.0f * cx - prev_positions[index * 3 + 0] + ax * dt_sq;
    float ny = 2.0f * cy - prev_positions[index * 3 + 1] + ay * dt_sq;
    float nz = 2.0f * cz - prev_positions[index * 3 + 2] + az * dt_sq;

    velocities[index * 3 + 0] = (nx - prev_positions[index * 3 + 0]) / (2.0f * dt);
    velocities[index * 3 + 1] = (ny - prev_positions[index * 3 + 1]) / (2.0f * dt);
    velocities[index * 3 + 2] = (nz - prev_positions[index * 3 + 2]) / (2.0f * dt);

    float sx = nx * pbc_inv[0] + ny * pbc_inv[3] + nz * pbc_inv[6];
    float sy = nx * pbc_inv[1] + ny * pbc_inv[4] + nz * pbc_inv[7];
    float sz = nx * pbc_inv[2] + ny * pbc_inv[5] + nz * pbc_inv[8];
    sx -= roundf(sx);
    sy -= roundf(sy);
    sz -= roundf(sz);
    nx = sx * pbc_matrix[0] + sy * pbc_matrix[3] + sz * pbc_matrix[6];
    ny = sx * pbc_matrix[1] + sy * pbc_matrix[4] + sz * pbc_matrix[7];
    nz = sx * pbc_matrix[2] + sy * pbc_matrix[5] + sz * pbc_matrix[8];

    prev_positions[index * 3 + 0] = cx;
    prev_positions[index * 3 + 1] = cy;
    prev_positions[index * 3 + 2] = cz;
    positions[index * 3 + 0] = nx;
    positions[index * 3 + 1] = ny;
    positions[index * 3 + 2] = nz;
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
                gpu_context.d_positions,
                gpu_context.d_velocities,
                gpu_context.d_forces,
                gpu_context.d_masses,
                gpu_context.d_prev_positions,
                np.float32(self.dt),
                np.float32(self.dt_sq),
                np.int32(number),
            ))
            self._initialized = True

        _kernels['step']((grid,), (block,), (
            gpu_context.d_positions,
            gpu_context.d_prev_positions,
            gpu_context.d_forces,
            gpu_context.d_masses,
            gpu_context.d_velocities,
            gpu_context.d_pbc_matrix,
            gpu_context.d_pbc_inv,
            np.float32(self.dt),
            np.float32(self.dt_sq),
            np.int32(number),
        ))
