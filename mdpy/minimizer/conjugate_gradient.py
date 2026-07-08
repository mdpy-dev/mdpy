from __future__ import annotations

import cupy as cp
import numpy as np
from mdpy.minimizer._base import Minimizer

_GRAD_SQ_KERNEL = r"""
extern "C" __global__
void grad_sq_reduction_kernel(
    const float* __restrict__ fx,
    const float* __restrict__ fy,
    const float* __restrict__ fz,
    const float* __restrict__ masses,
    float* __restrict__ output,
    int num_particles
) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float sum_sq = 0.0f;
    if (i < num_particles) {
        float mass = masses[i];
        if (mass > 0.0f) {
            float inv_mass = 1.0f / mass;
            float gx = fx[i] * inv_mass;
            float gy = fy[i] * inv_mass;
            float gz = fz[i] * inv_mass;
            sum_sq = gx * gx + gy * gy + gz * gz;
        }
    }

    shared[tid] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}
"""

_STEP_KERNEL = r"""
extern "C" __global__
void conjugate_gradient_step_kernel(
    float* __restrict__ pos_x,
    float* __restrict__ pos_y,
    float* __restrict__ pos_z,
    const float* __restrict__ fx,
    const float* __restrict__ fy,
    const float* __restrict__ fz,
    const float* __restrict__ masses,
    float* __restrict__ dir_x,
    float* __restrict__ dir_y,
    float* __restrict__ dir_z,
    float step_size,
    float beta,
    int num_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    float mass = masses[i];
    if (mass <= 0.0f) return;
    float inv_mass = 1.0f / mass;

    float gx = fx[i] * inv_mass;
    float gy = fy[i] * inv_mass;
    float gz = fz[i] * inv_mass;

    float dx = gx + beta * dir_x[i];
    float dy = gy + beta * dir_y[i];
    float dz = gz + beta * dir_z[i];

    pos_x[i] += step_size * dx;
    pos_y[i] += step_size * dy;
    pos_z[i] += step_size * dz;

    dir_x[i] = dx;
    dir_y[i] = dy;
    dir_z[i] = dz;
}
"""


class ConjugateGradientMinimizer(Minimizer):

    def __init__(self, step_size=0.1):
        super().__init__()
        self.step_size = float(step_size)
        self._step_kernel = None
        self._grad_sq_kernel = None
        self._direction_x = None
        self._direction_y = None
        self._direction_z = None
        self._prev_grad_sq = 0.0
        self._grad_sq_buffer = None

    def _ensure_step_kernel(self):
        if self._step_kernel is not None:
            return
        self._step_kernel = cp.RawKernel(
            _STEP_KERNEL, "conjugate_gradient_step_kernel"
        )

    def _ensure_grad_sq_kernel(self):
        if self._grad_sq_kernel is not None:
            return
        self._grad_sq_kernel = cp.RawKernel(
            _GRAD_SQ_KERNEL, "grad_sq_reduction_kernel"
        )

    def _ensure_buffers(self, num_particles):
        if self._direction_x is not None and self._direction_x.size >= num_particles:
            return
        self._direction_x = cp.zeros(num_particles, dtype=cp.float32)
        self._direction_y = cp.zeros(num_particles, dtype=cp.float32)
        self._direction_z = cp.zeros(num_particles, dtype=cp.float32)
        self._prev_grad_sq = 0.0

    def _ensure_grad_sq_buffer(self, num_blocks):
        if self._grad_sq_buffer is not None and self._grad_sq_buffer.size >= num_blocks:
            return
        self._grad_sq_buffer = cp.zeros(num_blocks, dtype=cp.float32)

    def step(self, system):
        state = system.state
        num_particles = state.num_particles
        threads_per_block = 256
        num_blocks = (num_particles + threads_per_block - 1) // threads_per_block

        self._ensure_buffers(num_particles)
        self._ensure_grad_sq_kernel()
        self._ensure_step_kernel()
        self._ensure_grad_sq_buffer(num_blocks)

        self._grad_sq_buffer[:] = 0.0
        shared_mem = threads_per_block * 4
        self._grad_sq_kernel(
            (num_blocks,), (threads_per_block,),
            (state.d_forces_x, state.d_forces_y, state.d_forces_z,
             state.d_masses, self._grad_sq_buffer, np.int32(num_particles)),
            shared_mem=shared_mem,
        )
        cur_grad_sq = float(cp.sum(self._grad_sq_buffer))

        if self._prev_grad_sq > 0.0:
            beta = cur_grad_sq / self._prev_grad_sq
        else:
            beta = 0.0

        self._prev_grad_sq = cur_grad_sq

        self._step_kernel(
            (num_blocks,), (threads_per_block,),
            (state.d_positions_x, state.d_positions_y, state.d_positions_z,
             state.d_forces_x, state.d_forces_y, state.d_forces_z,
             state.d_masses,
             self._direction_x, self._direction_y, self._direction_z,
             np.float32(self.step_size), np.float32(beta), np.int32(num_particles)),
        )
