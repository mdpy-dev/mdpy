from __future__ import annotations

import cupy as cp
import numpy as np

_MAX_FORCE_KERNEL = r"""
extern "C" __global__
void max_force_kernel(
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

    float max_val = 0.0f;
    if (i < num_particles) {
        float mass = masses[i];
        if (mass > 0.0f) {
            float inv_mass = 1.0f / mass;
            max_val = fmaxf(fmaxf(fabsf(fx[i] * inv_mass), fabsf(fy[i] * inv_mass)), fabsf(fz[i] * inv_mass));
        }
    }

    shared[tid] = max_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}
"""

_RMS_FORCE_KERNEL = r"""
extern "C" __global__
void rms_force_kernel(
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
            float ax = fx[i] * inv_mass;
            float ay = fy[i] * inv_mass;
            float az = fz[i] * inv_mass;
            sum_sq = ax * ax + ay * ay + az * az;
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
        atomicAdd(output, shared[0]);
    }
}
"""


class Minimizer:

    def __init__(self):
        self._max_force_kernel = None
        self._rms_force_kernel = None
        self._output_buffer = None

    def _ensure_max_force_kernel(self):
        if self._max_force_kernel is not None:
            return
        self._max_force_kernel = cp.RawKernel(_MAX_FORCE_KERNEL, "max_force_kernel")

    def _ensure_rms_force_kernel(self):
        if self._rms_force_kernel is not None:
            return
        self._rms_force_kernel = cp.RawKernel(_RMS_FORCE_KERNEL, "rms_force_kernel")

    def _ensure_output_buffer(self, num_blocks):
        if self._output_buffer is None or self._output_buffer.size < num_blocks:
            self._output_buffer = cp.zeros(num_blocks, dtype=cp.float32)

    def compute_max_force(self, system):
        state = system.state
        num_particles = state.num_particles
        threads_per_block = 256
        num_blocks = (num_particles + threads_per_block - 1) // threads_per_block
        self._ensure_output_buffer(num_blocks)
        self._ensure_max_force_kernel()
        self._output_buffer[:] = 0.0
        shared_mem = threads_per_block * 4
        self._max_force_kernel(
            (num_blocks,), (threads_per_block,),
            (state.d_forces_x, state.d_forces_y, state.d_forces_z,
             state.d_masses, self._output_buffer, np.int32(num_particles)),
            shared_mem=shared_mem,
        )
        return float(cp.max(self._output_buffer))

    def compute_rms_force(self, system):
        state = system.state
        num_particles = state.num_particles
        threads_per_block = 256
        num_blocks = (num_particles + threads_per_block - 1) // threads_per_block
        self._ensure_output_buffer(num_blocks)
        self._ensure_rms_force_kernel()
        self._output_buffer[:] = 0.0
        shared_mem = threads_per_block * 4
        self._rms_force_kernel(
            (num_blocks,), (threads_per_block,),
            (state.d_forces_x, state.d_forces_y, state.d_forces_z,
             state.d_masses, self._output_buffer, np.int32(num_particles)),
            shared_mem=shared_mem,
        )
        return float(np.sqrt(cp.sum(self._output_buffer) / num_particles))

    def step(self, system):
        raise NotImplementedError
