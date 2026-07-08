from __future__ import annotations

import cupy as cp
import numpy as np
from mdpy.minimizer._base import Minimizer

_POWER_KERNEL = r"""
extern "C" __global__
void fire_power_kernel(
    const float* __restrict__ fx,
    const float* __restrict__ fy,
    const float* __restrict__ fz,
    const float* __restrict__ vx,
    const float* __restrict__ vy,
    const float* __restrict__ vz,
    float* __restrict__ output,
    int num_particles
) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    if (i < num_particles) {
        sum = fx[i]*vx[i] + fy[i]*vy[i] + fz[i]*vz[i];
    }

    shared[tid] = sum;
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
void fire_step_kernel(
    float* __restrict__ pos_x,
    float* __restrict__ pos_y,
    float* __restrict__ pos_z,
    float* __restrict__ vel_x,
    float* __restrict__ vel_y,
    float* __restrict__ vel_z,
    const float* __restrict__ fx,
    const float* __restrict__ fy,
    const float* __restrict__ fz,
    const float* __restrict__ masses,
    float time_step,
    float alpha,
    int num_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    float mass = masses[i];
    if (mass <= 0.0f) return;

    float inv_mass = 1.0f / mass;
    float ax = fx[i] * inv_mass;
    float ay = fy[i] * inv_mass;
    float az = fz[i] * inv_mass;

    float vx = vel_x[i] + time_step * ax;
    float vy = vel_y[i] + time_step * ay;
    float vz = vel_z[i] + time_step * az;

    pos_x[i] += time_step * vx;
    pos_y[i] += time_step * vy;
    pos_z[i] += time_step * vz;

    float v_norm = sqrtf(vx*vx + vy*vy + vz*vz);
    float a_norm = sqrtf(ax*ax + ay*ay + az*az);

    float scale = 0.0f;
    if (a_norm > 0.0f && v_norm > 0.0f) {
        scale = v_norm / a_norm;
    }

    float one_minus_alpha = 1.0f - alpha;
    vel_x[i] = one_minus_alpha * vx + alpha * scale * ax;
    vel_y[i] = one_minus_alpha * vy + alpha * scale * ay;
    vel_z[i] = one_minus_alpha * vz + alpha * scale * az;
}
"""


class FIREMinimizer(Minimizer):

    def __init__(self, time_step=1.0, dt_max=10.0, f_inc=1.1, f_dec=0.5,
                 alpha_start=0.1, f_alpha=0.99, n_min=5):
        super().__init__()
        self._time_step = float(time_step)
        self._dt_max = float(dt_max)
        self._f_inc = float(f_inc)
        self._f_dec = float(f_dec)
        self._alpha_start = float(alpha_start)
        self._f_alpha = float(f_alpha)
        self._n_min = int(n_min)

        self._alpha = float(alpha_start)
        self._n_positive = 0

        self._power_kernel = None
        self._step_kernel = None
        self._power_buffer = None

        self._velocity_x = None
        self._velocity_y = None
        self._velocity_z = None

    def _ensure_power_kernel(self):
        if self._power_kernel is not None:
            return
        self._power_kernel = cp.RawKernel(_POWER_KERNEL, "fire_power_kernel")

    def _ensure_step_kernel(self):
        if self._step_kernel is not None:
            return
        self._step_kernel = cp.RawKernel(_STEP_KERNEL, "fire_step_kernel")

    def _ensure_power_buffer(self, num_blocks):
        if self._power_buffer is not None and self._power_buffer.size >= num_blocks:
            return
        self._power_buffer = cp.zeros(num_blocks, dtype=cp.float32)

    def _ensure_buffers(self, num_particles):
        if self._velocity_x is not None and self._velocity_x.size >= num_particles:
            return
        self._velocity_x = cp.zeros(num_particles, dtype=cp.float32)
        self._velocity_y = cp.zeros(num_particles, dtype=cp.float32)
        self._velocity_z = cp.zeros(num_particles, dtype=cp.float32)

    def step(self, system):
        state = system.state
        num_particles = state.num_particles
        threads_per_block = 256
        num_blocks = (num_particles + threads_per_block - 1) // threads_per_block

        self._ensure_buffers(num_particles)
        self._ensure_power_kernel()
        self._ensure_step_kernel()
        self._ensure_power_buffer(num_blocks)

        self._power_buffer[:] = 0.0
        shared_mem = threads_per_block * 4
        self._power_kernel(
            (num_blocks,), (threads_per_block,),
            (state.d_forces_x, state.d_forces_y, state.d_forces_z,
             self._velocity_x, self._velocity_y, self._velocity_z,
             self._power_buffer, np.int32(num_particles)),
            shared_mem=shared_mem,
        )
        power = float(cp.sum(self._power_buffer))

        if power > 0.0:
            self._n_positive += 1
            if self._n_positive > self._n_min:
                self._time_step = min(self._time_step * self._f_inc, self._dt_max)
                self._alpha = self._alpha * self._f_alpha
        else:
            self._n_positive = 0
            self._time_step = self._time_step * self._f_dec
            self._velocity_x[:] = 0.0
            self._velocity_y[:] = 0.0
            self._velocity_z[:] = 0.0
            self._alpha = self._alpha_start

        self._step_kernel(
            (num_blocks,), (threads_per_block,),
            (state.d_positions_x, state.d_positions_y, state.d_positions_z,
             self._velocity_x, self._velocity_y, self._velocity_z,
             state.d_forces_x, state.d_forces_y, state.d_forces_z,
             state.d_masses,
             np.float32(self._time_step), np.float32(self._alpha), np.int32(num_particles)),
        )
