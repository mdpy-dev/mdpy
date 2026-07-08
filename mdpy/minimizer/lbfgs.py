from __future__ import annotations

import cupy as cp
import numpy as np
from mdpy.minimizer._base import Minimizer

_DOT_KERNEL = r"""
extern "C" __global__
void lbfgs_dot_kernel(
    const float* __restrict__ a_x,
    const float* __restrict__ a_y,
    const float* __restrict__ a_z,
    const float* __restrict__ b_x,
    const float* __restrict__ b_y,
    const float* __restrict__ b_z,
    float* __restrict__ output,
    int num_particles
) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    if (i < num_particles) {
        sum = a_x[i]*b_x[i] + a_y[i]*b_y[i] + a_z[i]*b_z[i];
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
void lbfgs_step_kernel(
    float* __restrict__ pos_x,
    float* __restrict__ pos_y,
    float* __restrict__ pos_z,
    const float* __restrict__ dir_x,
    const float* __restrict__ dir_y,
    const float* __restrict__ dir_z,
    float step_size,
    int num_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    pos_x[i] += step_size * dir_x[i];
    pos_y[i] += step_size * dir_y[i];
    pos_z[i] += step_size * dir_z[i];
}
"""


class LBFGSMinimizer(Minimizer):

    def __init__(self, step_size=1.0, history_size=5):
        super().__init__()
        self.step_size = float(step_size)
        self.history_size = history_size

        self._dot_kernel = None
        self._step_kernel = None
        self._dot_buffer = None

        self._num_updates = 0
        self._head = 0
        self._count = 0

        self._prev_positions_x = None
        self._prev_positions_y = None
        self._prev_positions_z = None
        self._prev_gradients_x = None
        self._prev_gradients_y = None
        self._prev_gradients_z = None

        self._s_x = None
        self._s_y = None
        self._s_z = None
        self._y_x = None
        self._y_y = None
        self._y_z = None

        self._direction_x = None
        self._direction_y = None
        self._direction_z = None

        self._grad_x = None
        self._grad_y = None
        self._grad_z = None

        self._rho = []
        self._alpha = []

    def _ensure_dot_kernel(self):
        if self._dot_kernel is not None:
            return
        self._dot_kernel = cp.RawKernel(_DOT_KERNEL, "lbfgs_dot_kernel")

    def _ensure_step_kernel(self):
        if self._step_kernel is not None:
            return
        self._step_kernel = cp.RawKernel(_STEP_KERNEL, "lbfgs_step_kernel")

    def _ensure_dot_buffer(self, num_blocks):
        if self._dot_buffer is not None and self._dot_buffer.size >= num_blocks:
            return
        self._dot_buffer = cp.zeros(num_blocks, dtype=cp.float32)

    def _ensure_buffers(self, num_particles):
        if self._direction_x is not None and self._direction_x.size >= num_particles:
            return

        self._prev_positions_x = cp.zeros(num_particles, dtype=cp.float32)
        self._prev_positions_y = cp.zeros(num_particles, dtype=cp.float32)
        self._prev_positions_z = cp.zeros(num_particles, dtype=cp.float32)
        self._prev_gradients_x = cp.zeros(num_particles, dtype=cp.float32)
        self._prev_gradients_y = cp.zeros(num_particles, dtype=cp.float32)
        self._prev_gradients_z = cp.zeros(num_particles, dtype=cp.float32)

        self._direction_x = cp.zeros(num_particles, dtype=cp.float32)
        self._direction_y = cp.zeros(num_particles, dtype=cp.float32)
        self._direction_z = cp.zeros(num_particles, dtype=cp.float32)

        self._grad_x = cp.zeros(num_particles, dtype=cp.float32)
        self._grad_y = cp.zeros(num_particles, dtype=cp.float32)
        self._grad_z = cp.zeros(num_particles, dtype=cp.float32)

        h = self.history_size
        self._s_x = cp.zeros((h, num_particles), dtype=cp.float32)
        self._s_y = cp.zeros((h, num_particles), dtype=cp.float32)
        self._s_z = cp.zeros((h, num_particles), dtype=cp.float32)
        self._y_x = cp.zeros((h, num_particles), dtype=cp.float32)
        self._y_y = cp.zeros((h, num_particles), dtype=cp.float32)
        self._y_z = cp.zeros((h, num_particles), dtype=cp.float32)

        self._rho = [0.0] * h
        self._head = 0
        self._count = 0
        self._num_updates = 0

    def _ring_index(self, i, m):
        return (self._head - m + i + self.history_size) % self.history_size

    def _compute_dot(self, a_x, a_y, a_z, b_x, b_y, b_z,
                     num_blocks, threads_per_block, num_particles):
        self._dot_buffer[:] = 0.0
        shared_mem = threads_per_block * 4
        self._dot_kernel(
            (num_blocks,), (threads_per_block,),
            (a_x, a_y, a_z, b_x, b_y, b_z,
             self._dot_buffer, np.int32(num_particles)),
            shared_mem=shared_mem,
        )
        return float(cp.sum(self._dot_buffer))

    def step(self, system):
        state = system.state
        num_particles = state.num_particles
        threads_per_block = 256
        num_blocks = (num_particles + threads_per_block - 1) // threads_per_block

        self._ensure_buffers(num_particles)
        self._ensure_dot_kernel()
        self._ensure_step_kernel()
        self._ensure_dot_buffer(num_blocks)

        self._grad_x[:] = state.d_forces_x / state.d_masses
        self._grad_y[:] = state.d_forces_y / state.d_masses
        self._grad_z[:] = state.d_forces_z / state.d_masses

        if self._num_updates == 0:
            self._direction_x[:] = self._grad_x
            self._direction_y[:] = self._grad_y
            self._direction_z[:] = self._grad_z
            self._prev_positions_x[:] = state.d_positions_x
            self._prev_positions_y[:] = state.d_positions_y
            self._prev_positions_z[:] = state.d_positions_z
            self._prev_gradients_x[:] = self._grad_x
            self._prev_gradients_y[:] = self._grad_y
            self._prev_gradients_z[:] = self._grad_z
            self._step_kernel(
                (num_blocks,), (threads_per_block,),
                (state.d_positions_x, state.d_positions_y, state.d_positions_z,
                 self._direction_x, self._direction_y, self._direction_z,
                 np.float32(self.step_size), np.int32(num_particles)),
            )
            self._num_updates = 1
            return

        slot = self._head
        cp.subtract(state.d_positions_x, self._prev_positions_x, out=self._s_x[slot])
        cp.subtract(state.d_positions_y, self._prev_positions_y, out=self._s_y[slot])
        cp.subtract(state.d_positions_z, self._prev_positions_z, out=self._s_z[slot])
        cp.subtract(self._grad_x, self._prev_gradients_x, out=self._y_x[slot])
        cp.subtract(self._grad_y, self._prev_gradients_y, out=self._y_y[slot])
        cp.subtract(self._grad_z, self._prev_gradients_z, out=self._y_z[slot])

        self._prev_positions_x[:] = state.d_positions_x
        self._prev_positions_y[:] = state.d_positions_y
        self._prev_positions_z[:] = state.d_positions_z
        self._prev_gradients_x[:] = self._grad_x
        self._prev_gradients_y[:] = self._grad_y
        self._prev_gradients_z[:] = self._grad_z

        rho = self._compute_dot(
            self._s_x[slot], self._s_y[slot], self._s_z[slot],
            self._y_x[slot], self._y_y[slot], self._y_z[slot],
            num_blocks, threads_per_block, num_particles,
        )
        if rho >= 0.0:
            self._direction_x[:] = self._grad_x
            self._direction_y[:] = self._grad_y
            self._direction_z[:] = self._grad_z
            self._step_kernel(
                (num_blocks,), (threads_per_block,),
                (state.d_positions_x, state.d_positions_y, state.d_positions_z,
                 self._direction_x, self._direction_y, self._direction_z,
                 np.float32(self.step_size), np.int32(num_particles)),
            )
            self._num_updates += 1
            return

        rho = 1.0 / rho
        self._rho[slot] = rho
        self._head = (self._head + 1) % self.history_size
        self._count = min(self._count + 1, self.history_size)

        self._direction_x[:] = -self._grad_x
        self._direction_y[:] = -self._grad_y
        self._direction_z[:] = -self._grad_z

        m = self._count
        self._alpha = [0.0] * m

        for i in range(m - 1, -1, -1):
            ring_idx = self._ring_index(i, m)
            rho_i = self._rho[ring_idx]
            s_xi = self._s_x[ring_idx]
            s_yi = self._s_y[ring_idx]
            s_zi = self._s_z[ring_idx]
            self._alpha[i] = rho_i * self._compute_dot(
                s_xi, s_yi, s_zi,
                self._direction_x, self._direction_y, self._direction_z,
                num_blocks, threads_per_block, num_particles,
            )
            self._direction_x[:] -= self._alpha[i] * self._y_x[ring_idx]
            self._direction_y[:] -= self._alpha[i] * self._y_y[ring_idx]
            self._direction_z[:] -= self._alpha[i] * self._y_z[ring_idx]

        if m > 0:
            newest = self._ring_index(m - 1, m)
            s_xi = self._s_x[newest]
            s_yi = self._s_y[newest]
            s_zi = self._s_z[newest]
            y_xi = self._y_x[newest]
            y_yi = self._y_y[newest]
            y_zi = self._y_z[newest]
            gamma_num = self._compute_dot(
                s_xi, s_yi, s_zi,
                y_xi, y_yi, y_zi,
                num_blocks, threads_per_block, num_particles,
            )
            gamma_den = self._compute_dot(
                y_xi, y_yi, y_zi,
                y_xi, y_yi, y_zi,
                num_blocks, threads_per_block, num_particles,
            )
            gamma = gamma_num / gamma_den if gamma_den > 0.0 else 1.0
        else:
            gamma = 1.0

        self._direction_x[:] *= gamma
        self._direction_y[:] *= gamma
        self._direction_z[:] *= gamma

        for i in range(m):
            ring_idx = self._ring_index(i, m)
            rho_i = self._rho[ring_idx]
            y_xi = self._y_x[ring_idx]
            y_yi = self._y_y[ring_idx]
            y_zi = self._y_z[ring_idx]
            beta = rho_i * self._compute_dot(
                y_xi, y_yi, y_zi,
                self._direction_x, self._direction_y, self._direction_z,
                num_blocks, threads_per_block, num_particles,
            )
            self._direction_x[:] += (self._alpha[i] - beta) * self._s_x[ring_idx]
            self._direction_y[:] += (self._alpha[i] - beta) * self._s_y[ring_idx]
            self._direction_z[:] += (self._alpha[i] - beta) * self._s_z[ring_idx]

        self._step_kernel(
            (num_blocks,), (threads_per_block,),
            (state.d_positions_x, state.d_positions_y, state.d_positions_z,
             self._direction_x, self._direction_y, self._direction_z,
             np.float32(self.step_size), np.int32(num_particles)),
        )
        self._num_updates += 1
