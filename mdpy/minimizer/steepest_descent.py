from __future__ import annotations

import cupy as cp
import numpy as np
from mdpy.minimizer._base import Minimizer

_STEP_KERNEL = r"""
extern "C" __global__
void steepest_descent_step_kernel(
    float* __restrict__ pos_x,
    float* __restrict__ pos_y,
    float* __restrict__ pos_z,
    const float* __restrict__ fx,
    const float* __restrict__ fy,
    const float* __restrict__ fz,
    const float* __restrict__ masses,
    float step_size,
    int num_particles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;
    float mass = masses[i];
    if (mass <= 0.0f) return;
    float inv_mass = 1.0f / mass;
    pos_x[i] += step_size * fx[i] * inv_mass;
    pos_y[i] += step_size * fy[i] * inv_mass;
    pos_z[i] += step_size * fz[i] * inv_mass;
}
"""


class SteepestDescentMinimizer(Minimizer):

    def __init__(self, step_size=0.1):
        super().__init__()
        self.step_size = float(step_size)
        self._step_kernel = None

    def _ensure_step_kernel(self):
        if self._step_kernel is not None:
            return
        self._step_kernel = cp.RawKernel(
            _STEP_KERNEL, "steepest_descent_step_kernel"
        )

    def step(self, system):
        state = system.state
        num_particles = state.num_particles
        threads_per_block = 256
        num_blocks = (num_particles + threads_per_block - 1) // threads_per_block
        self._ensure_step_kernel()
        self._step_kernel(
            (num_blocks,), (threads_per_block,),
            (state.d_positions_x, state.d_positions_y, state.d_positions_z,
             state.d_forces_x, state.d_forces_y, state.d_forces_z,
             state.d_particle_masses, np.float32(self.step_size), np.int32(num_particles)),
        )
