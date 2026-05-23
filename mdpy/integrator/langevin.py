from __future__ import annotations

import math

import numpy as np
from numba import cuda, uint64

from mdpy.core.gpu_kernels import minimum_image


_BOLTZMANN = 8.314462618e-7

_LCG_MULT = np.uint64(6364136223846793005)
_LCG_ADD = np.uint64(1442695040888963407)
_LCG_SEED_STRIDE = np.uint64(12345)
_LCG_SHIFT = np.uint64(33)
_LCG_MASK = np.uint64(0x7FFFFFFF)
_LCG_DIVISOR = 1073741824.0


@cuda.jit
def langevin_init_kernel(pos_x, pos_y, pos_z,
                         vel_x, vel_y, vel_z,
                         f_x, f_y, f_z,
                         masses,
                         prev_pos_x, prev_pos_y, prev_pos_z,
                         dt, dt_sq, number_particles):
    index = cuda.grid(1)
    if index >= number_particles:
        return
    if masses[index] <= 0.0:
        return
    inv_mass = 1.0 / masses[index]
    prev_pos_x[index] = pos_x[index] - vel_x[index] * dt + 0.5 * f_x[index] * inv_mass * dt_sq
    prev_pos_y[index] = pos_y[index] - vel_y[index] * dt + 0.5 * f_y[index] * inv_mass * dt_sq
    prev_pos_z[index] = pos_z[index] - vel_z[index] * dt + 0.5 * f_z[index] * inv_mass * dt_sq


@cuda.jit
def langevin_baoab_kernel(pos_x, pos_y, pos_z,
                          prev_pos_x, prev_pos_y, prev_pos_z,
                          f_x, f_y, f_z,
                          masses,
                          pbc_matrix, pbc_inv,
                          dt, dt_half, alpha, temperature, boltzmann,
                          seed, number_particles):
    index = cuda.grid(1)
    if index >= number_particles:
        return
    if masses[index] <= 0.0:
        return

    state = seed + uint64(index) * _LCG_SEED_STRIDE
    state = state * _LCG_MULT + _LCG_ADD
    rand1 = float((state >> _LCG_SHIFT) & _LCG_MASK) / _LCG_DIVISOR - 1.0
    state = state * _LCG_MULT + _LCG_ADD
    rand2 = float((state >> _LCG_SHIFT) & _LCG_MASK) / _LCG_DIVISOR - 1.0
    state = state * _LCG_MULT + _LCG_ADD
    rand3 = float((state >> _LCG_SHIFT) & _LCG_MASK) / _LCG_DIVISOR - 1.0
    state = state * _LCG_MULT + _LCG_ADD
    rand4 = float((state >> _LCG_SHIFT) & _LCG_MASK) / _LCG_DIVISOR - 1.0
    state = state * _LCG_MULT + _LCG_ADD
    rand5 = float((state >> _LCG_SHIFT) & _LCG_MASK) / _LCG_DIVISOR - 1.0
    state = state * _LCG_MULT + _LCG_ADD
    rand6 = float((state >> _LCG_SHIFT) & _LCG_MASK) / _LCG_DIVISOR - 1.0

    inv_mass = 1.0 / masses[index]
    sigma = math.sqrt(boltzmann * temperature * inv_mass * (1.0 - alpha * alpha))

    velocity_x = (pos_x[index] - prev_pos_x[index]) / dt
    velocity_y = (pos_y[index] - prev_pos_y[index]) / dt
    velocity_z = (pos_z[index] - prev_pos_z[index]) / dt

    velocity_x += 0.5 * dt * f_x[index] * inv_mass
    velocity_y += 0.5 * dt * f_y[index] * inv_mass
    velocity_z += 0.5 * dt * f_z[index] * inv_mass

    position_x = pos_x[index] + 0.5 * dt * velocity_x
    position_y = pos_y[index] + 0.5 * dt * velocity_y
    position_z = pos_z[index] + 0.5 * dt * velocity_z

    velocity_x = alpha * velocity_x + sigma * rand1
    velocity_y = alpha * velocity_y + sigma * rand2
    velocity_z = alpha * velocity_z + sigma * rand3

    position_x += 0.5 * dt * velocity_x
    position_y += 0.5 * dt * velocity_y
    position_z += 0.5 * dt * velocity_z

    position_x, position_y, position_z = minimum_image(
        position_x, position_y, position_z, pbc_matrix, pbc_inv
    )

    prev_pos_x[index] = pos_x[index]
    prev_pos_y[index] = pos_y[index]
    prev_pos_z[index] = pos_z[index]
    pos_x[index] = position_x
    pos_y[index] = position_y
    pos_z[index] = position_z


class LangevinBAOABIntegrator:

    def __init__(self, time_step, temperature, friction):
        self.dt = float(time_step)
        self.dt_half = self.dt * 0.5
        self.temperature = float(temperature)
        self.friction = float(friction)
        self.alpha = np.float32(np.exp(-friction * time_step))
        self._initialized = False
        self._step_counter = 0

    def step(self, gpu_context):
        number = gpu_context.number_particles
        block = 256
        grid = (number + block - 1) // block

        if not self._initialized:
            langevin_init_kernel[grid, block](
                gpu_context.d_positions_x, gpu_context.d_positions_y, gpu_context.d_positions_z,
                gpu_context.d_velocities_x, gpu_context.d_velocities_y, gpu_context.d_velocities_z,
                gpu_context.d_forces_x, gpu_context.d_forces_y, gpu_context.d_forces_z,
                gpu_context.d_masses,
                gpu_context.d_prev_positions_x, gpu_context.d_prev_positions_y, gpu_context.d_prev_positions_z,
                np.float32(self.dt),
                np.float32(self.dt * self.dt),
                number,
            )
            self._initialized = True

        self._step_counter += 1

        langevin_baoab_kernel[grid, block](
            gpu_context.d_positions_x, gpu_context.d_positions_y, gpu_context.d_positions_z,
            gpu_context.d_prev_positions_x, gpu_context.d_prev_positions_y, gpu_context.d_prev_positions_z,
            gpu_context.d_forces_x, gpu_context.d_forces_y, gpu_context.d_forces_z,
            gpu_context.d_masses,
            gpu_context.d_pbc_matrix,
            gpu_context.d_pbc_inv,
            np.float32(self.dt),
            np.float32(self.dt_half),
            np.float32(self.alpha),
            np.float32(self.temperature),
            np.float32(_BOLTZMANN),
            np.uint64(self._step_counter) * np.uint64(1000003),
            number,
        )
