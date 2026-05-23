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
def langevin_init_kernel(positions, velocities, forces, masses,
                         prev_positions, dt, dt_sq, number_particles):
    index = cuda.grid(1)
    if index >= number_particles:
        return
    if masses[index] <= 0.0:
        return
    inv_mass = 1.0 / masses[index]
    prev_positions[index * 3] = positions[index * 3] - velocities[index * 3] * dt + 0.5 * forces[index * 3] * inv_mass * dt_sq
    prev_positions[index * 3 + 1] = positions[index * 3 + 1] - velocities[index * 3 + 1] * dt + 0.5 * forces[index * 3 + 1] * inv_mass * dt_sq
    prev_positions[index * 3 + 2] = positions[index * 3 + 2] - velocities[index * 3 + 2] * dt + 0.5 * forces[index * 3 + 2] * inv_mass * dt_sq


@cuda.jit
def langevin_baoab_kernel(positions, prev_positions, forces, masses,
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

    velocity_x = (positions[index * 3] - prev_positions[index * 3]) / dt
    velocity_y = (positions[index * 3 + 1] - prev_positions[index * 3 + 1]) / dt
    velocity_z = (positions[index * 3 + 2] - prev_positions[index * 3 + 2]) / dt

    velocity_x += 0.5 * dt * forces[index * 3] * inv_mass
    velocity_y += 0.5 * dt * forces[index * 3 + 1] * inv_mass
    velocity_z += 0.5 * dt * forces[index * 3 + 2] * inv_mass

    position_x = positions[index * 3] + 0.5 * dt * velocity_x
    position_y = positions[index * 3 + 1] + 0.5 * dt * velocity_y
    position_z = positions[index * 3 + 2] + 0.5 * dt * velocity_z

    velocity_x = alpha * velocity_x + sigma * rand1
    velocity_y = alpha * velocity_y + sigma * rand2
    velocity_z = alpha * velocity_z + sigma * rand3

    position_x += 0.5 * dt * velocity_x
    position_y += 0.5 * dt * velocity_y
    position_z += 0.5 * dt * velocity_z

    position_x, position_y, position_z = minimum_image(
        position_x, position_y, position_z, pbc_matrix, pbc_inv
    )

    prev_positions[index * 3] = positions[index * 3]
    prev_positions[index * 3 + 1] = positions[index * 3 + 1]
    prev_positions[index * 3 + 2] = positions[index * 3 + 2]
    positions[index * 3] = position_x
    positions[index * 3 + 1] = position_y
    positions[index * 3 + 2] = position_z


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
                gpu_context.d_positions,
                gpu_context.d_velocities,
                gpu_context.d_forces,
                gpu_context.d_masses,
                gpu_context.d_prev_positions,
                np.float32(self.dt),
                np.float32(self.dt * self.dt),
                number,
            )
            self._initialized = True

        self._step_counter += 1

        langevin_baoab_kernel[grid, block](
            gpu_context.d_positions,
            gpu_context.d_prev_positions,
            gpu_context.d_forces,
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
