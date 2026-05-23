from __future__ import annotations

import math

import numpy as np
from numba import cuda

from mdpy.core.gpu_kernels import minimum_image


@cuda.jit
def verlet_init_kernel(positions, velocities, forces, masses,
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
def verlet_kernel(positions, prev_positions, forces, masses,
                  velocities, pbc_matrix, pbc_inv, dt, dt_sq, number_particles):
    index = cuda.grid(1)
    if index >= number_particles:
        return
    if masses[index] <= 0.0:
        return
    inv_mass = 1.0 / masses[index]
    acceleration_x = forces[index * 3] * inv_mass
    acceleration_y = forces[index * 3 + 1] * inv_mass
    acceleration_z = forces[index * 3 + 2] * inv_mass

    current_x = positions[index * 3]
    current_y = positions[index * 3 + 1]
    current_z = positions[index * 3 + 2]

    new_x = 2.0 * current_x - prev_positions[index * 3] + acceleration_x * dt_sq
    new_y = 2.0 * current_y - prev_positions[index * 3 + 1] + acceleration_y * dt_sq
    new_z = 2.0 * current_z - prev_positions[index * 3 + 2] + acceleration_z * dt_sq

    velocities[index * 3] = (new_x - prev_positions[index * 3]) / (2.0 * dt)
    velocities[index * 3 + 1] = (new_y - prev_positions[index * 3 + 1]) / (2.0 * dt)
    velocities[index * 3 + 2] = (new_z - prev_positions[index * 3 + 2]) / (2.0 * dt)

    new_x, new_y, new_z = minimum_image(new_x, new_y, new_z, pbc_matrix, pbc_inv)

    prev_positions[index * 3] = current_x
    prev_positions[index * 3 + 1] = current_y
    prev_positions[index * 3 + 2] = current_z
    positions[index * 3] = new_x
    positions[index * 3 + 1] = new_y
    positions[index * 3 + 2] = new_z


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
            verlet_init_kernel[grid, block](
                gpu_context.d_positions,
                gpu_context.d_velocities,
                gpu_context.d_forces,
                gpu_context.d_masses,
                gpu_context.d_prev_positions,
                np.float32(self.dt),
                np.float32(self.dt_sq),
                number,
            )
            self._initialized = True

        verlet_kernel[grid, block](
            gpu_context.d_positions,
            gpu_context.d_prev_positions,
            gpu_context.d_forces,
            gpu_context.d_masses,
            gpu_context.d_velocities,
            gpu_context.d_pbc_matrix,
            gpu_context.d_pbc_inv,
            np.float32(self.dt),
            np.float32(self.dt_sq),
            number,
        )
