from __future__ import annotations

import numpy as np
import cupy as cp

from mdpy.unit import KB, default_energy_unit, kelvin

# Boltzmann constant in mdpy internal units (file-local).
BOLTZMANN = float(KB.convert_to(default_energy_unit / kelvin).value)

_LANGEVIN_INIT_KERNEL = r"""
extern "C" __global__
void langevin_init_kernel(
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

_LANGEVIN_BAOAB_KERNEL = r"""
extern "C" __global__
void langevin_baoab_kernel(
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
    float time_step, float half_time_step, float alpha, float temperature, float boltzmann,
    unsigned long long seed, int num_particles
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_particles) return;
    float mass = masses[index];
    if (mass <= 0.0f) return;

    unsigned long long state = seed + (unsigned long long)index * 12345ULL;
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    float rand1 = (float)((state >> 33) & 0x7FFFFFFFULL) / 1073741824.0f - 1.0f;
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    float rand2 = (float)((state >> 33) & 0x7FFFFFFFULL) / 1073741824.0f - 1.0f;
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    float rand3 = (float)((state >> 33) & 0x7FFFFFFFULL) / 1073741824.0f - 1.0f;
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    float rand4 = (float)((state >> 33) & 0x7FFFFFFFULL) / 1073741824.0f - 1.0f;
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    float rand5 = (float)((state >> 33) & 0x7FFFFFFFULL) / 1073741824.0f - 1.0f;
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    float rand6 = (float)((state >> 33) & 0x7FFFFFFFULL) / 1073741824.0f - 1.0f;

    float inv_mass = 1.0f / mass;
    float sigma = sqrtf(boltzmann * temperature * inv_mass * (1.0f - alpha * alpha));

    float r1 = sqrtf(-2.0f * logf(max(1.0f - fabsf(rand1), 1e-30f)));
    float theta1 = 2.0f * 3.14159265358979323846f * rand2;
    float g1 = r1 * cosf(theta1);
    float r2 = sqrtf(-2.0f * logf(max(1.0f - fabsf(rand3), 1e-30f)));
    float theta2 = 2.0f * 3.14159265358979323846f * rand4;
    float g3 = r2 * cosf(theta2);
    float r3 = sqrtf(-2.0f * logf(max(1.0f - fabsf(rand5), 1e-30f)));
    float theta3 = 2.0f * 3.14159265358979323846f * rand6;
    float g5 = r3 * cosf(theta3);

    float dx = pos_x[index] - prev_pos_x[index];
    float dy = pos_y[index] - prev_pos_y[index];
    float dz = pos_z[index] - prev_pos_z[index];
    float vx = dx / time_step;
    float vy = dy / time_step;
    float vz = dz / time_step;

    // B: half-kick (force half-step). This is the leading B of BAOAB; the
    // trailing B of the previous step is folded in here because forces are
    // recomputed between steps.
    vx += 0.5f * time_step * f_x[index] * inv_mass;
    vy += 0.5f * time_step * f_y[index] * inv_mass;
    vz += 0.5f * time_step * f_z[index] * inv_mass;

    // A: half-drift (position half-step)
    float px = pos_x[index] + 0.5f * time_step * vx;
    float py = pos_y[index] + 0.5f * time_step * vy;
    float pz = pos_z[index] + 0.5f * time_step * vz;

    // O: Ornstein-Uhlenbeck (stochastic velocity update)
    vx = alpha * vx + sigma * g1;
    vy = alpha * vy + sigma * g3;
    vz = alpha * vz + sigma * g5;

    // A: half-drift (position half-step)
    px += 0.5f * time_step * vx;
    py += 0.5f * time_step * vy;
    pz += 0.5f * time_step * vz;

    prev_pos_x[index] = pos_x[index];
    prev_pos_y[index] = pos_y[index];
    prev_pos_z[index] = pos_z[index];
    pos_x[index] = px;
    pos_y[index] = py;
    pos_z[index] = pz;
}
"""

_kernels = {
    'init': cp.RawKernel(_LANGEVIN_INIT_KERNEL, 'langevin_init_kernel'),
    'step': cp.RawKernel(_LANGEVIN_BAOAB_KERNEL, 'langevin_baoab_kernel'),
}


class LangevinBAOABIntegrator:
    """Langevin dynamics integrator using the BAOAB splitting.

    BAOAB = B-A-O-A-B, a symmetric split-step scheme (Leimkuhler & Matthews 2013):
        B: half-kick (deterministic force, dt/2)
        A: half-drift (position update, dt/2)
        O: stochastic Ornstein-Uhlenbeck velocity update (full dt)
        A: half-drift (position update, dt/2)
        B: half-kick (deterministic force, dt/2)

    State is stored in position-Verlet (leapfrog) form using (pos, prev_pos); the
    velocity is reconstructed as v = (pos - prev_pos)/dt at the start of each step.
    Because forces are recomputed between steps, the trailing B of step N merges
    with the leading B of step N+1, so each kernel call performs B-A-O-A.

    The O-step uses the damping factor `alpha = exp(-friction * time_step)` and
    Gaussian noise scaled by
    `sigma = sqrt(boltzmann * temperature * (1 - alpha^2) / mass)`.
    """

    def __init__(self, time_step, temperature, friction):
        self.time_step = float(time_step)
        self.half_time_step = self.time_step * 0.5
        self.temperature = float(temperature)
        self.friction = float(friction)
        self.alpha = np.float32(np.exp(-friction * time_step))  # Langevin damping factor (velocity retention per step)
        self._initialized = False
        self._step_counter = 0

    def step(self, system):
        state = system.state
        number = state.num_particles
        threads_per_block = 256
        grid = (number + threads_per_block - 1) // threads_per_block

        if not self._initialized:
            _kernels['init']((grid,), (threads_per_block,), (
                state.d_positions_x, state.d_positions_y, state.d_positions_z,
                state.d_velocities_x, state.d_velocities_y, state.d_velocities_z,
                state.d_forces_x, state.d_forces_y, state.d_forces_z,
                state.d_particle_masses,
                state.d_prev_positions_x, state.d_prev_positions_y, state.d_prev_positions_z,
                np.float32(self.time_step),
                np.float32(self.time_step * self.time_step),
                np.int32(number),
            ))
            self._initialized = True

        self._step_counter += 1

        _kernels['step']((grid,), (threads_per_block,), (
            state.d_positions_x, state.d_positions_y, state.d_positions_z,
            state.d_prev_positions_x, state.d_prev_positions_y, state.d_prev_positions_z,
            state.d_forces_x, state.d_forces_y, state.d_forces_z,
            state.d_particle_masses,
            np.float32(self.time_step),
            np.float32(self.half_time_step),
            np.float32(self.alpha),
            np.float32(self.temperature),
            np.float32(BOLTZMANN),
            np.uint64(self._step_counter) * np.uint64(1000003),
            np.int32(number),
        ))
