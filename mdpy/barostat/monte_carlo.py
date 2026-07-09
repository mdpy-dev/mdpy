from __future__ import annotations

import numpy as np
import cupy as cp

from mdpy.unit import KB, default_energy_unit, kelvin

BOLTZMANN = float(KB.convert_to(default_energy_unit / kelvin).value)


class MonteCarloBarostat:
    """Monte Carlo barostat for isotropic pressure control.

    Periodically proposes a box volume change, scales positions/velocities,
    recomputes the energy, and accepts/rejects based on the Metropolis criterion:

        P_accept = min(1, exp(-(ΔE + P*ΔV - N*kT*ln(V_new/V_old)) / kT))

    This produces a correct NPT ensemble. The barostat is applied every
    `frequency` steps.

    References:
        Allen & Tildesley, Computer Simulation of Liquids (1987), section 7.5
    """

    def __init__(
        self,
        target_pressure,
        temperature,
        frequency=25,
        volume_max_change=0.001,
    ):
        self.target_pressure = float(target_pressure)
        self.frequency = frequency
        self.volume_max_change = float(volume_max_change)
        self._step_counter = 0
        self._num_attempts = 0
        self._num_accepted = 0

    @property
    def acceptance_rate(self):
        if self._num_attempts == 0:
            return 0.0
        return self._num_accepted / self._num_attempts

    def apply(self, system, temperature, time_step):
        """Attempt a Monte Carlo volume change (only on frequency-aligned steps)."""
        self._step_counter += 1
        if self._step_counter % self.frequency != 0:
            return

        state = system.state
        num_particles = state.num_particles
        kB = BOLTZMANN
        beta = 1.0 / (kB * float(temperature))

        old_volume = state.box_x * state.box_y * state.box_z

        kinetic_energy = 1.5 * num_particles * kB * float(temperature)

        old_energy = float(cp.asnumpy(state.d_energy[0]))
        old_virial = float(cp.asnumpy(state.d_virial[0]))

        old_box_x = state.box_x
        old_box_y = state.box_y
        old_box_z = state.box_z
        old_pos_x = state.d_positions_x.copy()
        old_pos_y = state.d_positions_y.copy()
        old_pos_z = state.d_positions_z.copy()
        old_vel_x = state.d_velocities_x.copy()
        old_vel_y = state.d_velocities_y.copy()
        old_vel_z = state.d_velocities_z.copy()
        old_prev_x = state.d_prev_positions_x.copy()
        old_prev_y = state.d_prev_positions_y.copy()
        old_prev_z = state.d_prev_positions_z.copy()

        delta_v = self.volume_max_change * old_volume * (2.0 * np.random.random() - 1.0)
        new_volume = old_volume + delta_v

        if new_volume <= 0:
            return

        scale_factor = np.float32((new_volume / old_volume) ** (1.0 / 3.0))

        state.scale_box(scale_factor)
        state.scale_positions_and_velocities(scale_factor)

        system.compute_forces(compute_energy=True, compute_virial=True)
        new_energy = float(cp.asnumpy(state.d_energy[0]))

        delta_E = new_energy - old_energy
        delta_V = new_volume - old_volume
        log_V_ratio = np.log(new_volume / old_volume)
        delta_H = (
            delta_E
            + self.target_pressure * delta_V
            - num_particles * kB * float(temperature) * log_V_ratio
        )

        accept = delta_H <= 0.0 or np.random.random() < np.exp(-beta * delta_H)

        self._num_attempts += 1

        if accept:
            self._num_accepted += 1
            del old_pos_x, old_pos_y, old_pos_z
            del old_vel_x, old_vel_y, old_vel_z
            del old_prev_x, old_prev_y, old_prev_z
        else:
            state.d_pbc_matrix[0] = np.float32(old_box_x)
            state.d_pbc_matrix[4] = np.float32(old_box_y)
            state.d_pbc_matrix[8] = np.float32(old_box_z)
            state.d_pbc_inv[0] = np.float32(1.0 / old_box_x)
            state.d_pbc_inv[4] = np.float32(1.0 / old_box_y)
            state.d_pbc_inv[8] = np.float32(1.0 / old_box_z)
            state._box_x = old_box_x
            state._box_y = old_box_y
            state._box_z = old_box_z
            state._inv_box_x = 1.0 / old_box_x
            state._inv_box_y = 1.0 / old_box_y
            state._inv_box_z = 1.0 / old_box_z

            state.d_positions_x[:] = old_pos_x
            state.d_positions_y[:] = old_pos_y
            state.d_positions_z[:] = old_pos_z
            state.d_velocities_x[:] = old_vel_x
            state.d_velocities_y[:] = old_vel_y
            state.d_velocities_z[:] = old_vel_z
            state.d_prev_positions_x[:] = old_prev_x
            state.d_prev_positions_y[:] = old_prev_y
            state.d_prev_positions_z[:] = old_prev_z

            del old_pos_x, old_pos_y, old_pos_z
            del old_vel_x, old_vel_y, old_vel_z
            del old_prev_x, old_prev_y, old_prev_z

            system.compute_forces(compute_energy=False, compute_virial=True)
