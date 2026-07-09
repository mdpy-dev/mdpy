from __future__ import annotations

import numpy as np
import cupy as cp

from mdpy.unit import KB, default_energy_unit, kelvin

BOLTZMANN = float(KB.convert_to(default_energy_unit / kelvin).value)


class LangevinPistonBarostat:
    """Langevin piston barostat for isotropic NPT simulations.

    Treats the box volume as an extended dynamical variable coupled to a
    Langevin thermostat. The piston equation of motion is:

        W * d^2(log V)/dt^2 = (P_current - P_target) * V - gamma * W * d(log V)/dt + sqrt(2*gamma*W*kT) * R(t)

    where W is the piston mass, gamma is the piston friction,
    and R(t) is Gaussian white noise.

    This produces a correct NPT ensemble.

    References:
        Feller et al., J. Chem. Phys. 103, 4613 (1995)
    """

    def __init__(
        self,
        target_pressure,
        temperature,
        piston_mass=None,
        piston_friction=1.0,
    ):
        self.target_pressure = float(target_pressure)
        self.temperature = float(temperature)
        self.piston_friction = float(piston_friction)
        self._piston_mass = piston_mass  # None -> auto-compute on first apply
        self._piston_velocity = 0.0  # d(log V)/dt
        self._step_counter = 0

    def apply(self, system, temperature, time_step):
        """Integrate piston equation of motion for one step."""
        state = system.state
        num_particles = state.num_particles
        dt = float(time_step)
        kB = BOLTZMANN

        if self._piston_mass is None:
            volume = state.box_x * state.box_y * state.box_z
            tau_P = 100.0
            self._piston_mass = (
                3.0
                * num_particles
                * kB
                * self.temperature
                * tau_P
                * tau_P
                / (volume ** (2.0 / 3.0))
            )

        W = self._piston_mass
        gamma = self.piston_friction
        volume = state.box_x * state.box_y * state.box_z

        kinetic_energy = 1.5 * num_particles * kB * float(temperature)
        virial_trace = float(cp.asnumpy(state.d_virial[0]))
        current_pressure = (2.0 * kinetic_energy + virial_trace) / (3.0 * volume)

        piston_force = (current_pressure - self.target_pressure) * volume / W
        friction_force = -gamma * self._piston_velocity

        noise_std = np.sqrt(2.0 * gamma * kB * self.temperature / W * dt)
        noise = np.random.normal(0.0, noise_std)

        dv = (piston_force + friction_force) * dt + noise
        self._piston_velocity += dv

        v_dt = self._piston_velocity * dt
        mu = np.float32(np.exp(v_dt / 3.0))

        state.scale_box(mu)
        state.scale_positions_and_velocities(mu)

        self._step_counter += 1
