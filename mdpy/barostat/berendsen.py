from __future__ import annotations

import numpy as np
import cupy as cp

from mdpy.unit import KB, default_energy_unit, kelvin


BOLTZMANN = float(KB.convert_to(default_energy_unit / kelvin).value)


class BerendsenBarostat:
    """Berendsen weak-coupling barostat for isotropic pressure control.

    Scales the box and all particle positions/velocities each step toward the
    target pressure. This is a deterministic method that does NOT produce a
    correct NPT ensemble — use MonteCarloBarostat for production.

    The scaling factor per step is:
        mu = [1 - (dt/tau_P) * (P_target - P_current)]^(1/3)

    where tau_P is the pressure coupling time constant.

    References:
        Berendsen et al., J. Chem. Phys. 81, 3684 (1984)
    """

    def __init__(
        self,
        target_pressure,
        pressure_coupling_time,
    ):
        """Initialize Berendsen barostat.

        Parameters
        ----------
        target_pressure : float
            Target pressure in internal units (dalton/(angstrom * fs^2)).
            Use bar.convert_to(default_pressure_unit).value to convert from bar.
        pressure_coupling_time : float
            Pressure coupling time constant in fs. Typical: 1000-5000 fs.
        """
        self.target_pressure = float(target_pressure)
        self.pressure_coupling_time = float(pressure_coupling_time)

    def apply(self, system, temperature, time_step):
        """Apply one step of Berendsen barostat scaling.

        Reads virial from state.d_virial[0] (set by compute_forces(compute_virial=True)).
        Computes instantaneous pressure, scales box and positions/velocities.

        Parameters
        ----------
        system : System
            The system to apply barostat to.
        temperature : float
            Current temperature in kelvin (for kinetic energy contribution).
        time_step : float
            Integration time step in fs.
        """
        state = system.state

        num_particles = state.num_particles
        volume = state.box_x * state.box_y * state.box_z

        kinetic_energy = 1.5 * num_particles * BOLTZMANN * float(temperature)

        virial_trace = float(cp.asnumpy(state.d_virial[0]))

        current_pressure = (2.0 * kinetic_energy + virial_trace) / (3.0 * volume)

        pressure_diff = self.target_pressure - current_pressure
        mu_cubed = 1.0 - (time_step / self.pressure_coupling_time) * pressure_diff

        mu_cubed = max(mu_cubed, 0.001)
        mu = np.float32(mu_cubed ** (1.0 / 3.0))

        state.scale_box(mu)
        state.scale_positions_and_velocities(mu)
