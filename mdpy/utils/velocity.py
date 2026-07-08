import numpy as np
from mdpy import precision
from mdpy.unit import KB, default_energy_unit, kelvin

# Boltzmann constant in mdpy internal units (file-local).
BOLTZMANN = float(KB.convert_to(default_energy_unit / kelvin).value)


def generate_velocity_from_temperature(temperature, masses, seed=None):
    rng = np.random.default_rng(seed)
    num_particles = len(masses)
    velocities = rng.standard_normal((num_particles, 3)).astype(precision.FLOAT)
    inv_sqrt_mass = 1.0 / np.sqrt(masses.astype(np.float64))
    velocities *= (inv_sqrt_mass[:, np.newaxis]).astype(precision.FLOAT)
    kinetic_energy_per_dof = (
        0.5
        * np.sum(
            masses.astype(np.float64)[:, np.newaxis]
            * velocities.astype(np.float64) ** 2
        )
        / (3 * num_particles)
    )
    if kinetic_energy_per_dof > 0:
        current_temperature = 2.0 * kinetic_energy_per_dof / BOLTZMANN
        velocities *= np.float32(np.sqrt(temperature / current_temperature))
    com_velocity = np.sum(
        masses.astype(np.float64)[:, np.newaxis] * velocities.astype(np.float64), axis=0
    ) / np.sum(masses.astype(np.float64))
    velocities -= com_velocity.astype(precision.FLOAT)
    return velocities
