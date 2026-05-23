import numpy as np
from mdpy import env


class ParticleTable:

    __slots__ = [
        'positions', 'velocities', 'forces',
        'masses', 'charges', 'particle_types', 'molecule_ids',
        'num_particles',
    ]

    def __init__(self, num_particles: int):
        number = num_particles
        self.positions = np.zeros((number, 3), dtype=env.NUMPY_FLOAT)
        self.velocities = np.zeros((number, 3), dtype=env.NUMPY_FLOAT)
        self.forces = np.zeros((number, 3), dtype=env.NUMPY_FLOAT)
        self.masses = np.zeros(number, dtype=env.NUMPY_FLOAT)
        self.charges = np.zeros(number, dtype=env.NUMPY_FLOAT)
        self.particle_types = np.zeros(number, dtype=env.NUMPY_INT)
        self.molecule_ids = np.zeros(number, dtype=env.NUMPY_INT)
        self.num_particles = number

    def zero_forces(self):
        self.forces[:] = 0

    def __repr__(self) -> str:
        return '<mdpy.core.ParticleTable: %d particles at %x>' % (
            self.num_particles, id(self)
        )
