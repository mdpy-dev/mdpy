#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : state.py
created time : 2021/10/17
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np
from .. import SPATIAL_DIM
from ..unit import *
from ..error import *
from ..math import *

class State:
    def __init__(self, particles) -> None:
        self._masses = [
            Quantity(particle.mass, default_mass_unit) for particle in particles
        ]
        self._num_particles = len(particles)
        self._matrix_shape = [self._num_particles, SPATIAL_DIM]
        self._positions = np.zeros(self._matrix_shape)
        self._velocities = np.zeros(self._matrix_shape)

    def __repr__(self) -> str:
        return '<mdpy.State object with %d particles at %x>' %(
            self._num_particles, id(self)
        )

    __str__ = __repr__

    def _check_matrix_shape(self, matrix: np.ndarray):
        row, col = matrix.shape
        if row != self._matrix_shape[0]:
            raise ParticleConflictError(
                'The dimension of position matrix [%d, %d] do not match the number of particles (%d) contained in topology'
                %(row, col, self._num_particles)
            )
        if col != self._matrix_shape[1]:
            raise SpatialDimError(
                'The column dimension of matrix should be %d, instead of %d' 
                %(self._matrix_shape[1], col)
            ) 

    def set_positions(self, positions: np.ndarray):
        self._check_matrix_shape(positions)
        self._positions = positions
    
    def set_velocities(self, velocities: np.ndarray):
        self._check_matrix_shape(velocities)
        self._velocities = velocities

    def set_velocities_to_temperature(self, temperature):
        temperature = check_quantity(temperature, default_temperature_unit)
        velocities = []
        factor = Quantity(3) * KB * temperature
        for i in range(self._num_particles):
            width = (factor/self._masses[i]).sqrt().convert_to(default_velocity_unit).value
            velocities.append(np.random.rand(3) * 2 * width - width)
        self.set_velocities(np.vstack(velocities))

    @property
    def positions(self):
        return self._positions

    @property
    def velocities(self):
        return self._velocities

    @property
    def matrix_shape(self):
        return self._matrix_shape