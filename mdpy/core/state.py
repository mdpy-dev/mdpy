#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : state.py
created time : 2021/10/17
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import numpy as np
import numba.cuda as cuda
from mdpy.core.cell_list import CellList
from mdpy.core.topology import Topology
from mdpy import SPATIAL_DIM, env
from mdpy.unit import *
from mdpy.error import *
from mdpy.utils import *

class State:
    def __init__(self, topology: Topology, pbc_matrix: np.ndarray) -> None:
        self._particles = topology.particles
        self._masses = topology.masses
        self._num_particles = len(self._particles)
        self._matrix_shape = [self._num_particles, SPATIAL_DIM]
        self._positions = np.zeros(self._matrix_shape, dtype=env.NUMPY_FLOAT)
        self._velocities = np.zeros(self._matrix_shape, dtype=env.NUMPY_FLOAT)
        self.set_pbc_matrix(pbc_matrix)
        self._cell_list = CellList(self._pbc_matrix.copy())
        # Device array
        self._device_postitions = None

    def __repr__(self) -> str:
        return '<mdpy.core.State object with %d particles at %x>' %(
            self._num_particles, id(self)
        )

    __str__ = __repr__

    def _check_matrix_shape(self, matrix: np.ndarray):
        if not isinstance(matrix, np.ndarray):
            raise TypeError('Matrix should be numpy.ndarray, instead of %s' %type(matrix))
        row, col = matrix.shape
        if row != self._matrix_shape[0] or col != self._matrix_shape[1]:
            raise ArrayDimError(
                'The dimension of array should be [%d, %d], while array [%d, %d] is provided'
                %(self._matrix_shape[0], self._matrix_shape[1], row, col)
            )

    def set_pbc_matrix(self, pbc_matrix):
        pbc_matrix = check_quantity_value(pbc_matrix, default_length_unit)
        pbc_matrix = check_pbc_matrix(pbc_matrix)
        # The origin define of pbc_matrix is the stack of 3 column vector
        # While in MDPy the position is in shape of n x 3
        # So the scaled position will be Position * PBC instead of PBC * Position as usual
        self._pbc_matrix = np.ascontiguousarray(pbc_matrix, dtype=env.NUMPY_FLOAT)
        self._pbc_inv = np.ascontiguousarray(np.linalg.inv(self._pbc_matrix), dtype=env.NUMPY_FLOAT)
        self._device_pbc_matrix = cuda.to_device(self._pbc_matrix)
        self._device_pbc_inv = cuda.to_device(self._pbc_inv)

    def set_positions(self, positions: np.ndarray):
        self._check_matrix_shape(positions)
        self._positions = wrap_positions(
            positions.astype(env.NUMPY_FLOAT), self._pbc_matrix, self._pbc_inv
        )
        self._cell_list.update(self._positions)
        self._device_postitions = cuda.to_device(self._positions)

    def set_velocities(self, velocities: np.ndarray):
        self._check_matrix_shape(velocities)
        self._velocities = velocities.astype(env.NUMPY_FLOAT)

    def set_velocities_to_temperature(self, temperature):
        temperature = check_quantity(temperature, default_temperature_unit)
        factor = Quantity(3) * KB * temperature / default_mass_unit
        factor = factor.convert_to(default_velocity_unit**2).value
        self.set_velocities(self.generator(self._masses, factor))

    @staticmethod
    def generator(masses, factor):
        num_particles = masses.shape[0]
        velocities = np.random.rand(num_particles, 3).astype(masses.dtype)
        for particle in range(num_particles):
            width = np.sqrt(factor/masses[particle])
            velocities[particle, :] = velocities[particle, :] * 2 * width - width
        return velocities

    @property
    def positions(self):
        return self._positions

    @property
    def device_positions(self):
        return self._device_postitions

    @property
    def velocities(self):
        return self._velocities

    @property
    def matrix_shape(self):
        return self._matrix_shape

    @property
    def pbc_matrix(self):
        return self._pbc_matrix

    @property
    def device_pbc_matrix(self):
        return self._device_pbc_matrix

    @property
    def pbc_inv(self):
        return self._pbc_inv

    @property
    def device_pbc_inv(self):
        return self._device_pbc_inv

    @property
    def pbc_info(self):
        return self._pbc_matrix, self._pbc_inv

    @property
    def cell_list(self):
        return self._cell_list