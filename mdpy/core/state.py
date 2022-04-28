#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : state.py
created time : 2021/10/17
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import numpy as np
import cupy as cp
from mdpy.core.neighbor_list import NeighborList
from mdpy.core.topology import Topology
from mdpy import SPATIAL_DIM
from mdpy.environment import *
from mdpy.unit import *
from mdpy.error import *
from mdpy.utils import *

class State:
    def __init__(self, topology: Topology, pbc_matrix: np.ndarray) -> None:
        self._topology = topology
        self._num_particles = len(self._topology.particles)
        self._matrix_shape = [self._num_particles, SPATIAL_DIM]
        self._positions = cp.zeros(self._matrix_shape, CUPY_FLOAT)
        self._velocities = cp.zeros(self._matrix_shape, CUPY_FLOAT)
        self.set_pbc_matrix(pbc_matrix)
        self._neighbor_list = NeighborList(self._pbc_matrix.copy())
        # Device array
        self._device_postitions = None

    def __repr__(self) -> str:
        return '<mdpy.core.State object with %d particles at %x>' %(
            self._num_particles, id(self)
        )

    __str__ = __repr__

    def _check_matrix_shape(self, matrix: np.ndarray):
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
        self._pbc_matrix = np.ascontiguousarray(pbc_matrix, dtype=NUMPY_FLOAT)
        self._pbc_diag = np.ascontiguousarray(np.diagonal(self._pbc_matrix), dtype=NUMPY_FLOAT)
        self._device_pbc_matrix = cp.array(self._pbc_matrix, CUPY_FLOAT)
        self._device_pbc_diag = cp.array(self._pbc_diag, CUPY_FLOAT)

    def set_positions(self, positions: cp.ndarray, is_update_neighbor_list=True):
        self._check_matrix_shape(positions)
        if isinstance(positions, np.ndarray):
            positions = cp.array(positions, CUPY_FLOAT)
        self._positions = positions
        move_vec = (positions < - self._device_pbc_diag) * self._device_pbc_diag
        move_vec -= (positions > self._device_pbc_diag) * self._device_pbc_diag
        self._positions += move_vec
        self._neighbor_list.update(self._positions, is_update_neighbor_list)

    def set_velocities(self, velocities: cp.ndarray):
        if isinstance(velocities, np.ndarray):
            velocities = cp.array(velocities, CUPY_FLOAT)
        self._check_matrix_shape(velocities)
        self._velocities = velocities.astype(CUPY_FLOAT)

    def set_velocities_to_temperature(self, temperature):
        temperature = check_quantity(temperature, default_temperature_unit)
        factor = Quantity(3) * KB * temperature / default_mass_unit
        factor = factor.convert_to(default_velocity_unit**2).value
        # Generate force
        velocities = cp.random.rand(self._num_particles, 3).astype(CUPY_FLOAT) - 0.5 # [-0.5, 0.5]
        width = 2 * cp.sqrt(factor / self._topology.device_masses)
        self.set_velocities(velocities * width)

    @staticmethod
    def generator(masses: cp.ndarray, factor):
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
    def neighbor_list(self):
        return self._neighbor_list