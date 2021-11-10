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
from cupy.cuda.nvtx import RangePush, RangePop
from .cell_list import CellList
from .topology import Topology
from .. import SPATIAL_DIM, env
from ..unit import *
from ..error import *
from ..math import *

class State:
    def __init__(self, topology: Topology) -> None:
        self._particles = topology.particles
        self._masses = topology.masses
        self._num_particles = len(self._particles)
        self._matrix_shape = [self._num_particles, SPATIAL_DIM]
        self._positions = np.zeros(self._matrix_shape, dtype=env.NUMPY_FLOAT)
        self._velocities = np.zeros(self._matrix_shape, dtype=env.NUMPY_FLOAT)

        self._pbc_matrix = np.zeros([SPATIAL_DIM, SPATIAL_DIM], dtype=env.NUMPY_FLOAT, order='C')
        self._pbc_inv = np.zeros([SPATIAL_DIM, SPATIAL_DIM], dtype=env.NUMPY_FLOAT, order='C')
        self._cell_list = CellList()

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

    def check_pbc_matrix(self):
        if np.linalg.det(self._pbc_matrix) == 0:
            raise PBCPoorDefinedError(
                'PBC of %s is poor defined. Two or more column vectors are linear corellated'
            )

    def set_pbc_matrix(self, pbc_matrix):
        pbc_matrix = check_quantity_value(pbc_matrix, default_length_unit)
        row, col = pbc_matrix.shape
        if row != SPATIAL_DIM or col != SPATIAL_DIM:
            raise SpatialDimError(
                'The pbc matrix should have shape [%d, %d], while matrix [%d %d] is provided'
                %(SPATIAL_DIM, SPATIAL_DIM, row, col)
            )
        # The origin define of pbc_matrix is the stack of 3 column vector
        # While in MDPy the position is in shape of n x 3
        # So the scaled position will be Position * PBC instead of PBC * Position as usual
        self._pbc_matrix = np.ascontiguousarray(pbc_matrix.T, dtype=env.NUMPY_FLOAT)
        self.check_pbc_matrix()
        self._pbc_inv = np.ascontiguousarray(np.linalg.inv(self._pbc_matrix), dtype=env.NUMPY_FLOAT)
        self._cell_list.set_pbc_matrix(self._pbc_matrix)

    def set_positions(self, positions: np.ndarray):
        self._check_matrix_shape(positions)
        self._positions = positions.astype(env.NUMPY_FLOAT)
        RangePush('Cell list creation')
        self._cell_list.update(self._positions)
        RangePop()
    
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
    def velocities(self):
        return self._velocities

    @property
    def matrix_shape(self):
        return self._matrix_shape

    @property
    def pbc_matrix(self):
        return self._pbc_matrix

    @property
    def pbc_inv(self):
        return self._pbc_inv

    @property
    def pbc_info(self):
        return self._pbc_matrix, self._pbc_inv

    @property
    def cell_list(self):
        return self._cell_list