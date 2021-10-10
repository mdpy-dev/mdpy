#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : ensemble.py
created time : 2021/09/28
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np
from . import SPATIAL_DIM
from .core import Topology
from .error import *

class Ensemble:
    def __init__(self, positions: np.ndarray, topology: Topology) -> None:
        # Read input
        self._topology = topology

        self._matrix_shape = [self._topology.num_particles, SPATIAL_DIM]
        self.set_positions(positions)
        self._velocities = np.zeros(self._matrix_shape)
        self._forces = np.zeros(self._matrix_shape)

        self._total_energy = 0
        self._potential_energy = 0
        self._kinetic_energy = 0
        self._segments = []
        self._num_segments = 0
        self._constraints = []
        self._num_constraints = 0

    def _check_matrix_shape(self, matrix: np.ndarray):
        row, col = matrix.shape
        if self._topology.num_particles != row:
            raise ParticleConflictError(
                'The dimension of position matrix [%d, %d] do not match the number of particle %d contains in topology'
                %(row, col, self._topology.num_particles)
            )
        if matrix.shape[1] != SPATIAL_DIM:
            raise SpatialDimError(
                'The column dimension of matrix should be 3, instead of %d' %col
            )

    def create_segment(self, keywords):
        pass

    def add_constraints(self, *constraints):
        for constraint in constraints:
            self._constraints.append(constraint)
            self._num_constraints += 1
    
    def set_positions(self, positions: np.ndarray):
        self._check_matrix_shape(positions)
        self._positions = positions
    
    def set_velocities(self, velocities: np.ndarray):
        self._check_matrix_shape(velocities)
        self._velocities = velocities

    def update_force(self):
        self._forces = np.zeros(self._matrix_shape)
        for constraint in self._constraints:
            self._forces += constraint.get_forces()

    def update_energy(self):
        self._update_potential_energy()
        self._update_kinetic_energy()
        self._total_energy = self._potential_energy + self._kinetic_energy

    def _update_potential_energy(self):
        self._potential_energy = 0
        for constraint in self._constraints:
            self._potential_energy += constraint.get_potential_energy() 
    
    def _update_kinetic_energy(self):
        # Without reshape, the result of the first sum will be a 1d vector
        # , which will be a matrix after multiple with a 2d vector
        self._kinetic_energy = ((
            (self._velocities**2).sum(1).reshape(self.topology.num_particles, 1) * self._topology.masses
        ).sum() / 2)
    
    @property
    def topology(self):
        return self._topology

    @property
    def positions(self):
        return self._positions
    
    @property
    def velocities(self):
        return self._velocities
    
    @property
    def forces(self):
        return self._forces
    
    @property
    def total_energy(self):
        return self._total_energy
    
    @property
    def potential_energy(self):
        return self._potential_energy
    
    @property
    def kinetic_energy(self):
        return self._kinetic_energy

    @property
    def segments(self):
        return self._segments

    @property
    def num_segments(self):
        return self._num_segments

    @property
    def constraints(self):
        return self._constraints
    
    @property
    def num_constraints(self):
        return self._num_constraints