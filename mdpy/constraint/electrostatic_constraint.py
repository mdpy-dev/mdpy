#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : electrostatic_constraint.py
created time : 2021/10/13
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np

from . import Constraint
from .. import SPATIAL_DIM
from ..ensemble import Ensemble
from ..math import *
from ..unit import *

class ElectrostaticConstraint(Constraint):
    def __init__(self, params=None, force_id: int = 0, force_group: int = 0) -> None:
        super().__init__(params, force_id=force_id, force_group=force_group)
        self._charges = []

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._force_id = ensemble.constraints.index(self)
        self._charges = self._parent_ensemble.topology.charges

    def get_forces(self):
        self._check_bound_state()
        k = 4 * np.pi * EPSILON0.value
        forces = np.zeros([self._parent_ensemble.topology.num_particles, SPATIAL_DIM])
        scaled_position = np.dot(
            self._parent_ensemble.state.positions,
            self._parent_ensemble.state.pbc_inv
        )
        for particle in self._parent_ensemble.topology.particles:
            id1 = particle.matrix_id
            scaled_position_diff = scaled_position[id1, :] - scaled_position[id1+1:, :]
            scaled_position_diff -= np.round(scaled_position_diff)
            dist_list = np.sqrt(((np.dot(
                scaled_position_diff, 
                self._parent_ensemble.state.pbc_matrix
            ))**2).sum(1))
            for index, dist in enumerate(dist_list):
                id2 = index + id1 + 1
                force_val = - self._charges[id1] * self._charges[id2] / k / dist**2
                force_vec = unwrap_vec(get_unit_vec(
                    self._parent_ensemble.state.positions[id2] - 
                    self._parent_ensemble.state.positions[id1]
                ), self._parent_ensemble.state.pbc_matrix, self._parent_ensemble.state.pbc_inv)
                force = force_vec * force_val
                forces[id1, :] += force
                forces[id2, :] -= force
        return forces

    def get_potential_energy(self):
        self._check_bound_state()
        potential_energy = 0
        k = 4 * np.pi * EPSILON0.value
        scaled_position = np.dot(
            self._parent_ensemble.state.positions,
            self._parent_ensemble.state.pbc_inv
        )
        for particle in self._parent_ensemble.topology.particles[:-1]:
            id1 = particle.matrix_id
            scaled_position_diff = scaled_position[id1, :] - scaled_position[id1+1:, :]
            scaled_position_diff -= np.round(scaled_position_diff)
            dist_list = np.sqrt(((np.dot(
                scaled_position_diff, 
                self._parent_ensemble.state.pbc_matrix
            ))**2).sum(1))
            for index, dist in enumerate(dist_list):
                id2 = index + id1 + 1
                potential_energy += self._charges[id1] * self._charges[id2] / k / dist
        return potential_energy