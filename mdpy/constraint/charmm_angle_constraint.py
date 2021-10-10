#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : charmm_angle_constraint.py
created time : 2021/10/10
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np
from . import Constraint
from ..ensemble import Ensemble
from ..math import *

class CharmmAngleConstraint(Constraint):
    def __init__(self, force_id: int, force_group: int) -> None:
        super().__init__(force_id, force_group)
        self._angle_type, self._angle_matrix_id, self._angle_info = [], [], []
        self._num_angles = 0

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        ensemble.add_constraints(self)
        self._angle_type, self._angle_matrix_id = [], []
        self._num_angles = 0
        for angle in self._parent_ensemble.topology.angles:
            self._angle_type.append('%s-%s-%s' %(
                self._parent_ensemble.topology.particles[angle[0]].particle_name,
                self._parent_ensemble.topology.particles[angle[1]].particle_name,
                self._parent_ensemble.topology.particles[angle[2]].particle_name
            ))
            self._angle_matrix_id.append([
                self._parent_ensemble.topology.particles[angle[0]].matrix_id,
                self._parent_ensemble.topology.particles[angle[1]].matrix_id,
                self._parent_ensemble.topology.particles[angle[2]].matrix_id
            ])
            self._num_angles += 1

    def set_params(self, params):
        self._test_bound_state()
        self._angle_info = []
        for index, angle, in enumerate(self._angle_type):
            self._angle_info.append(self._angle_matrix_id[index] + params[angle])
    
    def get_forces(self):
        # V(angle) = Ktheta(Theta - Theta0)**2
        forces = np.zeros([self._parent_ensemble.topology.num_particles, 3])
        for angle_info in self._angle_info:
            id1, id2, id3, k, theta0 = angle_info

    def get_potential_energy(self):
        potential_energy = 0
        for angle_info in self._angle_info:
            id1, id2, id3, k, theta0 = angle_info
            angle = get_angle(
                self._parent_ensemble.positions[id1, :], 
                self._parent_ensemble.positions[id2, :],
                self._parent_ensemble.positions[id3, :], is_angular=False
            )
            potential_energy += k * (angle - theta0)**2
        return potential_energy


    @property
    def num_angles(self):
        return self._num_angles