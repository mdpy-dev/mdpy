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
from .. import SPATIAL_DIM
from ..ensemble import Ensemble
from ..math import *

class CharmmAngleConstraint(Constraint):
    def __init__(self, params, force_id: int = 0, force_group: int = 0) -> None:
        super().__init__(params, force_id=force_id, force_group=force_group)
        self._angle_info = []
        self._num_angles = 0

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._force_id = ensemble.constraints.index(self)
        self._angle_info = []
        self._num_angles = 0
        for angle in self._parent_ensemble.topology.angles:
            angle_type = '%s-%s-%s' %(
                self._parent_ensemble.topology.particles[angle[0]].particle_name,
                self._parent_ensemble.topology.particles[angle[1]].particle_name,
                self._parent_ensemble.topology.particles[angle[2]].particle_name
            )
            matrix_id = [
                self._parent_ensemble.topology.particles[angle[0]].matrix_id,
                self._parent_ensemble.topology.particles[angle[1]].matrix_id,
                self._parent_ensemble.topology.particles[angle[2]].matrix_id
            ]
            self._angle_info.append(matrix_id + self._params[angle_type])
            self._num_angles += 1
    
    def get_forces(self):
        self._check_bound_state()
        # V(angle) = Ktheta(Theta - Theta0)**2
        forces = np.zeros([self._parent_ensemble.topology.num_particles, SPATIAL_DIM])
        for angle_info in self._angle_info:
            id1, id2, id3, k, theta0 = angle_info
            theta = get_angle(
                self._parent_ensemble.state.positions[id1, :], 
                self._parent_ensemble.state.positions[id2, :],
                self._parent_ensemble.state.positions[id3, :], is_angular=False
            )
            theta_rad = np.deg2rad(theta)
            force_val = 2 * k * (theta - theta0) / np.abs(np.sin(theta_rad)) # The - is declined by the minus of 1/sin\theta
            vec0 = self._parent_ensemble.state.positions[id1, :] - self._parent_ensemble.state.positions[id2, :]
            vec1 = self._parent_ensemble.state.positions[id3, :] - self._parent_ensemble.state.positions[id2, :]
            norm_vec0, norm_vec1 = np.linalg.norm(vec0), np.linalg.norm(vec1)
            vec0 = vec0 / norm_vec0
            vec1 = vec1 / norm_vec1
            force_vec0 = (vec1 - vec0 * np.cos(theta_rad)) / norm_vec0
            force_vec2 = (vec0 - vec1 * np.cos(theta_rad)) / norm_vec1
            forces[id1, :] += force_val * force_vec0
            forces[id2, :] -= force_val * (force_vec0 + force_vec2) 
            forces[id3, :] += force_val * force_vec2
        return forces

    def get_potential_energy(self):
        self._check_bound_state()
        potential_energy = 0
        for angle_info in self._angle_info:
            id1, id2, id3, k, theta0 = angle_info
            theta = get_angle(
                self._parent_ensemble.state.positions[id1, :], 
                self._parent_ensemble.state.positions[id2, :],
                self._parent_ensemble.state.positions[id3, :], is_angular=False
            )
            potential_energy += k * (theta - theta0)**2
        return potential_energy


    @property
    def num_angles(self):
        return self._num_angles