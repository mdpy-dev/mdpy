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

    def __repr__(self) -> str:
        return '<mdpy.constraint.CharmmAngleConstraint object>'

    __str__ = __repr__

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

    def update(self):
        self._check_bound_state()
        # V(angle) = Ktheta(Theta - Theta0)**2
        self._forces = np.zeros([self._parent_ensemble.topology.num_particles, SPATIAL_DIM])
        self._potential_energy = 0
        for angle_info in self._angle_info:
            id1, id2, id3, k, theta0 = angle_info
            theta = get_pbc_angle(
                self._parent_ensemble.state.positions[id1, :], 
                self._parent_ensemble.state.positions[id2, :],
                self._parent_ensemble.state.positions[id3, :],
                *self._parent_ensemble.state.pbc_info, is_angular=False
            )
            # Force
            theta_rad = np.deg2rad(theta)
            force_val = 2 * k * (theta - theta0) / np.abs(np.sin(theta_rad)) # The - is declined by the minus of 1/sin\theta
            vec0 = unwrap_vec(
                self._parent_ensemble.state.positions[id1, :] - 
                self._parent_ensemble.state.positions[id2, :],
                *self._parent_ensemble.state.pbc_info
            )
            vec1 = unwrap_vec(
                self._parent_ensemble.state.positions[id3, :] - 
                self._parent_ensemble.state.positions[id2, :],
                *self._parent_ensemble.state.pbc_info
            )
            norm_vec0, norm_vec1 = np.linalg.norm(vec0), np.linalg.norm(vec1)
            vec0 = vec0 / norm_vec0
            vec1 = vec1 / norm_vec1
            force_vec0 = (vec1 - vec0 * np.cos(theta_rad)) / norm_vec0
            force_vec2 = (vec0 - vec1 * np.cos(theta_rad)) / norm_vec1
            self._forces[id1, :] += force_val * force_vec0
            self._forces[id2, :] -= force_val * (force_vec0 + force_vec2) 
            self._forces[id3, :] += force_val * force_vec2
            # Potential energy
            self._potential_energy += k * (theta - theta0)**2

    @property
    def num_angles(self):
        return self._num_angles