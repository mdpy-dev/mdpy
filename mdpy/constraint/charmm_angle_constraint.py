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
import numba as nb
from . import Constraint
from .. import env
from ..ensemble import Ensemble
from ..utils import *

class CharmmAngleConstraint(Constraint):
    def __init__(self, params, force_id: int = 0, force_group: int = 0) -> None:
        super().__init__(params, force_id=force_id, force_group=force_group)
        self._int_params = []
        self._float_params = []
        self._num_angles = 0
        self._kernel = nb.njit(
            (env.NUMBA_INT[:, :], env.NUMBA_FLOAT[:, :], env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[:, ::1])
        )(self.kernel)

    def __repr__(self) -> str:
        return '<mdpy.constraint.CharmmAngleConstraint object>'

    __str__ = __repr__

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._force_id = ensemble.constraints.index(self)
        self._int_params = []
        self._float_params = []
        self._num_angles = 0
        for angle in self._parent_ensemble.topology.angles:
            angle_type = '%s-%s-%s' %(
                self._parent_ensemble.topology.particles[angle[0]].particle_name,
                self._parent_ensemble.topology.particles[angle[1]].particle_name,
                self._parent_ensemble.topology.particles[angle[2]].particle_name
            )
            self._int_params.append([
                self._parent_ensemble.topology.particles[angle[0]].matrix_id,
                self._parent_ensemble.topology.particles[angle[1]].matrix_id,
                self._parent_ensemble.topology.particles[angle[2]].matrix_id
            ])
            self._float_params.append(self._params[angle_type])
            self._num_angles += 1
        self._int_params = np.vstack(self._int_params).astype(env.NUMPY_INT)
        self._float_params = np.vstack(self._float_params).astype(env.NUMPY_FLOAT)

    @staticmethod
    def kernel(int_params, float_params, positions, pbc_matrix, pbc_inv):
        forces = np.zeros_like(positions)
        potential_energy = forces[0, 0]
        num_angles = int_params.shape[0]
        for angle in range(num_angles):
            id1, id2, id3 = int_params[angle]
            k, theta0, ku, u0 = float_params[angle, :]
            r21 = unwrap_vec(
                positions[id1, :] - positions[id2, :],
                pbc_matrix, pbc_inv
            )
            l21 = np.linalg.norm(r21)
            r23 = unwrap_vec(
                positions[id3, :] - positions[id2, :],
                pbc_matrix, pbc_inv
            )
            l23 = np.linalg.norm(r23)
            cos_theta = np.dot(r21, r23) / (l21 * l23)
            theta = np.arccos(cos_theta)
            # Force
            force_val = - 2 * k * (theta - theta0) 
            vec_norm = np.cross(r21, r23)
            force_vec1 = get_unit_vec(np.cross(r21, vec_norm)) / l21
            force_vec3 = get_unit_vec(np.cross(-r23, vec_norm)) / l23
            forces[id1, :] += force_val * force_vec1
            forces[id2, :] -= force_val * (force_vec1 + force_vec3) 
            forces[id3, :] += force_val * force_vec3
            # Potential energy
            potential_energy += k * (theta - theta0)**2
            # Urey-Bradley
            r13 = unwrap_vec(
                positions[id3, :] - positions[id1, :],
                pbc_matrix, pbc_inv
            )
            l13 = np.linalg.norm(r13)
            force_val = 2 * ku * (l13 - u0)
            force_vec = r13 / l13 
            forces[id1, :] += force_val * force_vec
            forces[id3, :] -= force_val * force_vec
            potential_energy += ku * (l13 - u0)**2
        return forces, potential_energy

    def update(self):
        self._check_bound_state()
        # V(angle) = Ktheta(Theta - Theta0)**2
        self._forces, self._potential_energy = self._kernel(
            self._int_params, self._float_params, 
            self._parent_ensemble.state.positions, 
            *self._parent_ensemble.state.pbc_info
        )

    @property
    def num_angles(self):
        return self._num_angles