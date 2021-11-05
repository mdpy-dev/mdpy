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
import numba as nb
from .. import NUMPY_INT, NUMPY_FLOAT, NUMBA_INT, NUMBA_FLOAT
from . import Constraint
from ..ensemble import Ensemble
from ..math import *
from ..unit import *

epsilon0 = EPSILON0.value

class ElectrostaticConstraint(Constraint):
    def __init__(self, params=None, force_id: int = 0, force_group: int = 0) -> None:
        super().__init__(params, force_id=force_id, force_group=force_group)
        self._int_params = []
        self._float_params = []
        self._kernel = nb.njit(
            (NUMBA_INT[:, :], NUMBA_FLOAT[:, :], NUMBA_FLOAT[:, ::1], NUMBA_FLOAT[:, ::1], NUMBA_FLOAT[:, ::1])
        )(self.kernel)

    def __repr__(self) -> str:
        return '<mdpy.constraint.ElectrostaticConstraint object>'

    __str__ = __repr__

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._force_id = ensemble.constraints.index(self)
        self._int_params = []
        self._float_params = []
        for particle1 in self._parent_ensemble.topology.particles:
            id1 = particle1.matrix_id
            for particle2 in self._parent_ensemble.topology.particles[id1+1:]:
                id2 = particle2.matrix_id
                if not id2 in particle1.bonded_particles:
                    self._int_params.append([id1, id2])
                    self._float_params.append(self._parent_ensemble.topology.charges[[id1, id2], 0])
        self._int_params = np.vstack(self._int_params).astype(NUMPY_INT)
        self._float_params = np.vstack(self._float_params).astype(NUMPY_FLOAT)

    @staticmethod
    def kernel(int_params, float_params, positions, pbc_matrix, pbc_inv):
        forces = np.zeros_like(positions)
        potential_energy = forces[0, 0]
        num_params = int_params.shape[0]
        k = 4 * np.pi * epsilon0
        for pair in range(num_params):
            id1, id2 = int_params[pair, :]
            e1, e2 = float_params[pair, :]
            force_vec = unwrap_vec(
                positions[id2, :] - positions[id1, :],
                pbc_matrix, pbc_inv
            )
            dist = np.linalg.norm(force_vec)
            force_vec /= dist
            force_val = - e1 * e2 / k / dist**2
            force = force_vec * force_val
            forces[id1, :] += force
            forces[id2, :] -= force
            # Potential energy
            potential_energy += e1 * e2 / k / dist
        return forces, potential_energy

    def update(self):
        self._check_bound_state()
        self._forces, self._potential_energy = self._kernel(
            self._int_params, self._float_params, 
            self._parent_ensemble.state.positions, 
            *self._parent_ensemble.state.pbc_info
        )