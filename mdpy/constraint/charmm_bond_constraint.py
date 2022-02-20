#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : amber_bond_constraint.py
created time : 2021/10/09
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

class CharmmBondConstraint(Constraint):
    def __init__(self, params, force_id: int = 0, force_group: int = 0) -> None:
        super().__init__(params, force_id=force_id, force_group=force_group)
        self._int_params = []
        self._float_params = []
        self._num_bonds = 0
        self._kernel = nb.njit(
            (env.NUMBA_INT[:, :], env.NUMBA_FLOAT[:, :], env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[:, ::1])
        )(self.kernel)

    def __repr__(self) -> str:
        return '<mdpy.constraint.CharmmBondConstraint object>'

    __str__ = __repr__

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._force_id = ensemble.constraints.index(self)
        self._int_params = []
        self._float_params = []
        self._num_bonds = 0
        for bond in self._parent_ensemble.topology.bonds:
            bond_type = '%s-%s' %(
                self._parent_ensemble.topology.particles[bond[0]].particle_name,
                self._parent_ensemble.topology.particles[bond[1]].particle_name
            )
            self._int_params.append([
                self._parent_ensemble.topology.particles[bond[0]].matrix_id,
                self._parent_ensemble.topology.particles[bond[1]].matrix_id
            ])
            self._float_params.append(self._params[bond_type])
            self._num_bonds += 1
        self._int_params = np.vstack(self._int_params).astype(env.NUMPY_INT)
        self._float_params = np.vstack(self._float_params).astype(env.NUMPY_FLOAT)

    @staticmethod
    def kernel(int_params, float_params, positions, pbc_matrix, pbc_inv):
        forces = np.zeros_like(positions)
        potential_energy = forces[0, 0]
        num_params = int_params.shape[0]
        for bond in range(num_params):
            id1, id2, = int_params[bond, :]
            k, r0 = float_params[bond, :]
            force_vec = unwrap_vec(
                positions[id2, :] - positions[id1, :], 
                pbc_matrix, pbc_inv
            )
            r = np.linalg.norm(force_vec)
            force_vec /= r
            # Forces
            force_val = 2 * k * (r - r0)
            force = force_val * force_vec
            forces[id1, :] += force
            forces[id2, :] -= force
            # Potential energy
            potential_energy += k * (r - r0)**2
        return forces, potential_energy

    def update(self):
        self._check_bound_state()
        self._forces, self._potential_energy = self._kernel(
            self._int_params, self._float_params, 
            self._parent_ensemble.state.positions, 
            *self._parent_ensemble.state.pbc_info
        )

    @property
    def num_bonds(self):
        return self._num_bonds