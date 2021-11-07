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
from .. import env
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
        self._kernel = nb.njit((
            env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[:, ::1], env.NUMBA_INT[:, ::1], 
            env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[:, ::1]
        ))(self.kernel)

    def __repr__(self) -> str:
        return '<mdpy.constraint.ElectrostaticConstraint object>'

    __str__ = __repr__

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._force_id = ensemble.constraints.index(self)

    @staticmethod
    def kernel(positions, charges, bonded_particles, pbc_matrix, pbc_inv):
        forces = np.zeros_like(positions)
        potential_energy = forces[0, 0]
        num_particles = positions.shape[0]
        k = 4 * np.pi * epsilon0
        for id1 in range(num_particles):
            cur_bonded_particles = bonded_particles[id1, :][bonded_particles[id1, :] != -1]
            for id2 in range(id1+1, num_particles):
                if not id2 in cur_bonded_particles:
                    e1 = charges[id1, 0]
                    e2 = charges[id2, 0]
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
            self._parent_ensemble.state.positions,
            self._parent_ensemble.topology.charges,
            self._parent_ensemble.topology.bonded_particles, 
            *self._parent_ensemble.state.pbc_info
        )