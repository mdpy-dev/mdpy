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
from . import Constraint
from .. import SPATIAL_DIM
from ..ensemble import Ensemble
from ..math import *

class CharmmBondConstraint(Constraint):
    def __init__(self, params, force_id: int = 0, force_group: int = 0) -> None:
        super().__init__(params, force_id=force_id, force_group=force_group)
        self._bond_info = []
        self._num_bonds = 0

    def __repr__(self) -> str:
        return '<mdpy.constraint.CharmmBondConstraint object>'

    __str__ = __repr__

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._force_id = ensemble.constraints.index(self)
        self._bond_info = []
        self._num_bonds = 0
        for bond in self._parent_ensemble.topology.bonds:
            bond_type = '%s-%s' %(
                self._parent_ensemble.topology.particles[bond[0]].particle_name,
                self._parent_ensemble.topology.particles[bond[1]].particle_name
            )
            matrix_id = [
                self._parent_ensemble.topology.particles[bond[0]].matrix_id,
                self._parent_ensemble.topology.particles[bond[1]].matrix_id
            ]
            self._bond_info.append(matrix_id + self._params[bond_type])
            self._num_bonds += 1

    def get_forces(self):
        self._check_bound_state()
        forces = np.zeros([self._parent_ensemble.topology.num_particles, SPATIAL_DIM])
        for bond_info in self._bond_info:
            id1, id2, k, r0 = bond_info
            r = get_pbc_bond(
                self._parent_ensemble.state.positions[id1, :], 
                self._parent_ensemble.state.positions[id2, :],
                self._parent_ensemble.state.pbc_matrix,
                self._parent_ensemble.state.pbc_inv
            )
            force_val = 2 * k * (r - r0)
            force_vec = get_unit_vec(
                self._parent_ensemble.state.positions[id2, :] - self._parent_ensemble.state.positions[id1, :]
            )
            force = force_val * force_vec
            forces[id1, :] += force
            forces[id2, :] -= force
        return forces

    def get_potential_energy(self):
        self._check_bound_state()
        potential_energy = 0
        for bond_info in self._bond_info:
            id1, id2, k, r0 = bond_info
            r = get_pbc_bond(
                self._parent_ensemble.state.positions[id1, :], 
                self._parent_ensemble.state.positions[id2, :],
                self._parent_ensemble.state.pbc_matrix,
                self._parent_ensemble.state.pbc_inv
            )
            potential_energy += k * (r - r0)**2
        return potential_energy

    @property
    def num_bonds(self):
        return self._num_bonds