#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : charmm_dihedral_constraint.py
created time : 2021/10/11
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np
from . import Constraint
from ..ensemble import Ensemble
from ..math import *

class CharmmDihedralConstraint(Constraint):
    def __init__(self, force_id: int, force_group: int) -> None:
        super().__init__(force_id, force_group)
        self._dihedral_type, self._dihedral_matrix_id, self._dihedral_info = [], [], []
        self._num_dihedrals = 0

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        ensemble.add_constraints(self)
        self._dihedral_type, self._dihedral_matrix_id = [], []
        self._num_dihedrals = 0
        for dihedral in self._parent_ensemble.topology.dihedrals:
            self._dihedral_type.append('%s-%s-%s-%s' %(
                self._parent_ensemble.topology.particles[dihedral[0]].particle_name,
                self._parent_ensemble.topology.particles[dihedral[1]].particle_name,
                self._parent_ensemble.topology.particles[dihedral[2]].particle_name,
                self._parent_ensemble.topology.particles[dihedral[3]].particle_name
            ))
            self._dihedral_matrix_id.append([
                self._parent_ensemble.topology.particles[dihedral[0]].matrix_id,
                self._parent_ensemble.topology.particles[dihedral[1]].matrix_id,
                self._parent_ensemble.topology.particles[dihedral[2]].matrix_id,
                self._parent_ensemble.topology.particles[dihedral[3]].matrix_id
            ])
            self._num_dihedrals += 1

    def set_params(self, params):
        self._test_bound_state()
        self._dihedral_info = []
        for index, dihedral in enumerate (self._dihedral_type):
            self._dihedral_info.append(self._dihedral_matrix_id[index] + params[dihedral])

    def get_forces(self):
        # V(dihedral) = Kchi(1 + cos(n(chi) - delta))
        forces = np.zeros([self._parent_ensemble.topology.num_particles, 3])
        for dihedral_info in self._dihedral_info:
            id1, id2, id3, id4, k, n, delta = dihedral_info
            theta = get_dihedral(
                self._parent_ensemble.positions[id1, :], 
                self._parent_ensemble.positions[id2, :],
                self._parent_ensemble.positions[id3, :], 
                self._parent_ensemble.positions[id4, :],
                is_angular=False
            )

    def get_potential_energy(self):
        potential_energy = 0
        for dihedral_info in self._dihedral_info:
            id1, id2, id3, id4, k, n, delta = dihedral_info
            theta = get_dihedral(
                self._parent_ensemble.positions[id1, :], 
                self._parent_ensemble.positions[id2, :],
                self._parent_ensemble.positions[id3, :], 
                self._parent_ensemble.positions[id4, :],
                is_angular=False
            )
            potential_energy += k * (1 + np.cos(n*theta - delta))
        return potential_energy

    @property
    def num_dihedrals(self):
        return self._num_dihedrals