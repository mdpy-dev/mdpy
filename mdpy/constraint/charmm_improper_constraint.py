#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : charmm_improper_constraint.py
created time : 2021/10/12
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np
from . import Constraint
from ..ensemble import Ensemble
from ..math import *

class CharmmImproperConstraint(Constraint):
    def __init__(self, force_id: int, force_group: int) -> None:
        super().__init__(force_id, force_group)
        self._improper_type, self._improper_matrix_id, self._improper_info = [], [], []
        self._num_impropers = 0

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        ensemble.add_constraints(self)
        self._improper_type, self._improper_matrix_id = [], []
        self._num_impropers = 0
        for improper in self._parent_ensemble.topology.impropers:
            self._improper_type.append('%s-%s-%s-%s' %(
                self._parent_ensemble.topology.particles[improper[0]].particle_name,
                self._parent_ensemble.topology.particles[improper[1]].particle_name,
                self._parent_ensemble.topology.particles[improper[2]].particle_name,
                self._parent_ensemble.topology.particles[improper[3]].particle_name
            ))
            self._improper_matrix_id.append([
                self._parent_ensemble.topology.particles[improper[0]].matrix_id,
                self._parent_ensemble.topology.particles[improper[1]].matrix_id,
                self._parent_ensemble.topology.particles[improper[2]].matrix_id,
                self._parent_ensemble.topology.particles[improper[3]].matrix_id
            ])
            self._num_impropers += 1

    def set_params(self, params):
        self._check_bound_state()
        self._improper_info = []
        for index, improper in enumerate(self._improper_type):
            self._improper_info.append(self._improper_matrix_id[index] + params[improper])

    def get_forces(self):
        self._check_bound_state()
        # V(improper) = Kpsi(psi - psi0)**2
        forces = np.zeros([self._parent_ensemble.topology.num_particles, SPATIAL_DIM])

    def get_potential_energy(self):
        pass

    @property
    def num_impropers(self):
        return self._num_impropers