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
from . import Constraint
from .. import SPATIAL_DIM
from ..ensemble import Ensemble
from ..math import *

class ElectrostaticConstraint(Constraint):
    def __init__(self, force_id: int = 0, force_group: int = 0) -> None:
        super().__init__(force_id=force_id, force_group=force_group)

    def bind_ensemble(self, ensemble: Ensemble):
        pass

    def get_forces(self):
        self._check_bound_state()
        forces = np.zeros([self._parent_ensemble.topology.num_particles, SPATIAL_DIM])

        return forces

    def get_potential_energy(self):
        self._check_bound_state()
        potential_energy = 0

        return potential_energy