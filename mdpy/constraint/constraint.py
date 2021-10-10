#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : constraint.py
created time : 2021/10/09
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

from ..error import *

class Constraint:
    def __init__(self, force_id: int, force_group: int) -> None:
        self._force_id = force_id
        self._force_group = force_group
        self._parent_ensemble = None

    def bind_ensemble(self, ensemble):
        raise NotImplementedError('The subclass of mdpy.constraint.Constarint class should overload bind_ensemble method')

    def set_param(self, param):
        raise NotImplementedError('The subclass of mdpy.constraint.Constarint class should overload set_param method')

    def _test_bound_state(self):
        if self._parent_ensemble == None:
            raise NonBoundedError(
                '%s has not been bounded to any Ensemble instance' %self
            )

    def get_forces(self):
        raise NotImplementedError('The subclass of mdpy.constraint.Constarint class should overload get_forces method')

    def get_potential_energy(self):
        raise NotImplementedError('The subclass of mdpy.constraint.Constarint class should overload get_potential method')