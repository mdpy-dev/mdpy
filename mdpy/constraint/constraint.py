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
from ..ensemble import Ensemble

class Constraint:
    def __init__(self, force_id: int=0, force_group: int=0) -> None:
        self._force_id = force_id
        self._force_group = force_group
        self._parent_ensemble = None

    def __repr__(self) -> str:
        return '<mdpy.Constraint class at %x>' %id(self)

    __str__ = __repr__

    def bind_ensemble(self, ensemble: Ensemble):
        raise NotImplementedError('The subclass of mdpy.constraint.Constarint class should overload bind_ensemble method')

    def set_params(self, params):
        raise NotImplementedError('The subclass of mdpy.constraint.Constarint class should overload set_param method')

    def _check_bound_state(self):
        if self._parent_ensemble == None:
            raise NonBoundedError(
                '%s has not been bounded to any Ensemble instance' %self
            )

    def update_neighbor(self):
        # Overloading of update_neighbor method is not compulsory
        raise NotImplementedError('The subclass of mdpy.constraint.Constarint class has not overloaded update_neighbor method') 

    def get_forces(self):
        raise NotImplementedError('The subclass of mdpy.constraint.Constarint class should overload get_forces method')

    def get_potential_energy(self):
        raise NotImplementedError('The subclass of mdpy.constraint.Constarint class should overload get_potential method')

    @property
    def force_id(self):
        return self._force_id

    @force_id.setter
    def force_id(self, index: int):
        self._force_id = index

    @property
    def force_group(self):
        return self._force_group

    @property
    def parent_ensemble(self):
        return self._parent_ensemble

    @parent_ensemble.setter
    def parent_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble