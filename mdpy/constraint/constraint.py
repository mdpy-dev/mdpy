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

from .. import env
from ..utils import check_quantity_value
from ..error import *
from ..unit import *
from ..ensemble import Ensemble

class Constraint:
    def __init__(self, params, force_id: int=0, force_group: int=0) -> None:
        self._params = params
        self._force_id = force_id
        self._force_group = force_group
        self._parent_ensemble = None
        self._forces = None
        self._potential_energy = None
        self._cutoff_radius = env.NUMPY_FLOAT(0)

    def __repr__(self) -> str:
        return '<mdpy.constraint.Constraint class>'

    __str__ = __repr__

    def __eq__(self, o: object) -> bool:
        if id(self) == id(o):
            return True
        return False

    def bind_ensemble(self, ensemble: Ensemble):
        raise NotImplementedError('The subclass of mdpy.constraint.Constarint class should overload bind_ensemble method')

    def _check_bound_state(self):
        if self._parent_ensemble == None:
            raise NonBoundedError(
                '%s has not been bounded to any Ensemble instance' %self
            )

    def update(self):
        raise NotImplementedError('The subclass of mdpy.constraint.Constarint class should overload update method')

    @property
    def force_id(self):
        return self._force_id

    @property
    def force_group(self):
        return self._force_group

    @property
    def parent_ensemble(self):
        return self._parent_ensemble

    @property
    def forces(self):
        return self._forces

    @property
    def potential_energy(self):
        return self._potential_energy

    def set_cutoff_radius(self, val):
        self._cutoff_radius = check_quantity_value(val, default_length_unit)

    @property
    def cutoff_radius(self):
        return self._cutoff_radius