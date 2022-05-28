#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : constraint.py
created time : 2021/10/09
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

from mdpy import env
from mdpy.core import Ensemble
from mdpy.utils import check_quantity_value
from mdpy.error import *
from mdpy.unit import *


class Constraint:
    def __init__(self) -> None:
        self._constraint_id = 0
        self._parent_ensemble = None
        self._forces = None
        self._potential_energy = None
        self._cutoff_radius = env.NUMPY_FLOAT(0)

    def __repr__(self) -> str:
        return "<mdpy.constraint.Constraint class>"

    __str__ = __repr__

    def __eq__(self, o: object) -> bool:
        if id(self) == id(o):
            return True
        return False

    def bind_ensemble(self, ensemble: Ensemble):
        raise NotImplementedError(
            "The subclass of mdpy.constraint.Constarint class should overload bind_ensemble method"
        )

    def sort_attributes(self):
        pass

    def _check_bound_state(self):
        if self._parent_ensemble == None:
            raise NonBoundedError(
                "%s has not been bounded to any Ensemble instance" % self
            )

    def update(self):
        raise NotImplementedError(
            "The subclass of mdpy.constraint.Constarint class should overload update method"
        )

    @property
    def constraint_id(self):
        return self._constraint_id

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
