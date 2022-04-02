#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : integrator.py
created time : 2021/10/18
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

from mdpy.ensemble import Ensemble
from mdpy.utils import *
from mdpy.unit import *

class Integrator:
    def __init__(self, time_step) -> None:
        self._time_step = check_quantity_value(time_step, default_time_unit)
        self._cur_positions = None
        self._pre_positions = None

    def erase_cache(self):
        self._cur_positions = None
        self._pre_positions = None

    def integrate(self, ensemble: Ensemble, num_steps: int=1):
        raise NotImplementedError(
            'The subclass of mdpy.integrator.Integrator class should overload integrate method'
        )

    @property
    def time_step(self):
        return self._time_step

    @time_step.setter
    def time_step(self, time_step):
        self._time_step = check_quantity_value(time_step, default_time_unit)

    @property
    def cur_positions(self):
        return self._cur_positions

    @property
    def pre_positions(self):
        return self._pre_positions

    @property
    def is_cached(self):
        if isinstance(self._cur_positions, type(None)):
            self.erase_cache()
            return False
        return True