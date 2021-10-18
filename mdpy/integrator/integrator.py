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

from ..ensemble import Ensemble
from ..math import *
from ..unit import *

class Integrator:
    def __init__(self, time_step) -> None:
        self._time_step = check_quantity_value(time_step, default_time_unit) 

    def step(self, ensemble: Ensemble, num_steps: int=1):
        raise NotImplementedError(
            'The subclass of mdpy.integrator.Integrator class should overload step method'
        )

    @property
    def time_step(self):
        return self._time_step

    @time_step.setter
    def time_step(self, time_step):
        self._time_step = check_quantity_value(time_step, default_time_unit)