#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : minimizer.py
created time : 2022/01/09
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

from mdpy.ensemble import Ensemble
from mdpy.unit import *

class Minimizer:
    def __init__(
        self, output_unit=kilojoule_permol, 
        output_unit_label='kj/mol',
        is_verbose=False, log_freq=5
    ) -> None:
        self._output_unit = output_unit
        self._output_unit_label = output_unit_label
        self._is_verbose = is_verbose
        self._log_freq = log_freq

    def minimize(self, ensemble: Ensemble, energy_tolerance=0.001, max_iterations: int=1000):
        raise NotImplementedError(
            'The subclass of mdpy.minimizer.Minimizer class should overload minimize method'
        )

    def _energy2str(self, energy: Quantity):
        return '%.5f %s' %(
            Quantity(energy, default_energy_unit).convert_to(self._output_unit).value,
            self._output_unit_label
        )