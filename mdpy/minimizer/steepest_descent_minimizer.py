#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : steepest_descent_minimizer.py
created time : 2022/01/09
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np
from . import Minimizer
from ..ensemble import Ensemble
from ..unit import *
from ..math import *

class SteepestDescentMinimizer(Minimizer):
    def __init__(
        self, alpha=0.1,
        output_unit=kilojoule_permol, 
        output_unit_label='kj/mol', is_verbose=False
    ) -> None:
        super().__init__(
            output_unit=output_unit, 
            output_unit_label=output_unit_label, 
            is_verbose=is_verbose
        )
        self._alpha = alpha

    def minimize(self, ensemble: Ensemble, energy_tolerance=0.001, max_iterations: int = 1000):
        ensemble.update()
        cur_iteration = 0
        cur_energy = ensemble.potential_energy
        pre_energy = cur_energy
        print('Start energy minimization with steepest decent method')
        print('Initial potential energy: %s' %self._energy2str(cur_energy))
        while cur_iteration < max_iterations:
            ensemble.state.set_positions(wrap_positions(
                ensemble.state.positions + self._alpha * ensemble.forces / np.linalg.norm(ensemble.forces, axis=1).reshape([-1, 1]),
                *ensemble.state.pbc_info
            ))
            ensemble.update()
            cur_energy = ensemble.potential_energy
            energy_error = Quantity(np.abs(cur_energy - pre_energy), default_energy_unit).convert_to(self._output_unit).value
            if self._is_verbose:
                print('Iteration %d: %s %.4f' %(cur_iteration+1, self._energy2str(cur_energy), energy_error))
            if energy_error < energy_tolerance:
                print('Penultimate potential energy %s' %self._energy2str(pre_energy))
                print('Final potential energy %s' %self._energy2str(cur_energy))
                print('Energy error: %e %s < %e %s' %(
                    energy_error, self._output_unit_label, energy_tolerance, self._output_unit_label
                ))
                return None
            pre_energy = cur_energy
            cur_iteration += 1
        print('Final potential energy: %s' %self._energy2str(cur_energy))