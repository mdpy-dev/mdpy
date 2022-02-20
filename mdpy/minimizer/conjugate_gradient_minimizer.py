#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : conjugate_gradient_minimizer.py
created time : 2022/01/09
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np
from . import Minimizer
from .. import SPATIAL_DIM
from ..ensemble import Ensemble
from ..unit import *
from ..utils import *

class ConjugateGradientMinimizer(Minimizer):
    def __init__(
        self, theta=0.001, force_tol=0.01, max_sub_iterations=10,
        output_unit=kilojoule_permol, 
        output_unit_label='kj/mol', is_verbose=False, log_freq=5
    ) -> None:
        super().__init__(
            output_unit=output_unit, 
            output_unit_label=output_unit_label, 
            is_verbose=is_verbose, 
            log_freq=log_freq
        )
        self._theta = theta
        self._force_tol = force_tol
        self._max_sub_iterations = max_sub_iterations

    def minimize(self, ensemble: Ensemble, energy_tolerance=0.001, max_iterations: int = 1000):
        ensemble.update()
        cur_iteration = 0
        cur_energy = ensemble.potential_energy
        print('Start energy minimization with steepest decent method')
        print('Initial potential energy: %s' %self._energy2str(cur_energy))
        while cur_iteration < max_iterations:
            cur_sub_iteration = 0
            pre_energy = cur_energy
            cur_positions = ensemble.state.positions.copy()
            f0 = ensemble.forces.reshape([-1, 1])
            while cur_sub_iteration < self._max_sub_iterations:
                d = np.zeros([ensemble.topology.num_particles*SPATIAL_DIM, 1])
                cur_f = ensemble.forces.reshape([-1, 1])
                pre_f = cur_f.copy()
                t = cur_f.copy()
                ensemble.state.set_positions(wrap_positions(
                    ensemble.state.positions + self._theta * t.reshape([-1, 3]),
                    *ensemble.state.pbc_info
                ))
                ensemble.update()
                omega = - (ensemble.forces.reshape([-1, 1]) - cur_f) / self._theta
                alpha = np.matmul(cur_f.T, cur_f) / (np.matmul(omega.T, t))
                d += alpha * t
                cur_f, pre_f = cur_f + alpha * omega, cur_f
                if (cur_f**2).sum() / (f0**2).sum() < self._force_tol:
                    break
                beta = np.matmul(cur_f.T, cur_f) / (np.matmul(pre_f.T, pre_f))
                t = cur_f + beta * t
                cur_sub_iteration += 1
            ensemble.state.set_positions(wrap_positions(
                cur_positions + d.reshape([-1, 3]),
                *ensemble.state.pbc_info
            ))
            ensemble.update()
            cur_energy = ensemble.potential_energy
            energy_error = np.abs((cur_energy - pre_energy) / pre_energy)
            cur_iteration += 1
            if self._is_verbose and cur_iteration % self._log_freq == 0:
                print('Iteration %d: %s %.4f' %(cur_iteration, self._energy2str(cur_energy), energy_error))
            if energy_error < energy_tolerance:
                print('Penultimate potential energy %s' %self._energy2str(pre_energy))
                print('Final potential energy %s' %self._energy2str(cur_energy))
                print('Energy error: %e < %e' %(energy_error, energy_tolerance))
                return None
        print('Final potential energy: %s' %self._energy2str(cur_energy))