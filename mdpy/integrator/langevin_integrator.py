#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : langevin_integrator.py
created time : 2021/10/25
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np
from numpy.random import randn
from . import Integrator
from ..ensemble import Ensemble
from ..unit import *
from ..utils import *

class LangevinIntegrator(Integrator):
    def __init__(self, time_step, temperature, friction_factor) -> None:
        super().__init__(time_step)
        self._temperature = check_quantity_value(temperature, default_temperature_unit)
        self._gamma = check_quantity_value(friction_factor, 1/default_time_unit)
        self._kbt = (Quantity(self._temperature, default_temperature_unit) * KB).convert_to(default_energy_unit).value
        self._sigma = np.sqrt(2*self._kbt*self._gamma)
        self._a = (
            (1 - self._gamma * self._time_step / 2) / 
            (1 + self._gamma * self._time_step / 2)
        )
        self._b = 1 / (1 + self._gamma * self._time_step / 2)
        self._time_step_square = self._time_step**2
        self._time_step_3_over_2 = self._time_step**(3/2)
        self._time_step_sqrt = np.sqrt(self._time_step)
        self._cur_velocities = None
        self._pre_velocities = None
        self._cur_acceleration = None
        self._pre_acceleration = None

    def integrate(self, ensemble: Ensemble, num_steps: int=1):
        # Setting variables
        cur_step = 0
        masses = ensemble.topology.masses
        sqrt_masses = np.sqrt(masses)
        self._cur_acceleration = ensemble.forces / masses
        if self.is_cached == False:
            self._cur_velocities = ensemble.state.velocities
            self._pre_velocities = ensemble.state.velocities
            self._cur_positions = ensemble.state.positions
            self._pre_positions = ensemble.state.positions
            self._matrix_shape = list(self._pre_positions.shape)
        while cur_step < num_steps:
            # Iterate position
            if cur_step != 0:
                ensemble.update()
            ensemble.update()
            xi_over_sqrt_masses = randn(*self._matrix_shape) / sqrt_masses
            self._pre_acceleration = self._cur_acceleration
            self._cur_positions, self._pre_positions = (
                self._pre_positions + 
                self._b * self._time_step * self._pre_velocities - 
                self._b * self._time_step_square / 2 * self._pre_acceleration +
                self._b * self._sigma * self._time_step_3_over_2 / 2 * xi_over_sqrt_masses
            ), self._cur_positions
            # Update position
            self._cur_positions = wrap_positions(self._cur_positions, *ensemble.state.pbc_info)
            ensemble.state.set_positions(self._cur_positions)
            ensemble.update()
            self._cur_acceleration = ensemble.forces / masses
            self._cur_velocities, self._pre_velocities = (
                self._a * self._pre_velocities -
                self._time_step / 2 * (self._a*self._pre_acceleration + self._cur_acceleration) +
                self._b * self._sigma * self._time_step_sqrt * xi_over_sqrt_masses
            ), self._cur_velocities
            cur_step += 1
        ensemble.state.set_velocities(self._cur_velocities)