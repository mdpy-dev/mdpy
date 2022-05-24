#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : langevin_integrator.py
created time : 2021/10/25
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import cupy as cp
import numpy as np
from mdpy.core import Ensemble
from mdpy.integrator import Integrator
from mdpy.unit import *
from mdpy.utils import *

class LangevinIntegrator(Integrator):
    def __init__(self, time_step, temperature, friction_rate, neighbor_list_update_freq=10) -> None:
        super().__init__(time_step, neighbor_list_update_freq)
        self._temperature = check_quantity_value(temperature, default_temperature_unit)
        self._gamma = check_quantity_value(friction_rate, 1/default_time_unit)
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
        masses = ensemble.topology.device_masses
        sqrt_masses = cp.sqrt(masses)
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
            xi_over_sqrt_masses = cp.random.randn(*self._matrix_shape) / sqrt_masses
            self._pre_acceleration = self._cur_acceleration
            self._cur_positions, self._pre_positions = (
                self._pre_positions +
                self._b * self._time_step * self._pre_velocities -
                self._b * self._time_step_square / 2 * self._pre_acceleration +
                self._b * self._sigma * self._time_step_3_over_2 / 2 * xi_over_sqrt_masses
            ), self._cur_positions
            # Update position
            ensemble.state.set_positions(self._cur_positions)
            if cur_step % self._neighbor_list_update_freq == 0:
                ensemble.update_tile_list()
            ensemble.update()
            self._cur_acceleration = ensemble.forces / masses
            self._cur_velocities, self._pre_velocities = (
                self._a * self._pre_velocities -
                self._time_step / 2 * (self._a*self._pre_acceleration + self._cur_acceleration) +
                self._b * self._sigma * self._time_step_sqrt * xi_over_sqrt_masses
            ), self._cur_velocities
            cur_step += 1
        ensemble.state.set_velocities(self._cur_velocities)