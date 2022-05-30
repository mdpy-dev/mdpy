#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : velocity_verlet_integrator.py
created time : 2022/05/30
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import cupy as cp
from mdpy.core import Ensemble
from mdpy.integrator import Integrator
from mdpy.environment import *


class VelocityVerletIntegrator(Integrator):
    def __init__(self, time_step, update_tile_list_frequency) -> None:
        super().__init__(time_step, update_tile_list_frequency)
        self._time_step_square = self._time_step**2

    def integrate(self, ensemble: Ensemble, num_steps: int = 1):
        acceleration_factor = (0.5 / ensemble.topology.device_masses).astype(CUPY_FLOAT)
        cur_step = 0
        cur_positions = ensemble.state.positions
        cur_velocities = ensemble.state.velocities
        cur_half_accelerations = ensemble.forces * acceleration_factor
        while cur_step < num_steps:
            # Integrate
            next_positions = (
                cur_positions
                + cur_velocities * self._time_step
                + cur_half_accelerations * self._time_step_square
            )
            ensemble.state.set_positions(next_positions)
            if cur_step % self._update_tile_list_frequency == 0:
                ensemble.update_tile_list()
            ensemble.update_constraints()
            next_half_accelerations = ensemble.forces * acceleration_factor
            next_velocities = (
                cur_velocities
                + (cur_half_accelerations + next_half_accelerations) * self._time_step
            )
            # Assign data
            cur_positions = ensemble.state.positions
            cur_velocities = next_velocities
            cur_half_accelerations = next_half_accelerations
            cur_step += 1
        ensemble.state.set_velocities(cur_velocities)
