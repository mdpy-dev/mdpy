#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : verlet_integrator.py
created time : 2021/10/18
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

from mdpy.core import Ensemble
from mdpy.integrator import Integrator
from mdpy.utils import *

class VerletIntegrator(Integrator):
    def __init__(self, time_step, neighbor_list_update_freq=10) -> None:
        super().__init__(time_step, neighbor_list_update_freq)
        self._time_step_square = self._time_step**2

    def integrate(self, ensemble: Ensemble, num_steps: int=1):
        # Setting variables
        cur_step = 0
        masses = ensemble.topology.device_masses
        # Update force
        ensemble.update()
        accelration = ensemble.forces / masses
        # Initialization
        if self.is_cached == False:
            velocities = ensemble.state.velocities
            self._cur_positions = ensemble.state.positions
            self._pre_positions = (
                self._cur_positions - velocities * self._time_step +
                accelration * self._time_step_square
            )
        while cur_step < num_steps:
            if cur_step != 0:
                ensemble.update()
                accelration = ensemble.forces / masses
            # Update positions and velocities
            self._cur_positions, self._pre_positions = (
                2 * self._cur_positions - self._pre_positions +
                accelration * self._time_step_square
            ), self._cur_positions
            if cur_step % self._neighbor_list_update_freq == 0:
                ensemble.state.set_positions(self._cur_positions, True)
            else:
                ensemble.state.set_positions(self._cur_positions, False)
            # Update step
            cur_step += 1
        ensemble.state.set_velocities(unwrap_vec(
            self._cur_positions - self._pre_positions, ensemble.state.device_pbc_diag
        ) / 2 / self._time_step)
