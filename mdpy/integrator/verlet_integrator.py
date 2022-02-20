#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : verlet_integrator.py
created time : 2021/10/18
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

from . import Integrator
from .. import env
from ..ensemble import Ensemble
from ..utils import *

class VerletIntegrator(Integrator):
    def __init__(self, time_step) -> None:
        super().__init__(time_step)
        self._time_step_square = self._time_step**2

    def integrate(self, ensemble: Ensemble, num_steps: int=1):
        # Setting variables
        cur_step = 0
        masses = ensemble.topology.masses
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
            self._cur_positions = wrap_positions(self._cur_positions, *ensemble.state.pbc_info)
            ensemble.state.set_positions(self._cur_positions)
            # Update step
            cur_step += 1
        ensemble.state.set_velocities(unwrap_vec(
            (self._cur_positions - self._pre_positions).astype(env.NUMPY_FLOAT), 
            *ensemble.state.pbc_info
        ) / 2 / self._time_step)
