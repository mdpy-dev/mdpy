#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : simulation.py
created time : 2021/09/28
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np
from .ensemble import Ensemble
from .integrator import Integrator
from .error import *

class Simulation:
    def __init__(self, ensemble: Ensemble, integrator: Integrator) -> None:
        self._ensemble = ensemble
        self._integrator = integrator
        self._cur_step = 0
        self._dumpers = []
        self._minimum_dump_frequency = 0
        self._num_dumpers = 0

    def add_dumpers(self, *dumpers):
        if self._num_dumpers == 0:
            self._minimum_dump_frequency = dumpers[0].dump_frequency
        for dumper in dumpers:
            self._dumpers.append(dumper)
            self._num_dumpers += 1
            self._minimum_dump_frequency = np.gcd(
                self._minimum_dump_frequency, dumper.dump_frequency
            )

    def _dump(self):
        for dumper in self._dumpers:
            if self._cur_step % dumper.dump_frequency == 0:
                dumper.dump(self)

    def reset_simulation_step(self):
        self._cur_step = 0

    def sample(self, num_steps: int):
        if self._num_dumpers == 0 or self._minimum_dump_frequency == 0:
            raise DumperPoorDefinedError(
                'No dumper has been added to Simulation yet.'
            )
        target_step = self._cur_step + num_steps
        self._dump()
        while self._cur_step < target_step:
            self._integrator.sample(self._ensemble, self._minimum_dump_frequency)
            self._dump()
            self._cur_step += self._minimum_dump_frequency

    @property
    def ensemble(self):
        return self._ensemble

    @property
    def integrator(self):
        return self._integrator

    @property
    def dumpers(self):
        return self._dumpers

    @property
    def num_dumpers(self):
        return self._num_dumpers

    @property
    def cur_step(self):
        return self._cur_step