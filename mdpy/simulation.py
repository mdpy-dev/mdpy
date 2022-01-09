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

from mdpy.math.pbc import wrap_positions
from .ensemble import Ensemble
from .integrator import Integrator
from .error import *
from .unit import *


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

    def integrate(self, num_steps: int):
        if self._num_dumpers == 0 or self._minimum_dump_frequency == 0:
            raise DumperPoorDefinedError(
                'No dumper has been added to Simulation yet.'
            )
        target_step = self._cur_step + num_steps
        self._dump()
        while self._cur_step < target_step:
            self._integrator.integrate(
                self._ensemble, self._minimum_dump_frequency)
            self._cur_step += self._minimum_dump_frequency
            self._dump()

    def minimize_energy(self, alpha, energy_tolerance=1e-6, max_iterations=1000):
        energy_convert_unit = default_energy_unit / kilojoule_permol
        self.ensemble.update()
        print('Start energy minimization:')
        print(
            'Initial potential energy: %.5f kj/mol'
            % Quantity(self._ensemble.potential_energy, energy_convert_unit).value
        )
        self._dump()
        cur_iteration = 0
        cur_energy, pre_energy = self._ensemble.potential_energy, self._ensemble.potential_energy
        while cur_iteration < max_iterations:
            self.ensemble.state.set_positions(wrap_positions(
                self._ensemble.state.positions + alpha * self._ensemble.forces,
                *self._ensemble.state.pbc_info
            ))
            self._ensemble.update()
            cur_energy = self._ensemble.potential_energy
            energy_error = np.abs((cur_energy - pre_energy) / pre_energy)
            if energy_error < energy_tolerance:
                print('Penultimate potential energy: %.5f kj/mol' %
                      (Quantity(pre_energy, energy_convert_unit).value))
                print('Final potential energy: %.5f kj/mol' %
                      (Quantity(cur_energy, energy_convert_unit).value))
                print('Energy error: %s < %e' %
                      (energy_error, energy_tolerance))
                self._dump()
                return None
            cur_iteration += 1
        print('Final potential energy: %.5f kj/mol' %
              (Quantity(cur_energy, energy_convert_unit).value))
        self._dump()

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
