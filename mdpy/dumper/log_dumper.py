#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : log_dumper.py
created time : 2021/10/21
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np
from datetime import datetime, timedelta
from mdpy.error import DumperPoorDefinedError
from .dumper import Dumper
from ..simulation import Simulation
from ..unit import *
from ..dumper import *
from mdpy import dumper

class LogDumper(Dumper):
    def __init__(
        self, file_path: str, dump_frequency: int,
        step: bool=False,
        sim_time: bool=False,
        sim_speed: bool=False,
        potential_energy: bool=False,
        kinetic_energy: bool=False,
        total_energy: bool=False,
        temperature: bool=False, 
        pressure: bool=False,
        volume: bool=False,
        density: bool=False, 
        cpu_time: bool=False,
        rest_time: bool=False,
        total_step: int=0,
        seperator: str='\t'
    ) -> None:
        super().__init__(file_path, dump_frequency)
        # Input
        self._step = step
        self._sim_time = sim_time
        self._potential_energy = potential_energy
        self._kinetic_energy = kinetic_energy
        self._total_energy = total_energy
        self._temperature = temperature
        self._pressure = pressure
        self._volume = volume
        self._density = density
        self._cpu_time = cpu_time
        self._rest_time = rest_time
        self._sim_speed = sim_speed
        self._total_step = total_step
        self._seperator = seperator
        if self._total_step == 0 and self._rest_time == True:
            raise DumperPoorDefinedError(
                'mdpy.core.LogDumper cannot provide rest_time without specifying total_step'
            )
        # Variables
        self._is_integrate_start = False
        self._initial_time = self._now()
        self._cur_dump_time = self._now()
        self._pre_dump_time = self._now()

    @staticmethod
    def _now():
        return datetime.now().replace(microsecond=0)

    def dump(self, simulation: Simulation):
        if self._num_dumpped_frames == 0:
            self._dump_header()
        if simulation.cur_step != 0 and self._is_integrate_start == False:
            self._initial_time = self._now()
            self._is_integrate_start = True
        self._dump_log(simulation)
        self._num_dumpped_frames += 1

    def _dump_header(self):
        header = ''
        if self._step:
            header += 'Step' + self._seperator
        if self._sim_time:
            header += 'Time (ns)' + self._seperator
        if self._potential_energy:
            header += 'E_p (kj/mol)' + self._seperator
        if self._kinetic_energy:
            header += 'E_k (kj/mol)' + self._seperator
        if self._total_energy:
            header += 'E_t (kj/mol)' + self._seperator
        if self._temperature:
            header += 'Temperature (K)' + self._seperator
        if self._volume:
            header += 'Volume (nm^3)' + self._seperator
        if self._pressure:
            header += 'Pressure (atm)' + self._seperator
        if self._density:
            header += 'Density (1e3*kg/m^3)' + self._seperator
        if self._cpu_time:
            header += 'CPU Time' + self._seperator
        if self._sim_speed:
            header += 'Speed (ns/day)' + self._seperator
        if self._rest_time:
            header += 'Rest Time' + self._seperator
        if header != '':
            header += '\n'
        self._dump_info(header)

    def _dump_log(self, simulation: Simulation):
        self._cur_dump_time = self._now()
        self._cpu_time = (self._cur_dump_time - self._initial_time)
        dump_info = ''
        if self._step:
            dump_info += self._get_step(simulation) + self._seperator
        if self._sim_time:
            dump_info += self._get_sim_time(simulation) + self._seperator
        if self._potential_energy:
            dump_info += self._get_potential_energy(simulation) + self._seperator
        if self._kinetic_energy:
            dump_info += self._get_kinetic_energy(simulation) + self._seperator
        if self._total_energy:
            dump_info += self._get_total_energy(simulation) + self._seperator
        if self._temperature:
            dump_info += self._get_temperature(simulation) + self._seperator
        if self._volume:
            dump_info += self._get_volume(simulation) + self._seperator
        if self._pressure:
            pass
        if self._density:
            dump_info += self._get_density(simulation) + self._seperator
        if self._cpu_time:
            dump_info += self._get_cpu_time(simulation) + self._seperator
        if self._sim_speed:
            dump_info += self._get_sim_speed(simulation) + self._seperator
        if self._rest_time:
            dump_info += self._get_rest_time(simulation) + self._seperator
        if dump_info != '':
            dump_info += '\n'
        self._dump_info(dump_info)

    def _get_step(self, simulation: Simulation):
        return '%d' %(simulation.cur_step)

    def _get_sim_time(self, simulation: Simulation):
        sim_time = simulation.cur_step * simulation.integrator.time_step
        sim_time = Quantity(sim_time, default_time_unit).convert_to(nanosecond).value
        return '%.2e' %sim_time

    def _get_potential_energy(self, simulation: Simulation):
        return '%.3f' %Quantity(
            simulation.ensemble.potential_energy, default_energy_unit
        ).convert_to(kilojoule_permol).value

    def _get_kinetic_energy(self, simulation: Simulation):
        return '%.3f' %Quantity(
            simulation.ensemble.kinetic_energy, default_energy_unit
        ).convert_to(kilojoule_permol).value

    def _get_total_energy(self, simulation: Simulation):
        return '%.3f' %Quantity(
            simulation.ensemble.total_energy, default_energy_unit
        ).convert_to(kilojoule_permol).value

    def _get_temperature(self, simulation: Simulation):
        kinetic_energy = Quantity(simulation.ensemble.kinetic_energy, default_energy_unit)
        num_particles = simulation.ensemble.topology.num_particles
        temperature = kinetic_energy * Quantity(2 / 3 / num_particles) / KB
        return '%.2f' %temperature.convert_to(default_temperature_unit).value

    def _get_volume(self, simulation: Simulation):
        pbc_matrix = simulation.ensemble.state.pbc_matrix
        volume = np.cross(pbc_matrix[0, :], pbc_matrix[1, :])
        volume = np.dot(volume, pbc_matrix[2, :])
        volume = Quantity(
            np.abs(volume),
            default_length_unit**3
        ).convert_to(nanometer**3).value
        return '%.3f' %volume

    def _get_density(self, simulation: Simulation):
        pbc_matrix = simulation.ensemble.state.pbc_matrix
        volume = np.cross(pbc_matrix[0, :], pbc_matrix[1, :])
        volume = np.abs(np.dot(volume, pbc_matrix[2, :]))
        mass = simulation.ensemble.topology.masses.sum()
        density = Quantity(
            mass/volume,
            default_mass_unit/default_length_unit**3
        ).convert_to(kilogram/meter**3).value
        return '%.3f' %(density/1000)

    def _get_cpu_time(self, simulation: Simulation):
        return '%s' %(self._cpu_time)

    def _get_rest_time(self, simulation: Simulation):
        if simulation.cur_step == 0:
            return '--:--:--'
        else:
            unfinished_ratio = 1 - (simulation.cur_step / self._total_step)
            cpu_seconds = self._cpu_time.total_seconds()
            rest_seconds = cpu_seconds / (1 - unfinished_ratio) * unfinished_ratio
            return '%s' %timedelta(seconds=int(rest_seconds))

    def _get_sim_speed(self, simulation: Simulation):
        self._cur_dump_time = self._now()
        cpu_time = self._cpu_time.total_seconds()
        if cpu_time == 0:
            return '--:--:--'
        else:
            cpu_time = Quantity(cpu_time, second).convert_to(day).value
            sim_time = simulation.cur_step * simulation.integrator.time_step
            sim_time = Quantity(sim_time, default_time_unit).convert_to(nanosecond).value
            return '%.4f' %(sim_time / cpu_time)