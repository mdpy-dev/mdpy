#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : log_writer.py
created time : 2022/04/06
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np
from datetime import datetime, timedelta
from mdpy.core import Ensemble
from mdpy.unit import *
from mdpy.utils import *
from mdpy.error import *

class LogWriter:
    def __init__(
        self, file_path: str,
        time_step,
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
        # Input
        if not file_path.endswith('.log'):
            raise FileFormatError('The file should end with .log suffix')
        self._file_path = file_path
        self._time_step = check_quantity_value(time_step, default_time_unit)
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
        # Refresh file
        open(file_path, 'w').close()
        # Variables
        self._initial_time = self._now()
        self._cur_write_time = self._now()
        self._pre_write_time = self._now()
        # Dump head
        self._write_header()

    @staticmethod
    def _now():
        return datetime.now().replace(microsecond=0)

    def _write_info(self, info):
        with open(self._file_path, 'a') as f:
            print(info, file=f, end='')

    def write(self, ensemble: Ensemble, cur_step: int):
        self._cur_write_time = self._now()
        self._cpu_time = (self._cur_write_time - self._initial_time)
        write_info = ''
        if self._step:
            write_info += self._get_step(cur_step) + self._seperator
        if self._sim_time:
            write_info += self._get_sim_time(cur_step) + self._seperator
        if self._potential_energy:
            write_info += self._get_potential_energy(ensemble) + self._seperator
        if self._kinetic_energy:
            write_info += self._get_kinetic_energy(ensemble) + self._seperator
        if self._total_energy:
            write_info += self._get_total_energy(ensemble) + self._seperator
        if self._temperature:
            write_info += self._get_temperature(ensemble) + self._seperator
        if self._volume:
            write_info += self._get_volume(ensemble) + self._seperator
        if self._pressure:
            pass
        if self._density:
            write_info += self._get_density(ensemble) + self._seperator
        if self._cpu_time:
            write_info += self._get_cpu_time() + self._seperator
        if self._sim_speed:
            write_info += self._get_sim_speed(cur_step) + self._seperator
        if self._rest_time:
            write_info += self._get_rest_time(cur_step) + self._seperator
        if write_info != '':
            write_info += '\n'
        self._write_info(write_info)

    def _write_header(self):
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
        self._write_info(header)

    def _get_step(self, cur_step: int):
        return '%d' %(cur_step)

    def _get_sim_time(self, cur_step: int):
        sim_time = cur_step * self._time_step
        sim_time = Quantity(sim_time, default_time_unit).convert_to(nanosecond).value
        return '%.2e' %sim_time

    def _get_potential_energy(self, ensemble: Ensemble):
        return '%.3f' %Quantity(
            ensemble.potential_energy, default_energy_unit
        ).convert_to(kilojoule_permol).value

    def _get_kinetic_energy(self, ensemble: Ensemble):
        return '%.3f' %Quantity(
            ensemble.kinetic_energy, default_energy_unit
        ).convert_to(kilojoule_permol).value

    def _get_total_energy(self, ensemble: Ensemble):
        return '%.3f' %Quantity(
            ensemble.total_energy, default_energy_unit
        ).convert_to(kilojoule_permol).value

    def _get_temperature(self, ensemble: Ensemble):
        kinetic_energy = Quantity(ensemble.kinetic_energy, default_energy_unit)
        num_particles = ensemble.topology.num_particles
        temperature = kinetic_energy * Quantity(2 / 3 / num_particles) / KB
        return '%.2f' %temperature.convert_to(default_temperature_unit).value

    def _get_volume(self, ensemble: Ensemble):
        pbc_matrix = ensemble.state.pbc_matrix
        volume = np.cross(pbc_matrix[0, :], pbc_matrix[1, :])
        volume = np.dot(volume, pbc_matrix[2, :])
        volume = Quantity(
            np.abs(volume),
            default_length_unit**3
        ).convert_to(nanometer**3).value
        return '%.3f' %volume

    def _get_density(self, ensemble: Ensemble):
        pbc_matrix = ensemble.state.pbc_matrix
        volume = np.cross(pbc_matrix[0, :], pbc_matrix[1, :])
        volume = np.abs(np.dot(volume, pbc_matrix[2, :]))
        mass = ensemble.topology.masses.sum()
        density = Quantity(
            mass/volume,
            default_mass_unit/default_length_unit**3
        ).convert_to(kilogram/meter**3).value
        return '%.3f' %(density/1000)

    def _get_cpu_time(self):
        return '%s' %(self._cpu_time)

    def _get_rest_time(self, cur_step: int):
        if cur_step == 0:
            return '--:--:--'
        else:
            unfinished_ratio = 1 - (cur_step / self._total_step)
            cpu_seconds = self._cpu_time.total_seconds()
            rest_seconds = cpu_seconds / (1 - unfinished_ratio) * unfinished_ratio
            return '%s' %timedelta(seconds=int(rest_seconds))

    def _get_sim_speed(self, cur_step: int):
        self._cur_write_time = self._now()
        cpu_time = self._cpu_time.total_seconds()
        if cpu_time == 0:
            return '--:--:--'
        else:
            cpu_time = Quantity(cpu_time, second).convert_to(day).value
            sim_time = cur_step * self._time_step
            sim_time = Quantity(sim_time, default_time_unit).convert_to(nanosecond).value
            return '%.4f' %(sim_time / cpu_time)