#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : diffusion_analyser.py
created time : 2022/02/20
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np
from scipy.optimize import curve_fit
from . import AnalyserResult
from .. import SPATIAL_DIM
from ..core import Trajectory
from ..utils import select, check_topological_selection_condition, parse_selection_condition
from ..unit import *
from ..error import *

class DiffusionAnalyser:
    def __init__(
        self, selection_condition: list[dict],
        max_msd_interval: int, 
    ) -> None:
        check_topological_selection_condition(selection_condition)
        self._selection_condition = selection_condition
        if not isinstance(max_msd_interval, int):
            raise TypeError('max_msd_interval should be integer, while %s is provided' %type(max_msd_interval))
        self._max_msd_interval = max_msd_interval
        self._msd_interval_series = np.arange(1, self._max_msd_interval + 1)

    def analysis(self, trajectory: Trajectory, is_dimensionless=False) -> AnalyserResult:
        # Extract positions
        # Topological selection for Trajectory will return a list with same list
        selected_matrix_ids = select(trajectory, self._selection_condition)[0]
        trajectory.unwrap_positions()
        positions = trajectory.unwrapped_position[:, selected_matrix_ids, :]
        # Analysis MSD
        mean_msd, std_msd = np.ones([self._max_msd_interval, SPATIAL_DIM]), np.ones([self._max_msd_interval, SPATIAL_DIM])
        for msd_interval in self._msd_interval_series:
            msd = (positions[msd_interval:, :, :] - positions[:-msd_interval, :, :])**2 # (n_frame - msd_interval) x n_particle x SPATIAL_DIM
            mean_msd[msd_interval-1, :] = msd.mean(axis=(0, 1))
            std_msd[msd_interval-1, :] = msd.std(axis=(0, 1))
        # Analysis diffusion coefficient
        mean_diffusion_coefficient = np.zeros([SPATIAL_DIM])
        std_diffusion_coefficient = np.zeros([SPATIAL_DIM])
        def fun(x, a, b):
            return a*x + b
        for i in range(SPATIAL_DIM):
            para, pcov = curve_fit(
                fun, self._msd_interval_series*trajectory.time_step, mean_msd[:, i]
            )
            mean_diffusion_coefficient[i] = para[0]
            std_diffusion_coefficient[i] = np.sqrt(pcov[0, 0])
        # Output
        if not is_dimensionless:
            mean_msd = Quantity(mean_msd, default_length_unit**2)
            std_msd = Quantity(std_msd, default_length_unit**2)
            mean_diffusion_coefficient = Quantity(mean_diffusion_coefficient, default_length_unit**2/default_time_unit)
            std_diffusion_coefficient = Quantity(std_diffusion_coefficient, default_length_unit**2/default_time_unit)
            msd_interval_series = Quantity(self._msd_interval_series*trajectory.time_step, default_time_unit)
        title = 'Diffusion coefficient of %s' %parse_selection_condition(self._selection_condition)
        description = {
            'mean_msd': 'The mean value of MSD, unit: default_length_unit**2',
            'std_msd': 'The std value of MSD, unit: default_length_unit**2',
            'mean_diffusion_coefficient': 'The mean value of diffusion_coefficient on x, y, z direction fitted from mean_msd, unit: default_length_unit**2/default_time_unit',
            'std_diffusion_coefficient': 'The estimated std of diffusion_coefficient on x, y, z direction, unit: default_length_unit**2/default_time_unit',
            'msd_interval_series': 'The correspoding time interval for each msd result, unit: default_time_unit',
        }
        data = {
            'mean_msd': mean_msd, 'std_msd': std_msd, 
            'mean_diffusion_coefficient': mean_diffusion_coefficient,
            'std_diffusion_coefficient': std_diffusion_coefficient, 
            'msd_interval_series': msd_interval_series
        }
        return AnalyserResult(title=title, description=description, data=data)

    @property
    def selection_condition(self):
        return self._selection_condition

    @selection_condition.setter
    def selection_condition(self, selection_condition: list[dict]):
        check_topological_selection_condition(selection_condition)
        self._selection_condition = selection_condition

    @property
    def max_msd_interval(self):
        return self._max_msd_interval

    @max_msd_interval.setter
    def max_msd_interval(self, max_msd_interval: int):
        if not isinstance(max_msd_interval, int):
            raise TypeError('max_msd_interval should be integer, while %s is provided' %type(max_msd_interval))
        self._msd_interval_series = np.arange(1, self._max_msd_interval + 1)