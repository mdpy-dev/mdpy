#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : mobility_analyser.py
created time : 2022/02/20
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import numpy as np
from mdpy import SPATIAL_DIM
from mdpy.analyser import AnalyserResult
from mdpy.core import Trajectory
from mdpy.utils import check_quantity_value
from mdpy.utils import select, check_topological_selection_condition, parse_selection_condition
from mdpy.unit import *
from mdpy.error import *

class MobilityAnalyser:
    def __init__(
        self, selection_condition: list[dict],
        electric_intensity, drift_velocity_interval
    ) -> None:
        check_topological_selection_condition(selection_condition)
        self._selection_condition = selection_condition
        electric_intensity = check_quantity_value(electric_intensity, default_electric_intensity_unit)
        if electric_intensity.size != SPATIAL_DIM:
            raise ArrayDimError(
                'electric_intensity should be a Quantity or ndarray with 3 element for electric intensity on 3 dimension'
            )
        self._electric_intensity = electric_intensity.reshape([3]) # vector 3
        self._drift_velocity_interval = drift_velocity_interval

    def analysis(self, trajectory: Trajectory, is_dimensionless=True) -> AnalyserResult:
        # Read input
        trajectory.unwrap_positions()
        selected_matrix_ids = select(trajectory, self._selection_condition)[0] # Topological selection for Trajectory will return a list with same list
        # Extract position
        positions = trajectory.unwrapped_position[:, selected_matrix_ids, :]
        # Analysis drift velocities
        drift_velocities = (
            positions[self._drift_velocity_interval:, :, :] -
            positions[:-self._drift_velocity_interval, :, :]
        ) / (trajectory.time_step * self._drift_velocity_interval) # (num_frame - interval) x num_selected_particle x 3
        drift_velocities = drift_velocities.mean(0) # num_selected_particle x 3
        mean_drift_velocities = drift_velocities.mean(0) # vector 3
        std_drift_velocities = drift_velocities.std(0) # vector 3
        # Analysis mobility
        mean_mobility, std_mobility = np.zeros([SPATIAL_DIM]), np.zeros([SPATIAL_DIM])
        for i in range(SPATIAL_DIM):
            if self._electric_intensity[i] == 0:
                continue
            else:
                mean_mobility[i] = mean_drift_velocities[i] / self._electric_intensity[i]
                std_mobility[i] = std_drift_velocities[i] / self._electric_intensity[i]
        # Output
        time_step = trajectory.time_step * self._drift_velocity_interval
        electric_intensity = self._electric_intensity
        if not is_dimensionless:
            unit = default_velocity_unit / default_electric_intensity_unit
            mean_mobility = Quantity(mean_mobility, unit)
            std_mobility = Quantity(std_mobility, unit)
            time_step = Quantity(time_step, default_time_unit)
            electric_intensity = Quantity(electric_intensity, default_electric_intensity_unit)
        title =  'Mobility of %s' %parse_selection_condition(self._selection_condition)
        description = {
            'mean': 'The mean of mobility result, unit: default_velocities_unit / default_electric_intensity_unit',
            'std': 'The std of mobility result, unit: default_velocities_unit / default_electric_intensity_unit',
            'time_step': 'Time step used to calculate the drift velocities, unit: default_time_unit',
            'electric_intensity': 'Electric intensity used to calculate the mobility, unit: default_electric_intensity_unit'
        }
        data = {'mean': mean_mobility, 'std': std_mobility, 'time_step': time_step, 'electric_intensity': electric_intensity}
        return AnalyserResult(title=title, description=description, data=data)

    @property
    def selection_condition(self):
        return self._selection_condition

    @selection_condition.setter
    def selection_condition(self, selection_condition):
        check_topological_selection_condition(selection_condition)
        self._selection_condition = selection_condition

    @property
    def electirc_intensity(self):
        return self._electric_intensity

    @electirc_intensity.setter
    def electric_intensity(self, electric_intensity):
        electric_intensity = check_quantity_value(electric_intensity, default_electric_intensity_unit)
        if electric_intensity.size != SPATIAL_DIM:
            raise ArrayDimError(
                'electric_intensity should be a Quantity or ndarray with 3 element for electric intensity on 3 dimension'
            )
        self._electric_intensity = electric_intensity