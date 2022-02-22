#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : coordination_number_analyser.py
created time : 2022/02/22
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np
from . import AnalyserResult
from ..core import Trajectory
from ..utils import check_quantity_value, unwrap_vec
from ..utils import select, check_topological_selection_condition, parse_selection_condition
from ..unit import *
from ..error import *

class CoordinationNumberAnalyser:
    def __init__(
        self, 
        selection_condition_1: list[dict], 
        selection_condition_2: list[dict],
        cutoff_radius, num_bins: int,
    ) -> None:
        check_topological_selection_condition(selection_condition_1)
        check_topological_selection_condition(selection_condition_2)
        self._selection_condition_1 = selection_condition_1
        self._selection_condition_2 = selection_condition_2
        self._cutoff_radius = check_quantity_value(cutoff_radius, default_length_unit)
        if not isinstance(num_bins, int):
            raise TypeError('num_bins should be integer, while %s is provided' %type(num_bins))
        self._num_bins = num_bins

    def analysis(self, trajectory: Trajectory, is_dimensionless=False) -> AnalyserResult:
        # Extract positions
        # Topological selection for Trajectory will return a list with same list
        selected_matrix_ids_1 = select(trajectory, self._selection_condition_1)[0] 
        selected_matrix_ids_2 = select(trajectory, self._selection_condition_2)[0]
        # Analysis
        coordination_number, bin_edge = [], []
        for id1 in selected_matrix_ids_1:
            cur_coordination_number = []
            for frame in range(trajectory.num_frames):
                vec = unwrap_vec(
                    trajectory.positions[frame, id1, :] - 
                    trajectory.positions[frame, selected_matrix_ids_2, :],
                    trajectory.pbc_matrix, trajectory.pbc_inv
                )
                dist = np.sqrt((vec**2).sum(1))
                hist, bin_edge = np.histogram(dist, self._num_bins, [0, self._cutoff_radius])
                cur_coordination_number.append(np.cumsum(hist))
            coordination_number.append(np.vstack(cur_coordination_number).mean(0))
            bin_edge = bin_edge
        coordination_number = np.vstack(coordination_number) # num_particle_id1 x num_bins
        mean_coordination_number = coordination_number.mean(0) # Vector num_bins,
        std_coordination_number = coordination_number.std(0) # Vector num_bins,
        mean_coordination_number[0] = mean_coordination_number[1] # Prevent counting self
        std_coordination_number[0] = std_coordination_number[1]
        # Output
        if not is_dimensionless:
            cutoff_radius = Quantity(self._cutoff_radius, default_length_unit)
            bin_edge = Quantity(bin_edge, default_length_unit)
        title = 'Coordination number function between %s --- %s' %(
            parse_selection_condition(self._selection_condition_1),
            parse_selection_condition(self._selection_condition_2)
        )
        description = {
            'mean': 'The mean value of coordination number function, unit: dimesionless',
            'std': 'The std value of coordination number function, unit: dimensionless',
            'cutoff_radius': 'The cutoff radius of coordination number function, unit: default_length_unit',
            'num_bins': 'The number of bins used to construct coordination number curve, unit: dimensionless',
            'bin_edge': 'The bin edge of coordination number function, unit: dimensionless'
        }
        data = {
            'mean': mean_coordination_number, 'std': std_coordination_number, 'cutoff_radius': cutoff_radius,
            'num_bins': self._num_bins, 'bin_edge': bin_edge
        }
        return AnalyserResult(title=title, description=description, data=data)

    @property
    def selection_condition_1(self):
        return self._selection_condition_1

    @selection_condition_1.setter
    def selection_condition_1(self, selection_condition: list[dict]):
        check_topological_selection_condition(selection_condition)
        self._selection_condition_1 = selection_condition

    @property
    def selection_condition_2(self):
        return self._selection_condition_2

    @selection_condition_2.setter
    def selection_condition_2(self, selection_condition: list[dict]):
        check_topological_selection_condition(selection_condition)
        self._selection_condition_2 = selection_condition

    @property
    def cutoff_radius(self):
        return self._cutoff_radius

    @cutoff_radius.setter
    def cutoff_radius(self, cutoff_radius):
        self._cutoff_radius = check_quantity_value(cutoff_radius, default_length_unit)

    @property
    def num_bins(self):
        return self._num_bins

    @num_bins.setter
    def num_bins(self, num_bins: int):
        if not isinstance(num_bins, int):
            raise TypeError('num_bins should be integer, while %s is provided' %type(num_bins))
        self._num_bins = num_bins