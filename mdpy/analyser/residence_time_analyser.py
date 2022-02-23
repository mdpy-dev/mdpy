#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : residence_time_analyser.py
created time : 2022/02/22
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np
from . import AnalyserResult
from .. import env
from ..core import Trajectory
from ..utils import check_quantity_value, unwrap_vec
from ..utils import select, check_topological_selection_condition, parse_selection_condition
from ..unit import *
from ..error import *

class ResidenceTimeAnalyser:
    def __init__(
        self, 
        selection_condition_1: list[dict], 
        selection_condition_2: list[dict],
        cutoff_radius, num_bins: int, 
        max_coorelation_time,
    ) -> None:
        check_topological_selection_condition(selection_condition_1)
        check_topological_selection_condition(selection_condition_2)
        self._selection_condition_1 = selection_condition_1
        self._selection_condition_2 = selection_condition_2
        self._cutoff_radius = check_quantity_value(cutoff_radius, default_length_unit)
        if not isinstance(num_bins, int):
            raise TypeError('num_bins should be integer, while %s is provided' %type(num_bins))
        self._num_bins = num_bins 
        self._bin_edge = np.linspace(0, self._cutoff_radius, self._num_bins + 1)
        self._bin_width = self._bin_edge[1] - self._bin_edge[0]
        self._max_coorelation_time = check_quantity_value(max_coorelation_time, default_time_unit)

    def analysis(self, trajectory:Trajectory, is_dimensionless=True) -> AnalyserResult:
        # Extract positions
        # Topological selection for Trajectory will return a list with same list
        selected_matrix_ids_1 = select(trajectory, self._selection_condition_1)[0]
        num_particles_1 = len(selected_matrix_ids_1) 
        selected_matrix_ids_2 = select(trajectory, self._selection_condition_2)[0]
        num_particles_2 = len(selected_matrix_ids_2)
        # Analysis neighbor of first frame
        neighbors = [[]] * num_particles_1
        neighbor_affiliations = [[]] * num_particles_1
        for index, id1 in enumerate(selected_matrix_ids_1):
            vec = unwrap_vec(
                trajectory.positions[0, id1, :] - 
                trajectory.positions[0, selected_matrix_ids_2, :],
                trajectory.pbc_matrix, trajectory.pbc_inv
            )
            dist = np.sqrt((vec**2).sum(1))
            neighbor_index = np.where(dist <= self._cutoff_radius)[0]
            neighbors[index].extend(
                [selected_matrix_ids_2[i] for i in neighbor_index]
            )
            neighbor_affiliations[index].extend(
                list((dist[neighbor_index] // self._bin_width).astype(env.NUMPY_INT))
            ) 
        # Analysis time coorelation function
        last_time_per_bin = [[[] for i in range(self._num_bins)] for j in range(num_particles_1)]
        for index1, id1 in enumerate(selected_matrix_ids_1):
            for index2, id2 in enumerate(neighbors[index1]):
                affiliation = neighbor_affiliations[index1][index2]
                for frame in range(1, trajectory.num_frames):
                    vec = unwrap_vec(
                        trajectory.positions[frame, id1, :] - 
                        trajectory.positions[frame, id2, :],
                        trajectory.pbc_matrix, trajectory.pbc_inv
                    )
                    dist = np.sqrt((vec**2).sum())
                    if dist < self._bin_edge[affiliation] or dist > self._bin_edge[affiliation+1]:
                        last_time_per_bin[index1][affiliation].append(frame-1)
                        break

        num_time_intervals = int(np.ceil(self._max_coorelation_time / trajectory.time_step))
        time_series = np.linspace(1, num_time_intervals+1) * trajectory.time_step
        mean_time_coorelation = np.zeros([self._num_bins, num_time_intervals])
        std_time_correlation = np.zeros([self._num_bins, num_time_intervals])
        for bin_id in range(self._num_bins):
            cur_time_coorelation = np.zeros([num_particles_1, num_time_intervals])
            for id1 in range(num_particles_1):
                cur_time_coorelation[id1, :] = self._time_coorelation_fun(last_time_per_bin[id1][bin_id], num_time_intervals)
            mean_time_coorelation[bin_id, :] = cur_time_coorelation.mean(0)
            std_time_correlation[bin_id, :] = cur_time_coorelation.std(0)
        residence_time = mean_time_coorelation.sum(1) * trajectory.time_step
        # Output
        cutoff_radius = self._cutoff_radius
        bin_edge = self._bin_edge
        if not is_dimensionless:
            time_series = Quantity(time_series, default_time_unit)
            residence_time = Quantity(residence_time, default_time_unit)
            cutoff_radius = Quantity(cutoff_radius, default_length_unit)
            bin_edge = Quantity(bin_edge, default_length_unit)
        title = 'Residence time between %s --- %s' %(
            parse_selection_condition(self._selection_condition_1),
            parse_selection_condition(self._selection_condition_2)
        )
        description = {
            'mean_time_coorelation': 'The mean value of time coorelation function verse time_interval of each bin, unit: dimesionless',
            'std_time_coorelation': 'The std value of time coorelation function verse time_series of each bin, unit: dimensionless',
            'time_series': 'The time serires of time coorelation function, unit: default_time_unit',
            'residence_time': 'The residence time tau verse bin_edge, unit: default_time_unit',
            'cutoff_radius': 'The cutoff radius of residence time function, unit: default_length_unit',
            'num_bins': 'The number of bins used to construct RDF curve, unit: dimensionless',
            'bin_edge': 'The bin edge of residence time function, unit: default_length_unit'
        }
        data = {
            'mean_time_coorelation': mean_time_coorelation,
            'std_time_coorelation': std_time_correlation,
            'time_series': time_series,
            'residence_time': residence_time,
            'cutoff_radius': cutoff_radius,
            'num_bins': self._num_bins,
            'bin_edge': bin_edge
        }
        return AnalyserResult(title=title, description=description, data=data)

    def _time_coorelation_fun(self, last_time: list, num_time_intervals):
        res = np.zeros([num_time_intervals])
        num_particles, left_particles = len(last_time), len(last_time)
        if num_particles == 0:
            return res
        res[0] = 1
        max_last_time_interval = np.max(last_time)
        for i in range(1, max_last_time_interval):
            try:
                left_particles -= last_time.index(i)
            except:
                break
            res[i] = left_particles / num_particles
        return res


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
        self._bin_edge = np.linspace(0, self._cutoff_radius, self._num_bins + 1)
        self._bin_width = self._bin_edge[1] - self._bin_edge[0]

    @property
    def max_coorelation_time(self):
        return self._max_coorelation_time

    @max_coorelation_time.setter
    def max_coorelation_time(self, max_coorelation_time):
        self._max_coorelation_time = check_quantity_value(max_coorelation_time, default_time_unit)