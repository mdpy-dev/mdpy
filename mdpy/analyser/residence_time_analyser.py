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
        max_coorelation_interval: int,
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

    def analysis(self, trajectory:Trajectory, is_dimensionless=False) -> AnalyserResult:
        # Extract positions
        # Topological selection for Trajectory will return a list with same list
        selected_matrix_ids_1 = select(trajectory, self._selection_condition_1)[0] 
        selection_condition_2 = self._selection_condition_2.copy()
        for condition in selection_condition_2:
            condition['nearby'] = [selected_matrix_ids_1, self._cutoff_radius]
        selected_matrix_ids_2 = select(trajectory, selection_condition_2)
        print(selected_matrix_ids_2[0])
        print(len(selected_matrix_ids_2[0]))
        # Analysis
        neighbor_array, num_neighbors_array = self._analysis_neighbor(
            trajectory, selected_matrix_ids_1, selected_matrix_ids_2
        )

    def _analysis_neighbor(self, trajectory: Trajectory, selected_matrix_ids_1, selected_matrix_ids_2):
        num_particles_1 = len(selected_matrix_ids_1)
        num_particles_2 = len(selected_matrix_ids_2)
        neighbor_array = np.ones([trajectory.num_frames, num_particles_1, self._num_bins, 20], dtype=env.NUMPY_INT)
        num_neighbors_array = np.zeros([trajectory.num_frames, num_particles_1, self._num_bins], dtype=env.NUMPY_INT)
        for frame in range(trajectory.num_frames):
            # print('frame')
            for index_id1, id1 in enumerate(selected_matrix_ids_1):
                vec = unwrap_vec(
                    trajectory.positions[frame, id1, :] - 
                    trajectory.positions[frame, selected_matrix_ids_2[frame], :],
                    trajectory.pbc_matrix, trajectory.pbc_inv
                )
                dist = np.sqrt((vec**2).sum(1))
                particle_affiliation = (dist // self._bin_width).astype(env.NUMPY_INT)
                for index_id2, affiliation in enumerate(particle_affiliation):
                    if affiliation < self._num_bins:
                        neighbor_array[
                            frame, index_id1, affiliation, 
                            num_neighbors_array[frame, index_id1, affiliation]
                        ] = selected_matrix_ids_2[frame][index_id2]
                        num_neighbors_array[frame, index_id1, affiliation] += 1
        return neighbor_array, num_neighbors_array

    @staticmethod
    def cuda_kernel(
        positions, selected_matrix_ids_1, selected_matrix_ids_2, 
        bin_width
    ):
        pass

    def _analysis_correlation_function(self, neighbor_array, num_neighbors_array):
        pass

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