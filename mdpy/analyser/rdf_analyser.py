#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : rdf_analyser.py
created time : 2022/02/20
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

class RDFAnalyser:
    def __init__(
        self, 
        selection_condition_1: list[dict], 
        selection_condition_2: list[dict],
        cutoff_radius, num_bins: int
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
        hist, bin_edge = [], []
        for id1 in selected_matrix_ids_1:
            hist_id1 = []
            for frame in range(trajectory.num_frames):
                vec = unwrap_vec(
                    trajectory.positions[frame, id1, :] - 
                    trajectory.positions[frame, selected_matrix_ids_2, :],
                    trajectory.pbc_matrix, trajectory.pbc_inv
                )
                dist = np.sqrt((vec**2).sum(1))
                cur_hist_id1, bin_edge = np.histogram(dist, self._num_bins, [0, self._cutoff_radius])
                hist_id1.append(cur_hist_id1)
            hist.append(np.vstack(hist_id1).mean(0))
            bin_edge = bin_edge
        bin_width = bin_edge[1] - bin_edge[0]
        hist = np.vstack(hist) / (4 * np.pi * bin_edge[1:]**2 * bin_width)
        mean_hist = hist.mean(0)
        std_hist = hist.std(0)
        factor = mean_hist.mean()
        mean_hist /= factor
        std_hist /= factor
        mean_hist[0], std_hist[0] = mean_hist[1], std_hist[1] # Prevent counting self in RDF
        # Output
        if not is_dimensionless:
            cutoff_radius = Quantity(self._cutoff_radius, default_length_unit)
            bin_edge = Quantity(bin_edge, default_length_unit)
        title = 'RDF between %s --- %s' %(
            parse_selection_condition(self._selection_condition_1),
            parse_selection_condition(self._selection_condition_2)
        )
        description = {
            'mean': 'The mean value of RDF function, unit: dimesionless',
            'std': 'The std value of RDF function, unit: dimensionless',
            'cutoff_radius': 'The cutoff radius of RDF function, unit: default_length_unit',
            'num_bins': 'The number of bins used to construct RDF curve, unit: dimensionless',
            'bin_edge': 'The bin edge of RDF function, unit: dimensionless'
        }
        data = {
            'mean': mean_hist, 'std': std_hist, 'cutoff_radius': cutoff_radius,
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
