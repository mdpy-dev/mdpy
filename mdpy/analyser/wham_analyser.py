#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : wham_analyser.py
created time : 2022/03/10
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import numpy as np
from mdpy.analyser import AnalyserResult
from mdpy.core import Trajectory
from mdpy.utils import check_quantity
from mdpy.unit import *


class WHAMAnalyser:
    def __init__(
        self, cv_analysis_fun, cv_range: list, num_bins: int, temperature
    ) -> None:
        self._cv_analysis_fun = cv_analysis_fun
        self._cv_range = cv_range
        self._num_bins = num_bins
        self._temperature = check_quantity(temperature, default_temperature_unit)

    def analysis(
        self,
        trajectory_list: list[Trajectory],
        kappa_list: list,
        cv_center_list: list,
        max_iterations=50,
        is_dimensionless=True,
    ) -> AnalyserResult:
        num_trajectories = len(trajectory_list)
        # Create unchanged array
        cv_hist_array = np.zeros([num_trajectories, self._num_bins])
        bias_factor_array = np.zeros([num_trajectories, self._num_bins])
        num_samples_vec = np.zeros([num_trajectories, 1])
        for index, trajectory in enumerate(trajectory_list):
            cv_temp = self._cv_analysis_fun(trajectory)
            cv_hist_array[index, :], bin_edges = np.histogram(
                cv_temp, self._num_bins, range=(self._cv_range[0], self._cv_range[1])
            )
            bin_width = bin_edges[1] - bin_edges[0]
            bin_centers = bin_edges[:-1] + bin_width / 2
            bias_factor_array[index, :] = (
                check_quantity(kappa_list[index], default_energy_unit)
                * Quantity((cv_center_list[index] - bin_centers) ** 2)
                / self._temperature
                / KB
            ).value
            num_samples_vec[index] = trajectory.num_frames
        # Iteration solve
        normalization_vec = np.ones([num_trajectories, 1])
        p_est_cur = cv_hist_array.sum(0) / (
            num_samples_vec * normalization_vec * bias_factor_array
        ).sum(0)
        for i in range(max_iterations):
            normalization_vec = (
                (bias_factor_array * p_est_cur * bin_width)
                .sum(1)
                .reshape([num_trajectories, 1])
            )
            p_est_cur, p_est_pre = (
                cv_hist_array.sum(0)
                * num_trajectories
                / (num_samples_vec * normalization_vec * bias_factor_array).sum(0),
                p_est_cur,
            )
        free_energy = (
            -(self._temperature * KB * Quantity(np.log(p_est_cur)))
            .convert_to(default_energy_unit)
            .value
        )
        # Output
        if not is_dimensionless:
            free_energy = Quantity(free_energy, default_energy_unit)
        title = "WHAM result"
        description = {
            "num_trajectories": "The number of trajectories, shape: 1, unit: dimensionless",
            "cv_hist_array": "The histogram of cv for each trajectory, shape: (num_trajectories, num_bins), unit: dimensionless",
            "cv_norm_hist_array": "The normalized histogram of cv for each trajectory, shape: (num_trajectories, num_bins), unit: dimensionless",
            "cv_bin_edges": "The bin edges of cv's histogram, shape: (num_bins+1), unit: the dimension of cv",
            "cv_bin_centers": "The bin centers of cv's histogram, shape: (num_bins), unit: the dimension of cv",
            "cv_bin_width": "The bin width of cv's histogram, shape: 1, unit: the dimension of cv",
            "p_est": "The estimation of probability distribution, shape: (num_bins), unit: dimensionless",
            "free_energy": "The free energy along bin_edge, shape: (num_bins), unit: default_energy_unit",
        }
        data = {
            "num_trajectories": num_trajectories,
            "cv_hist_array": cv_hist_array,
            "cv_norm_hist_array": cv_hist_array / num_samples_vec,
            "cv_bin_edges": bin_edges,
            "cv_bin_centers": bin_centers,
            "cv_bin_width": bin_width,
            "p_est": p_est_cur,
            "free_energy": free_energy,
        }
        return AnalyserResult(title=title, description=description, data=data)
