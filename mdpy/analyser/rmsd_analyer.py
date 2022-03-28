#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : rmsd_analyer.py
created time : 2022/03/01
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np
import scipy.optimize as optimize
from mdpy import SPATIAL_DIM
from mdpy.analyser import AnalyserResult
from mdpy.core import Trajectory
from mdpy.utils import generate_rotation_matrix
from mdpy.utils import check_quantity_value
from mdpy.utils import select, parse_selection_condition, check_topological_selection_condition
from mdpy.unit import *
from mdpy.error import *

class RMSDAnalyser:
    def __init__(
        self, reference_positions,
        selection_condition: list[dict]
    ) -> None:
        reference_positions = check_quantity_value(
            reference_positions, default_length_unit
        )
        if reference_positions.ndim != 2 or reference_positions.shape[-1] != SPATIAL_DIM:
            raise ArrayDimError(
                'Position matrix should have shape [n, 3], while %s is provided'
                %list(reference_positions.shape)
            )
        self._reference_positions = reference_positions
        self._num_reference_particles = reference_positions.shape[0]
        check_topological_selection_condition(selection_condition)
        self._selection_condition = selection_condition

    def analysis(self, trajectory: Trajectory, is_dimensionless=True) -> AnalyserResult:
        # Read input
        selected_matrix_id = select(trajectory, self._selection_condition)[0]
        num_particles = len(selected_matrix_id)
        if self._num_reference_particles != num_particles:
            raise AnalyserPoorDefinedError(
                'RMSD reference contain %d particles, while %d is selected'
                %(self._num_reference_particles, num_particles)
            )
        # Analysis RMSD
        parameters = np.array([0, 0, 0])
        rmsd = np.zeros([trajectory.num_frames])
        rotation = np.zeros([trajectory.num_frames, SPATIAL_DIM, SPATIAL_DIM])
        rotation_angle = np.zeros([trajectory.num_frames, SPATIAL_DIM])
        for frame in range(trajectory.num_frames):
            res = optimize.minimize(self._rmsd, parameters, args=(
                self._reference_positions, trajectory.positions[frame, selected_matrix_id, :]
            ))
            parameters = res.x
            yaw, pitch, roll = parameters
            rmsd[frame] = res.fun
            rotation[frame, :, :] = generate_rotation_matrix(yaw, pitch, roll)
            rotation_angle[frame, :] = np.array([yaw, pitch, roll])
        # Output
        if not is_dimensionless:
            rmsd = Quantity(rmsd, default_length_unit)
        title = 'RMSD of %s' %(
            parse_selection_condition(self._selection_condition)
        )
        description = {
            'rmsd': 'The RMSD value verse reference of each frame, unit: default_length_unit',
            'rotation': 'The rotation matrix used to align the positions of each frame, unit: dimensionless',
            'rotation_angles': 'The rotation angles of [roll, pitch, yaw] of each frame, unit: degree'
        }
        data = {
            'rmsd': rmsd,
            'rotation': rotation, 'rotation_angles': rotation_angle
        }
        return AnalyserResult(title=title, description=description, data=data)

    @staticmethod
    def _rmsd(parameters, reference_positions, positions):
        yaw, pitch, roll = parameters
        rotation = generate_rotation_matrix(yaw, pitch, roll)
        transformed_positions = positions - positions.mean(0)
        transformed_positions = np.dot(transformed_positions, rotation)
        return np.sqrt(((
            reference_positions - reference_positions.mean(0) -
            transformed_positions
        )**2).sum())

    @property
    def selection_condition(self):
        return self._selection_condition_1

    @selection_condition.setter
    def selection_condition(self, selection_condition: list[dict]):
        check_topological_selection_condition(selection_condition)
        self._selection_condition = selection_condition

    @property
    def reference_positions(self):
        return self._reference_positions

    @reference_positions.setter
    def reference_positions(self, reference_positions):
        reference_positions = check_quantity_value(
            reference_positions, default_length_unit
        )
        if reference_positions.ndim != 2 or reference_positions.shape[-1] != SPATIAL_DIM:
            raise ArrayDimError(
                'Position matrix should have shape [n, 3], while %s is provided'
                %list(reference_positions.shape)
            )
        self._reference_positions = reference_positions
        self._num_reference_particles = reference_positions.shape[0]

    @property
    def num_reference_particles(self):
        return self._num_reference_particles