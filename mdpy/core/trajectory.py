#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : trajectory.py
created time : 2022/02/19
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np
from . import Topology
from .. import SPATIAL_DIM, env
from ..utils import check_quantity_value
from ..error import *
from ..unit import *
from ..utils.select import *

class Trajectory:
    def __init__(
        self, topology: Topology,
        contain_positions=True,
        contain_velocities=False,
        contain_forces = False
    ) -> None:
        self._topology = topology
        self._contain_positions = contain_positions
        self._contain_velocities = contain_velocities
        self._contain_forces = contain_forces

        self._num_frames = 0
        self._positions = np.zeros([self._num_frames, self._topology.num_particles, SPATIAL_DIM], dtype=env.NUMPY_FLOAT)
        self._unwrapped_positions = np.zeros([self._num_frames, self._topology.num_particles, SPATIAL_DIM], dtype=env.NUMPY_FLOAT)
        self._velocities = np.zeros([self._num_frames, self._topology.num_particles, SPATIAL_DIM], dtype=env.NUMPY_FLOAT)
        self._forces = np.zeros([self._num_frames, self._topology.num_particles, SPATIAL_DIM], dtype=env.NUMPY_FLOAT)

        self._pbc_matrix = np.zeros([SPATIAL_DIM, SPATIAL_DIM], dtype=env.NUMPY_FLOAT)
        self._pbc_inv = np.zeros([SPATIAL_DIM, SPATIAL_DIM], dtype=env.NUMPY_FLOAT)
        self._is_pbc_specified = False
        self._time_step = None

    def __repr__(self) -> str:
        return '<mdpy.core.Trajectory object of %d particles with %d frames at %x>' %(
            self._topology.num_particles, self._num_frames, id(self)
        )

    __str__ = __repr__

    def set_time_step(self, time_step):
        self._time_step = check_quantity_value(time_step, default_time_unit)

    def _check_pbc_matrix(self, pbc_matrix):
        row, col = pbc_matrix.shape
        if row != SPATIAL_DIM or col != SPATIAL_DIM:
            raise ArrayDimError(
                'The pbc matrix should have shape [%d, %d], while matrix [%d %d] is provided'
                %(SPATIAL_DIM, SPATIAL_DIM, row, col)
            )
        if np.linalg.det(pbc_matrix) == 0:
            raise PBCPoorDefinedError(
                'PBC of %s is poor defined. Two or more column vectors are linear corellated'
            )

    def set_pbc_matrix(self, pbc_matrix):
        pbc_matrix = check_quantity_value(pbc_matrix, default_length_unit)
        self._check_pbc_matrix(pbc_matrix)
        # The origin define of pbc_matrix is the stack of 3 column vector
        # While in MDPy the position is in shape of n x 3
        # So the scaled position will be Position * PBC instead of PBC * Position as usual
        self._pbc_matrix = np.ascontiguousarray(pbc_matrix.T, dtype=env.NUMPY_FLOAT)
        self._pbc_inv = np.ascontiguousarray(np.linalg.inv(self._pbc_matrix), dtype=env.NUMPY_FLOAT)
        self._is_pbc_specified = True

    def _check_array(self, array: np.ndarray):
        if not isinstance(array, np.ndarray):
            raise TypeError('Array should be numpy.ndarray, instead of %s' %type(array))
        shape = list(array.shape)
        num_dims = len(shape)
        if num_dims == 2: # Single frame
            if shape[0] != self._topology.num_particles or shape[1] != SPATIAL_DIM:
                raise ArrayDimError(
                    'The dimension of array should be [%d, %d], while array [%d, %d] is provided'
                    %(self._topology.num_particles, SPATIAL_DIM, shape[0], shape[1])
                )
        elif num_dims == 3:
            if shape[1] != self._topology.num_particles or shape[2] != SPATIAL_DIM:
                raise ArrayDimError(
                    'The dimension of array should be [x, %d, %d], while array [x, %d, %d] is provided'
                    %(self._topology.num_particles, SPATIAL_DIM, shape[1], shape[2])
                )
        else:
            raise ArrayDimError('The dimensions of array should be 2 or 3 instead of %d' %num_dims)

        if num_dims == 2:
            return array.reshape([1, shape[0], shape[1]]).astype(env.NUMPY_FLOAT)
        return array.astype(env.NUMPY_FLOAT)

    def append(self, positions=None, velocities=None, forces=None):
        # Check input
        num_frames = [-1] * 3
        if self._contain_positions:
            if isinstance(positions, type(None)):
                raise TrajectoryPoorDefinedError('This mdpy.core.Trajectory instance contains positions, keyword positions should be specified')
            else:
                positions = self._check_array(positions)
                num_frames[0] = positions.shape[0]
        else:
            if not isinstance(positions, type(None)):
                raise TrajectoryPoorDefinedError('This mdpy.core.Trajectory instance does not contain positions, keyword positions should be `None`')

        if self._contain_velocities:
            if isinstance(velocities, type(None)):
                raise TrajectoryPoorDefinedError('This mdpy.core.Trajectory instance contains velocities, keyword velocities should be specified')
            else:
                velocities = self._check_array(velocities)
                num_frames[1] = velocities.shape[0]
        else:
            if not isinstance(velocities, type(None)):
                raise TrajectoryPoorDefinedError('This mdpy.core.Trajectory instance does not contain velocities, keyword velocities should be `None`')
        
        if self._contain_forces:
            if isinstance(forces, type(None)):
                raise TrajectoryPoorDefinedError('This mdpy.core.Trajectory instance contains forces, keyword forces should be specified')
            else:
                forces = self._check_array(forces)
                num_frames[2] = forces.shape[0]
        else:
            if not isinstance(forces, type(None)):
                raise TrajectoryPoorDefinedError('This mdpy.core.Trajectory instance does not contain forces, keyword forces should be `None`')

        num_frames = [i for i in num_frames if i != -1]
        if len(num_frames) == 0:
            return None
        else:
            if len(set(num_frames)) == 1:
                num_frames = num_frames[0]
            else:
                raise ArrayDimError('The frames of each array is not same, please check')

        # Append data
        if self._contain_positions:
            self._positions = np.vstack([self._positions, positions])
        
        if self._contain_velocities:
            self._velocities = np.vstack([self._velocities, velocities])

        if self._contain_forces:
            self._forces = np.vstack([self._forces, forces])

        self._num_frames += num_frames

    def unwrap_positions(self):
        self._check_pbc_matrix(self._pbc_matrix)
        scaled_positions = np.dot(self._positions, self._pbc_inv)
        scaled_diff = scaled_positions[1:, :, :] - scaled_positions[0:-1, :, :]
        scaled_diff -= np.round(scaled_diff)
        diff = np.dot(scaled_diff, self._pbc_matrix)
        self._unwrapped_positions = np.zeros_like(self._positions, dtype=env.NUMPY_FLOAT)
        self._unwrapped_positions[0, :, :] = self._positions[0, :, :]
        for frame in range(1, self._num_frames):
            self._unwrapped_positions[frame, :, :] = self._unwrapped_positions[frame-1, :, :] + diff[frame-1, :, :]

    @property
    def topology(self):
        return self._topology

    @property
    def num_frames(self):
        return self._num_frames

    @property
    def time_step(self):
        return self._time_step

    @property
    def time_series(self):
        if self._time_step == None:
            raise TrajectoryPoorDefinedError('time_step has not been specified, please call mdpy.core.Trajectory.set_time_step()')
        return (np.arange(0, self._num_frames) * self._time_step).astype(env.NUMPY_FLOAT)

    @property
    def pbc_matrix(self):
        if not self._is_pbc_specified:
            raise PBCPoorDefinedError('PBC has not been specified before calling.')
        return self._pbc_matrix
    
    @property
    def pbc_inv(self):
        if not self._is_pbc_specified:
            raise PBCPoorDefinedError('PBC has not been specified before calling.')
        return self._pbc_inv

    @property
    def positions(self):
        if self._contain_positions == False:
            raise TrajectoryPoorDefinedError('Positions is not contained in this mdpy.core.Trajectory instance')
        return self._positions

    @property
    def unwrapped_position(self):
        return self._unwrapped_positions

    @property
    def velocities(self):
        if self._contain_velocities == False:
            raise TrajectoryPoorDefinedError('Velocities is not contained in this mdpy.core.Trajectory instance')
        return self._velocities

    @property
    def forces(self):
        if self._contain_forces == False:
            raise TrajectoryPoorDefinedError('Forces is not contained in this mdpy.core.Trajectory instance')
        return self._forces