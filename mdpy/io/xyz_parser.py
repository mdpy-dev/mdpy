#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : xyz_parser.py
created time : 2022/04/26
author : Yinong Zhao
copyright : (C)Copyright 2021-present, mdpy organization
'''

from turtle import end_fill
import numpy as np
from mdpy import SPATIAL_DIM
from mdpy.environment import *
from mdpy.error import *

class XYZParser:
    def __init__(self, file_path, is_parse_all=True) -> None:
        # Initial reader and parser setting
        if not file_path.endswith('.xyz'):
            raise FileFormatError('The file should end with .xyz suffix')
        self._file_path = file_path
        self._is_parse_all = is_parse_all

        self._file = open(self._file_path, 'r')
        line = self._file.readline()
        self._num_particles = int(line)

        self._num_frames = 0
        multiframe_positions = []
        while line:
            line, particle_types, positions = self._parse_single_frame(line)
            multiframe_positions.append(positions)
            self._num_frames += 1
        if self._num_particles != multiframe_positions[0].shape[0]:
            raise ArrayDimError(
                'XYZ file contains %d particles, while positions of %d particles are provided' %(
                    self._num_particles, multiframe_positions[0].shape[0]
                )
            )
        if self._num_frames == 1:
            self._positions = multiframe_positions[0]
        else:
            self._positions = np.stack(multiframe_positions).astype(NUMPY_FLOAT)
        self._particle_types = particle_types

    def _parse_single_line(self, line: str):
        data = line.strip().split()
        particle_type = data[0]
        position = [float(data[-3]), float(data[-2]), float(data[-1])]
        return particle_type, position

    def _parse_single_frame(self, line: str):
        particle_types, positions = [], []
        line = self._file.readline() # Skip num particle line
        line = self._file.readline() # Skip comment line
        while not line.startswith('%d' %self._num_particles):
            if line == '':
                break
            particle_type, position = self._parse_single_line(line)
            particle_types.append(particle_type)
            positions.append(position)
            line = self._file.readline()
        return line, particle_types, np.stack(positions).astype(NUMPY_FLOAT)

    def get_positions(self, *frames):
        num_target_frames = len(frames)
        if num_target_frames == 1:
            if frames[0] >= self._num_frames:
                raise ArrayDimError(
                    '%d beyond the number of frames %d stored in pdb file'
                    %(frames[0], self._num_frames)
                )
            result = self._reader.trajectory[frames[0]].positions.copy().astype(NUMPY_FLOAT)
        else:
            result = np.zeros([num_target_frames, self._num_particles, SPATIAL_DIM])
            for index, frame in enumerate(frames):
                if frame >= self._num_frames:
                    raise ArrayDimError(
                        '%d beyond the number of frames %d stored in pdb file'
                        %(frame, self._num_frames)
                    )
                result[index, :, :] = self._reader.trajectory[frame].positions.astype(NUMPY_FLOAT)
        return result

    @property
    def particle_types(self):
        return self._particle_types

    @property
    def num_frames(self):
        return self._num_frames

    @property
    def num_particles(self):
        return self._num_particles

    @property
    def positions(self) -> np.ndarray:
        if not self._is_parse_all:
            raise IOPoorDefinedError(
                'positions property is not supported as `is_parse_all==False`, calling `get_position` method'
            )
        return self._positions.copy()
