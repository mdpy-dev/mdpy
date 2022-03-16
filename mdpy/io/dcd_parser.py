#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : dcd_parser.py
created time : 2022/03/10
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np
import MDAnalysis as mda
from mdpy import env, SPATIAL_DIM
from mdpy.error import *

class DCDParser:
    def __init__(self, file_path: str, is_parse_all=True) -> None:
        # Initial reader and parser setting
        if not file_path.endswith('.dcd'):
            raise FileFormatError('The file should end with .dcd suffix')
        self._file_path = file_path
        self._is_parse_all = is_parse_all
        self._reader = mda.coordinates.DCD.DCDReader(self._file_path)
        if self._is_parse_all:
            if self._reader.n_frames == 1:
                self._positions = self._reader.ts.positions.astype(env.NUMPY_FLOAT)
            else:
                self._positions = [ts.positions.astype(env.NUMPY_FLOAT) for ts in self._reader.trajectory]
                self._positions = np.stack(self._positions)
        self._pbc_matrix = self._reader.ts.triclinic_dimensions

    def get_positions(self, *frames):
        num_frames = self._reader.trajectory.n_frames
        num_particles = self._reader.trajectory.n_atoms
        num_target_frames = len(frames)
        if num_target_frames == 1:
            if frames[0] >= num_frames:
                raise ArrayDimError(
                    '%d beyond the number of frames %d stored in dcd file'
                    %(frames[0], num_frames)
                )
            result = self._reader.trajectory[frames[0]].positions.copy().astype(env.NUMPY_FLOAT)
        else:
            result = np.zeros([num_target_frames, num_particles, SPATIAL_DIM])
            for index, frame in enumerate(frames):
                if frame >= num_frames:
                    raise ArrayDimError(
                        '%d beyond the number of frames %d stored in dcd file'
                        %(frame, num_frames)
                    )
                result[index, :, :] = self._reader.trajectory[frame].positions.astype(env.NUMPY_FLOAT)
        return result

    @property
    def positions(self) -> np.ndarray:
        if not self._is_parse_all:
            raise ParserPoorDefinedError(
                'positions property is not supported as `is_parse_all==False`, calling `get_position` method'
            )
        return self._positions.copy()

    @property
    def pbc_matrix(self) -> np.ndarray:
        return self._pbc_matrix.copy()