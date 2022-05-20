#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : xyz_writer.py
created time : 2022/05/15
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import datetime
import numpy as np
from mdpy import env
from mdpy.core import Topology
from mdpy.utils import check_quantity_value
from mdpy.unit import *
from mdpy.error import *

class XYZWriter:
    def __init__(
        self, file_path: str, mode: str = 'w',
        topology: Topology=Topology(),
        pbc_matrix = np.diag([0]*3).astype(env.NUMPY_FLOAT)
    ) -> None:
        if not file_path.endswith('.xyz'):
            raise FileFormatError('The file should end with .xyz suffix')
        self._file_path = file_path
        self._mode = mode
        if not isinstance(topology, Topology):
            raise TypeError(
                'The topology attribute should be the instance of mdpy.core.Topology class'
            )
        self._topology = topology
        self._pbc_matrix = check_quantity_value(pbc_matrix, default_length_unit)
        f = open(file_path, mode)
        f.close()

    def _write_info(self, info: str):
        with open(self._file_path, 'a') as f:
            print(info, file=f, end='')

    def write(self, positions: np.ndarray):
        shape, is_shape_error = positions.shape, False
        is_2d_array = True if len(shape) == 2 else False
        if is_2d_array and shape[0] != self._topology.num_particles:
            is_shape_error = True
        elif not is_2d_array and len(shape) != 3:
            is_shape_error = True
        elif not is_2d_array and shape[1] != self._topology.num_particles:
            is_shape_error = True
        if is_shape_error:
            raise ArrayDimError(
                'The topology contain %s particles while a positions array with shape %s is provided'
                %(self._topology.num_particles, list(shape))
            )
        if is_2d_array:
            self._write_frame(positions)
        else:
            for frame in range(shape[0]):
                self._write_frame(positions[frame, :])

    def _write_frame(self, positions: np.ndarray):
        self._write_info('%d\n' %self._topology.num_particles)
        self._write_info(
            'XYZ FILE CREATED WITH MDPY %-11s\n' %(
                datetime.date.today().strftime('%d-%b-%Y').upper()
            )
        )
        for i in range(self._topology.num_particles):
            self._write_info(
                '%s %.3f %.3f %.3f\n' %(
                    self._topology.particles[i].particle_type,
                    positions[i, 0],
                    positions[i, 1],
                    positions[i, 2]
                )
            )