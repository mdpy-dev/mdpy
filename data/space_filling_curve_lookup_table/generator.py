#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : generator.py
created time : 2022/05/20
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : space_filling_curve.py
created time : 2022/05/13
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import os
from markupsafe import string
import numpy as np
from typing import Tuple
from mdpy import SPATIAL_DIM
from mdpy.environment import *
from hilbert import decode

class Generator:
    def __init__(self, num_bits: int, num_dims: int=SPATIAL_DIM) -> None:
        self._num_bits = num_bits
        self._num_dims = num_dims
        # attribute
        self._num_single_dim_points = 2**self._num_bits
        self._num_points = self._num_single_dim_points**self._num_dims

    def generate(self, curve_type: string):
        if curve_type == 'morton':
            return self._generate_morton_curve()
        elif curve_type == 'hilbert':
            return self._generate_hilbert_curve()
        else:
            raise KeyError('Only morton and hilbert are supported')

    def _generate_morton_curve(self) -> Tuple[np.ndarray]:
        X, Y, Z = np.meshgrid(
            np.arange(self._num_single_dim_points, dtype=NUMPY_INT),
            np.arange(self._num_single_dim_points, dtype=NUMPY_INT),
            np.arange(self._num_single_dim_points, dtype=NUMPY_INT)
        )
        x = X.reshape([self._num_points])
        y = Y.reshape([self._num_points])
        z = Z.reshape([self._num_points])

        x = (x | (x << 16)) & 0x030000FF
        x = (x | (x <<  8)) & 0x0300F00F
        x = (x | (x <<  4)) & 0x030C30C3
        x = (x | (x <<  2)) & 0x09249249

        y = (y | (y << 16)) & 0x030000FF
        y = (y | (y <<  8)) & 0x0300F00F
        y = (y | (y <<  4)) & 0x030C30C3
        y = (y | (y <<  2)) & 0x09249249

        z = (z | (z << 16)) & 0x030000FF
        z = (z | (z <<  8)) & 0x0300F00F
        z = (z | (z <<  4)) & 0x030C30C3
        z = (z | (z <<  2)) & 0x09249249

        points_index = np.argsort(x | (y << 1) | (z << 2))
        points_index = np.stack([
            X.reshape([self._num_points])[points_index],
            Y.reshape([self._num_points])[points_index],
            Z.reshape([self._num_points])[points_index],
        ], axis=1).astype(NUMPY_INT)
        lookup_table = np.zeros([self._num_single_dim_points]*3, NUMPY_INT)
        lookup_table[points_index[:, 0], points_index[:, 1], points_index[:, 2]] = np.arange(self._num_points)
        index = 198
        print(points_index[index, :], lookup_table[points_index[index, 0], points_index[index, 1], points_index[index, 2]])
        return points_index, lookup_table

    def _generate_hilbert_curve(self) -> Tuple[np.ndarray]:
        points_index = decode(np.arange(self._num_points), self._num_dims, self._num_bits)
        lookup_table = np.zeros([self._num_single_dim_points]*3, NUMPY_INT)
        lookup_table[points_index[:, 0], points_index[:, 1], points_index[:, 2]] = np.arange(self._num_points)
        index = 198
        print(points_index[index, :], lookup_table[points_index[index, 0], points_index[index, 1], points_index[index, 2]])
        return points_index, lookup_table

    def write_xyz_file(self, file_name: str, points_index: np.ndarray):
        with open(file_name, 'w') as f:
            print('%d' %self._num_points, file=f)
            print('test' , file=f)
            for i in range(self._num_points):
                print('H %.2f %.2f %.2f %d' %(points_index[i, 0], points_index[i, 1], points_index[i, 2], i), file=f)

    def write_npy_file(self, file_name: str, lookup_table: np.ndarray):
        np.save(file_name, lookup_table.astype(NUMPY_INT))

if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    num_bits = 6
    generator = Generator(num_bits)

    curve_type = 'hilbert'
    points_index, lookup_table = generator.generate(curve_type)

    # output
    xyz_file = os.path.join(cur_dir, '%s.xyz' %curve_type)
    generator.write_xyz_file(xyz_file, points_index)
    # npy_file = os.path.join(cur_dir, 'morton_%d_bits.npy' %num_bits)
    # generator.write_npy_file(npy_file, lookup_table)
