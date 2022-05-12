#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : space_filling_curve.py
created time : 2022/05/13
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import numpy as np
from mdpy import SPATIAL_DIM
from mdpy.environment import *
from hilbert import decode

def generate_hilbert_curve_sequence(order):
    num_points_one_direction = 2**order
    num_points = num_points_one_direction**SPATIAL_DIM
    index = decode(np.arange(num_points), SPATIAL_DIM, order)
    index = list(
        index[:, 2] + index[:, 1] * num_points_one_direction +
        index[:, 0] * num_points_one_direction**2
    )
    cell_sequence = np.zeros([num_points], NUMPY_INT)
    for i in range(num_points):
        cell_sequence[i] = index.index(i)
    return cell_sequence
