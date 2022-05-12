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
from hilbert import decode

def generate_hilbert_curve(order):
    num_points_one_direction = 2**order
    num_points = num_points_one_direction**SPATIAL_DIM
    index = decode(np.arange(num_points), SPATIAL_DIM, order)
    # index = (
    #     index[:, 2] + index[:, 1] * num_points_one_direction +
    #     index[:, 0] * num_points_one_direction**2
    # )
    return index
