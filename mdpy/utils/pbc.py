#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : pbc.py
created time : 2021/10/22
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np
import numba as nb
from .. import SPATIAL_DIM
from ..error import *

def check_pbc_matrix(pbc_matrix):
    row, col = pbc_matrix.shape
    if row != SPATIAL_DIM or col != SPATIAL_DIM:
        raise ArrayDimError(
            'The pbc matrix should have shape [%d, %d], while matrix [%d %d] is provided'
            %(SPATIAL_DIM, SPATIAL_DIM, row, col)
        )
    if np.linalg.det(pbc_matrix) == 0:
        raise PBCPoorDefinedError(
            'PBC is poor defined. Two or more column vectors are linear corellated'
        )
    return pbc_matrix

def wrap_positions(positions: np.ndarray, pbc_matrix: np.ndarray, pbc_inv: np.array):
    move_vec = - np.round(np.dot(positions, pbc_inv))
    if np.max(np.abs(move_vec)) >= 2:
        particle_id = np.unique([i[0] for i in np.argwhere(np.abs(move_vec) >= 2)])
        raise ParticleLossError(
            'Atom(s) with matrix id: %s moved beyond 2 PBC image.' %(particle_id)
        )
    move_vec = np.dot(move_vec, pbc_matrix)
    return positions + move_vec

@nb.njit()
def unwrap_vec(vec: np.ndarray, pbc_matrix: np.ndarray, pbc_inv: np.array):
    scaled_vec = np.dot(vec, pbc_inv)
    temp_vec = np.empty(scaled_vec.shape)
    np.round_(scaled_vec, 0, temp_vec)
    scaled_vec -= temp_vec
    return np.dot(scaled_vec, pbc_matrix)