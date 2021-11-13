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
from ..error import *

def wrap_positions(positions: np.ndarray, pbc_matrix: np.ndarray, pbc_inv: np.array):
    move_vec = - np.round(np.dot(positions, pbc_inv))
    if np.max(np.abs(move_vec)) >= 2:
        atom_id = np.unique([i[0] for i in np.argwhere(np.abs(move_vec) >= 2)])
        raise AtomLossError(
            'Atom(s) with matrix id: %s moved beyond 2 PBC image.' %atom_id
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