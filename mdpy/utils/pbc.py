#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : pbc.py
created time : 2021/10/22
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import numpy as np
import numba as nb
from mdpy import SPATIAL_DIM
from mdpy.error import *


def check_pbc_matrix(pbc_matrix):
    row, col = pbc_matrix.shape
    if row != SPATIAL_DIM or col != SPATIAL_DIM:
        raise ArrayDimError(
            "The pbc matrix should have shape [%d, %d], while matrix [%d %d] is provided"
            % (SPATIAL_DIM, SPATIAL_DIM, row, col)
        )
    if np.linalg.det(pbc_matrix) == 0:
        raise PBCPoorDefinedError(
            "PBC is poor defined. Two or more column vectors are linear corellated"
        )
    return pbc_matrix


def wrap_positions(positions, pbc_diag):
    half_pbc_diag = pbc_diag / 2
    move_vec = (positions < -half_pbc_diag) * pbc_diag
    move_vec -= (positions > half_pbc_diag) * pbc_diag
    return positions + move_vec


def unwrap_vec(vec, pbc_diag):
    half_pbc_diag = pbc_diag / 2
    shift = (vec < -half_pbc_diag) * pbc_diag
    shift -= (vec > half_pbc_diag) * pbc_diag
    return vec + shift
