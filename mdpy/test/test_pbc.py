#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_pbc.py
created time : 2021/10/22
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest
import numpy as np
from ..utils import *
from ..error import *
from .. import env

pbc_matrix = np.diag(np.ones(3)*10).astype(env.NUMPY_FLOAT)
pbc_inv = np.linalg.inv(pbc_matrix).astype(env.NUMPY_FLOAT)

def test_wrap_positions():
    positions = np.array([
        [0, 0, 0],
        [4, 5, 1],
        [-4, -1, -5],
        [6, 8, 9],
        [8, 0, 1],
        [-7, -8, 9],
        [11, 12, 3],
        [-3, -12., -14]
    ])
    wrapped_positions = wrap_positions(positions, pbc_matrix, pbc_inv)
    assert wrapped_positions[0, 0] == 0
    assert wrapped_positions[3, 0] == -4
    assert wrapped_positions[1, 1] == -5
    assert wrapped_positions[2, 2] == 5
    assert wrapped_positions[-1, 1] == -2
    assert wrapped_positions[-2, 0] == 1
    assert wrapped_positions[-2, 2] == 3

    with pytest.raises(AtomLossError):
        wrap_positions(np.array([16, 0, 1]), pbc_matrix, pbc_inv)

def test_unwrap_vec():
    vec = np.array([0, 6, 1]).astype(env.NUMPY_FLOAT)
    unwrapped_vec = unwrap_vec(vec, pbc_matrix, pbc_inv)
    assert unwrapped_vec[0] == 0
    assert unwrapped_vec[1] == pytest.approx(-4)
    assert unwrapped_vec[2] == 1

    vec = np.array([-5, -6, 9]).astype(env.NUMPY_FLOAT)
    unwrapped_vec = unwrap_vec(vec, pbc_matrix, pbc_inv)
    assert unwrapped_vec[0] == -5
    assert unwrapped_vec[1] == pytest.approx(4)
    assert unwrapped_vec[2] == pytest.approx(-1)

    p1 = np.array([0, 0, 0]).astype(env.NUMPY_FLOAT)
    p2 = np.array([0, 1, 0]).astype(env.NUMPY_FLOAT)
    vec1 = p1 - p2
    vec2 = unwrap_vec(p1 + 10 - p2, pbc_matrix, pbc_inv)
    vec3 = unwrap_vec(p1 - 10 - p2, pbc_matrix, pbc_inv)
    assert vec1[0] == pytest.approx(vec2[0])
    assert vec1[0] == pytest.approx(vec3[0])
    assert vec1[1] == pytest.approx(vec2[1])
    assert vec1[1] == pytest.approx(vec3[1])
    assert vec1[2] == pytest.approx(vec2[2])
    assert vec1[2] == pytest.approx(vec3[2])

    vec = unwrap_vec(np.array([
        [0, 9, 0],
        [0, 1, 0],
        [9, -1, 0],
        [0, 1, -8]
    ]).astype(env.NUMPY_FLOAT), pbc_matrix, pbc_inv)
    assert vec[0, 1] == pytest.approx(-1)
    assert vec[3, 2] == pytest.approx(2)
