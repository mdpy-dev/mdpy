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
from ..math import *
from ..error import *

pbc_matrix = np.diag(np.ones(3)*10)
pbc_inv = np.linalg.inv(pbc_matrix)

def test_wrap_pbc():
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
    assert wrapped_positions[1, 1] == 5
    assert wrapped_positions[2, 2] == -5
    assert wrapped_positions[-1, 1] == -2
    assert wrapped_positions[-2, 0] == 1
    assert wrapped_positions[-2, 2] == 3

    with pytest.raises(AtomLossError):
        wrap_positions(np.array([16, 0, 1]), pbc_matrix, pbc_inv)

def test_unwrap_vec():
    vec = np.array([0, 6, 1])
    unwrapped_vec = unwrap_vec(vec, pbc_matrix, pbc_inv)
    assert unwrapped_vec[0] == 0
    assert unwrapped_vec[1] == pytest.approx(-4)
    assert unwrapped_vec[2] == 1

    vec = np.array([
        [0, 6, 1], [-5, -6, 9]
    ])
    unwrapped_vec = unwrap_vec(vec, pbc_matrix, pbc_inv)
    assert unwrapped_vec[1, 0] == -5
    assert unwrapped_vec[1, 1] == pytest.approx(4)
    assert unwrapped_vec[1, 2] == pytest.approx(-1)