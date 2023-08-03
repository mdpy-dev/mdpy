#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_pbc.py
created time : 2021/10/22
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import pytest
import numpy as np
from mdpy.utils import *
from mdpy.error import *
from mdpy.environment import *


def test_check_pbc_matrix():
    with pytest.raises(PBCPoorDefinedError):
        check_pbc_matrix(np.ones([3, 3]))

    with pytest.raises(ArrayDimError):
        check_pbc_matrix(np.ones([4, 3]))


def test_wrap_positions():
    positions = cp.array(
        [
            [0, 0, 0],
            [4, 5, 1],
            [-4, -1, -5],
            [6, 8, 9],
            [8, 0, 1],
            [-7, -8, 9],
            [11, 12, 3],
            [-3, -12.0, -14],
        ]
    ).astype(CUPY_FLOAT)
    pbc_matrix = cp.diag(cp.ones(3) * 10).astype(CUPY_FLOAT)
    pbc_inv = cp.linalg.inv(pbc_matrix)
    wrapped_positions = wrap_positions(positions, pbc_matrix, pbc_inv).get()
    assert wrapped_positions[0, 0] == 0
    assert wrapped_positions[3, 0] == -4
    assert wrapped_positions[1, 1] == 5
    assert wrapped_positions[2, 2] == -5
    assert wrapped_positions[-1, 1] == -2
    assert wrapped_positions[-2, 0] == 1
    assert wrapped_positions[-2, 2] == 3

    # with pytest.raises(ParticleLossError):
    #     wrap_positions(np.array([16, 0, 1]), pbc_matrix, pbc_inv)


def test_unwrap_vec():
    vec = np.array([0, 6, 1]).astype(NUMPY_FLOAT)
    pbc_matrix = np.diag(np.ones(3) * 10).astype(NUMPY_FLOAT)
    pbc_diag = np.diagonal(pbc_matrix)
    unwrapped_vec = unwrap_vec(vec, pbc_diag)
    assert unwrapped_vec[0] == 0
    assert unwrapped_vec[1] == pytest.approx(-4)
    assert unwrapped_vec[2] == 1

    vec = np.array([-5, -6, 9]).astype(NUMPY_FLOAT)
    unwrapped_vec = unwrap_vec(vec, pbc_diag)
    assert unwrapped_vec[0] == -5
    assert unwrapped_vec[1] == pytest.approx(4)
    assert unwrapped_vec[2] == pytest.approx(-1)

    p1 = np.array([0, 0, 0]).astype(NUMPY_FLOAT)
    p2 = np.array([0, 1, 0]).astype(NUMPY_FLOAT)
    vec1 = p1 - p2
    vec2 = unwrap_vec(p1 + 10 - p2, pbc_diag)
    vec3 = unwrap_vec(p1 - 10 - p2, pbc_diag)
    assert vec1[0] == pytest.approx(vec2[0])
    assert vec1[0] == pytest.approx(vec3[0])
    assert vec1[1] == pytest.approx(vec2[1])
    assert vec1[1] == pytest.approx(vec3[1])
    assert vec1[2] == pytest.approx(vec2[2])
    assert vec1[2] == pytest.approx(vec3[2])

    vec = unwrap_vec(
        np.array([[0, 9, 0], [0, 1, 0], [9, -1, 0], [0, 1, -8]]).astype(NUMPY_FLOAT),
        pbc_diag,
    )
    assert vec[0, 1] == pytest.approx(-1)
    assert vec[3, 2] == pytest.approx(2)
