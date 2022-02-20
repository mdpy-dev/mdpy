#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_geometry.py
created time : 2021/10/09
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest
import numpy as np
from .. import env
from ..utils import *

def test_get_unit_vec():
    vec = np.array([1, 1], dtype=env.NUMPY_FLOAT)
    unit_vec = get_unit_vec(vec)
    assert unit_vec[0] == pytest.approx(np.sqrt(2) / 2)
    assert unit_vec[1] == pytest.approx(np.sqrt(2) / 2)

def test_get_norm_vec():
    p1 = np.array([0, 0, 0])
    p2 = np.array([1, 0, 0])
    p3 = np.array([0, 1, 0])
    norm_vec = get_norm_vec(p1, p2, p3)
    assert norm_vec[0] == 0
    assert norm_vec[1] == 0
    assert norm_vec[2] == 1

def test_get_bond():
    position1 = [0, 1, 0]
    position2 = [0, 0, 0]
    assert get_bond(position1, position2) == 1

    position2 = [3, 2, 2]
    assert get_bond(position2, position1) == pytest.approx(np.sqrt(14))

def test_get_pbc_bond():
    pbc_matrix = np.diag(np.ones(3)*10).astype(env.NUMPY_FLOAT)
    pbc_inv = np.linalg.inv(pbc_matrix).astype(env.NUMPY_FLOAT)

    position1 = np.array([0, 1, 0]).astype(env.NUMPY_FLOAT)
    position2 = np.array([0, 0, 0]).astype(env.NUMPY_FLOAT)
    assert get_pbc_bond(position1, position2, pbc_matrix, pbc_inv) == 1

    position1 = np.array([0, 5, 0]).astype(env.NUMPY_FLOAT)
    position2 = np.array([0, 0, 0]).astype(env.NUMPY_FLOAT)
    assert get_pbc_bond(position1, position2, pbc_matrix, pbc_inv) == 5

    position1 = np.array([0, 6, 0]).astype(env.NUMPY_FLOAT)
    position2 = np.array([0, 0, 0]).astype(env.NUMPY_FLOAT)
    assert get_pbc_bond(position1, position2, pbc_matrix, pbc_inv) == pytest.approx(4)

def test_get_angle():
    p1 = np.array([1, 1])
    p2 = np.array([0, 0])
    p3 = np.array([0, 1])
    angle = get_angle(p1, p2, p3)
    assert angle == pytest.approx(np.pi / 4)
    
    angle = get_angle(p1, p2, p3, is_angular=False)
    assert angle == pytest.approx(45)

def test_get_pbc_angle():
    pbc_matrix = np.diag(np.ones(3)*10).astype(env.NUMPY_FLOAT)
    pbc_inv = np.linalg.inv(pbc_matrix).astype(env.NUMPY_FLOAT)

    p1 = np.array([1, 1, 0]).astype(env.NUMPY_FLOAT)
    p2 = np.array([0, 0, 0]).astype(env.NUMPY_FLOAT)
    p3 = np.array([0, 1, 0]).astype(env.NUMPY_FLOAT)
    angle = get_pbc_angle(p1, p2, p3, pbc_matrix, pbc_inv)
    assert angle == pytest.approx(np.pi / 4)

    p1 = np.array([11, 1, 0]).astype(env.NUMPY_FLOAT)
    p2 = np.array([0, 0, 0]).astype(env.NUMPY_FLOAT)
    p3 = np.array([0, 1, 0]).astype(env.NUMPY_FLOAT)
    angle = get_pbc_angle(p1, p2, p3, pbc_matrix, pbc_inv)
    assert angle == pytest.approx(np.pi / 4)
    p1 = np.array([1, 1, 0]).astype(env.NUMPY_FLOAT) + 5
    p2 = np.array([0, 0, 0]).astype(env.NUMPY_FLOAT) + 5
    p3 = np.array([0, 1, 0]).astype(env.NUMPY_FLOAT) + 5
    angle = get_pbc_angle(p1, p2, p3, pbc_matrix, pbc_inv)
    assert angle == pytest.approx(np.pi / 4)

def test_get_included_angle():
    p1 = np.array([1, 1])
    p2 = np.array([0, 0])
    p3 = np.array([0, 1])
    v1 = p1 - p2
    v2 = p3 - p2
    angle = get_included_angle(v1, v2)
    assert angle == pytest.approx(np.pi / 4)
    
    angle = get_included_angle(v1, v2, is_angular=False)
    assert angle == pytest.approx(45)

def test_get_dihedral():
    p1 = np.array([0, 1, 1])
    p2 = np.array([0, 0, 0])
    p3 = np.array([1, 0, 0])
    p4 = np.array([1, 1, 0])
    
    dihedral = get_dihedral(p1, p2, p3, p4)
    assert dihedral == pytest.approx(- np.pi / 4)
    dihedral = get_dihedral(p1, p2, p3, p4, is_angular=False)
    assert dihedral == pytest.approx(- 45)

    p1 = np.array([0, 1, 1])
    p2 = np.array([0, 0, 0])
    p3 = np.array([1, 0, 0])
    p4 = np.array([1, 0, -1])
    
    dihedral = get_dihedral(p1, p2, p3, p4)
    assert dihedral == pytest.approx(- np.pi * 3/ 4)
    dihedral = get_dihedral(p1, p2, p3, p4, is_angular=False)
    assert dihedral == pytest.approx(- 135)

    p1 = np.array([0, 1, 1])
    p2 = np.array([0, 0, 0])
    p3 = np.array([1, 0, 0])
    p4 = np.array([1, 0, 1])
    
    dihedral = get_dihedral(p1, p2, p3, p4)
    assert dihedral == pytest.approx(np.pi / 4)
    dihedral = get_dihedral(p1, p2, p3, p4, is_angular=False)
    assert dihedral == pytest.approx(45)

    p1 = np.array([0, 1, 1])
    p2 = np.array([0, 0, 0])
    p3 = np.array([1, 0, 0])
    p4 = np.array([1, -1, 0])
    
    dihedral = get_dihedral(p1, p2, p3, p4)
    assert dihedral == pytest.approx(np.pi * 3/ 4)
    dihedral = get_dihedral(p1, p2, p3, p4, is_angular=False)
    assert dihedral == pytest.approx(135)

    # Improper
    p1 = np.array([0, 0, 0])
    p2 = np.array([1, 0, 0])
    p3 = np.array([0, 1, 0])
    p4 = np.array([0.5, 0.5, 1])
    
    improper = get_dihedral(p1, p2, p3, p4)
    assert improper == pytest.approx(np.pi / 2)

    improper = get_dihedral(p1, p2, p3, p4, is_angular=False)
    assert improper == pytest.approx(90)

    p1 = np.array([0, 0, 0])
    p2 = np.array([1, 0, 0])
    p3 = np.array([0, 1, 0])
    p4 = np.array([0.5, 0.5, -1])
    
    improper = get_dihedral(p1, p2, p3, p4)
    assert improper == pytest.approx(- np.pi / 2)

    improper = get_dihedral(p1, p2, p3, p4, is_angular=False)
    assert improper == pytest.approx(-90)

def test_get_pbc_dihedral():
    pbc_matrix = np.diag(np.ones(3)*10).astype(env.NUMPY_FLOAT)
    pbc_inv = np.linalg.inv(pbc_matrix).astype(env.NUMPY_FLOAT)

    p1 = np.array([0, 1, 1]).astype(env.NUMPY_FLOAT) + 10
    p2 = np.array([0, 0, 0]).astype(env.NUMPY_FLOAT)
    p3 = np.array([1, 0, 0]).astype(env.NUMPY_FLOAT)
    p4 = np.array([1, 1, 0]).astype(env.NUMPY_FLOAT)
    
    dihedral = get_pbc_dihedral(p1, p2, p3, p4, pbc_matrix, pbc_inv)
    assert dihedral == pytest.approx(- np.pi / 4)

    p1 = np.array([0, 0, 0]).astype(env.NUMPY_FLOAT)
    p2 = np.array([1, 0, 0]).astype(env.NUMPY_FLOAT)
    p3 = np.array([0, 1, 0]).astype(env.NUMPY_FLOAT) - 10
    p4 = np.array([0.5, 0.5, -1]).astype(env.NUMPY_FLOAT)
    
    improper = get_pbc_dihedral(p1, p2, p3, p4, pbc_matrix, pbc_inv)
    assert improper == pytest.approx(- np.pi / 2)

    p1 = np.array([0, 1, 1]).astype(env.NUMPY_FLOAT)
    p2 = np.array([0, 0, 0]).astype(env.NUMPY_FLOAT) 
    p3 = np.array([1, 0, 0]).astype(env.NUMPY_FLOAT)
    p4 = np.array([1, 0, -1]).astype(env.NUMPY_FLOAT) - 10
    
    dihedral = get_pbc_dihedral(p1, p2, p3, p4, pbc_matrix, pbc_inv)
    assert dihedral == pytest.approx(- np.pi * 3/ 4)