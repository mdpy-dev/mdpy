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
from ..math import *

def test_get_unit_vec():
    vec = [1, 1]
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

def test_get_angle():
    p1 = np.array([1, 1])
    p2 = np.array([0, 0])
    p3 = np.array([0, 1])
    angle = get_angle(p1, p2, p3)
    assert angle == pytest.approx(np.pi / 4)
    
    angle = get_angle(p1, p2, p3, is_angular=False)
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