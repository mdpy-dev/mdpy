#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : geometry.py
created time : 2021/10/09
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np
from numpy import arccos

def get_unit_vec(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec

def get_norm_vec(position1, position2, position3):
    position1, position2, position3 = np.array(position1), np.array(position2), np.array(position3)
    v0 = position2 - position1
    v1 = position3 - position1
    
    norm_vec = np.cross(v0, v1)
    
    return get_unit_vec(norm_vec)

def get_bond(position1, position2):
    return np.sqrt(((np.array(position1) - np.array(position2))**2).sum())

def get_angle(position1, position2, position3, is_angular=True):
    position1, position2, position3 = np.array(position1), np.array(position2), np.array(position3)
    v0 = position1 - position2
    v1 = position3 - position2

    cos_phi = np.dot(v0, v1) / (np.linalg.norm(v0)*np.linalg.norm(v1))

    if is_angular:
        return arccos(cos_phi)
    else:
        return arccos(cos_phi) / np.pi * 180

def get_included_angle(vec1, vec2, is_angular=True):
    cos_phi = np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2))
    if is_angular:
        return arccos(cos_phi)
    else:
        return arccos(cos_phi) / np.pi * 180

def get_dihedral(position1, position2, position3, position4, is_angular=True):
    position1, position2 = np.array(position1), np.array(position2)
    position3, position4 = np.array(position3), np.array(position4)
    r1 = position2 - position1
    r2 = position3 - position2
    r3 = position4 - position3

    n1 = np.cross(r1, r2)
    n2 = np.cross(r2, r3)

    x = np.dot(np.linalg.norm(r2) * r1, n2)
    y = np.dot(n1, n2)

    if is_angular:
        return np.arctan2(x, y)
    else:
        return np.arctan2(x, y) / np.pi * 180
