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
    v0 = np.array(position2 - position1)
    v1 = np.array(position3 - position1)
    
    norm_vec = np.cross(v0, v1)
    
    return get_unit_vec(norm_vec)

def get_bond(position1, position2):
    return np.sqrt(((np.array(position1) - np.array(position2))**2).sum())

def get_angle(position1, position2, position3, is_angular=True):
    v0 = np.array(position1 - position2)
    v1 = np.array(position3 - position2)

    cos_phi = np.dot(v0, v1) / (np.linalg.norm(v0)*np.linalg.norm(v1))

    if is_angular:
        return arccos(cos_phi)
    else:
        return arccos(cos_phi) / np.pi * 180

def get_dihedral(position1, position2, position3, position4, is_angular=True):
    v0 = np.array(position1 - position2)
    v1 = np.array(position3 - position2)
    v2 = np.array(position4 - position3)
    
    # Calculate the vertical vector of each plane
    # Note the order of cross product
    na = np.cross(v1, v0)
    nb = np.cross(v1, v2)

    # Note that we delete the absolute value  
    cos_phi = np.dot(na, nb) / (np.linalg.norm(na)*np.linalg.norm(nb))

    # Sign of angle
    omega = np.dot(v0, np.cross(v1, v2))
    sign = omega / np.abs(omega)

    if is_angular:
        return sign * arccos(cos_phi)
    else:
        return sign * arccos(cos_phi) / np.pi * 180