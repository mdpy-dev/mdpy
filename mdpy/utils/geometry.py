#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : geometry.py
created time : 2021/10/09
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import numpy as np
import numba as nb
from numpy import arccos
from numpy import sin, cos
from mdpy import env, SPATIAL_DIM
from mdpy.utils.pbc import *

@nb.njit()
def get_unit_vec(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec

def get_norm_vec(p1, p2, p3):
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    v0 = p2 - p1
    v1 = p3 - p1
    norm_vec = np.cross(v0, v1).astype(env.NUMPY_FLOAT)
    return get_unit_vec(norm_vec)

def get_bond(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_pbc_bond(p1, p2, pbc_matrix, pbc_inv):
    bond_vec = unwrap_vec(np.array(p1) - np.array(p2), pbc_matrix, pbc_inv)
    return np.linalg.norm(bond_vec)

def get_angle(p1, p2, p3, is_angular=True):
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    v0 = p1 - p2
    v1 = p3 - p2
    cos_phi = np.dot(v0, v1) / (np.linalg.norm(v0)*np.linalg.norm(v1))
    if is_angular:
        return arccos(cos_phi)
    else:
        return arccos(cos_phi) / np.pi * 180

@nb.njit()
def get_pbc_angle(p1, p2, p3, pbc_matrix, pbc_inv):
    v0 = unwrap_vec(p1 - p2, pbc_matrix, pbc_inv)
    v1 = unwrap_vec(p3 - p2, pbc_matrix, pbc_inv)
    cos_phi = np.dot(v0, v1) / (np.linalg.norm(v0)*np.linalg.norm(v1))
    return arccos(cos_phi)

def get_included_angle(vec1, vec2, is_angular=True):
    cos_phi = np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2))
    if is_angular:
        return arccos(cos_phi)
    else:
        return arccos(cos_phi) / np.pi * 180

def get_dihedral(p1, p2, p3, p4, is_angular=True):
    p1, p2 = np.array(p1), np.array(p2)
    p3, p4 = np.array(p3), np.array(p4)
    r1 = p2 - p1
    r2 = p3 - p2
    r3 = p4 - p3

    n1 = np.cross(r1, r2)
    n2 = np.cross(r2, r3)

    x = np.dot(np.linalg.norm(r2) * r1, n2)
    y = np.dot(n1, n2)

    if is_angular:
        return np.arctan2(x, y)
    else:
        return np.arctan2(x, y) / np.pi * 180

@nb.njit()
def get_pbc_dihedral(p1, p2, p3, p4, pbc_matrix, pbc_inv):
    r1 = unwrap_vec(p2 - p1, pbc_matrix, pbc_inv)
    r2 = unwrap_vec(p3 - p2, pbc_matrix, pbc_inv)
    r3 = unwrap_vec(p4 - p3, pbc_matrix, pbc_inv)

    n1 = np.cross(r1, r2)
    n2 = np.cross(r2, r3)

    x = np.dot(np.linalg.norm(r2) * r1, n2)
    y = np.dot(n1, n2)

    return np.arctan2(x, y)

def generate_rotation_matrix(yaw, pitch, roll):
    shape = [SPATIAL_DIM, SPATIAL_DIM]
    Rx, Ry, Rz = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    Rx[0, 0], Rx[1, 1], Rx[2, 2] = 1, cos(roll), cos(roll)
    Rx[1, 2], Rx[2, 1] = -sin(roll), sin(roll)
    Ry[0, 0], Ry[1, 1], Ry[2, 2] = cos(pitch), 1, cos(pitch)
    Ry[0, 2], Ry[2, 0] = sin(pitch), -sin(pitch)
    Rz[0, 0], Rz[1, 1], Rz[2, 2] = cos(yaw), cos(yaw), 1
    Rz[0, 1], Rz[1, 0] = -sin(yaw), sin(yaw)
    rotation = Rz.dot(Ry).dot(Rx).T
    return rotation