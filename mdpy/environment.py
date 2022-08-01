#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : environment.py
created time : 2021/11/05
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import cupy as cp
import numpy as np
import numba as nb
import numba.cuda as cuda
from mdpy.error import *

nb.config.CUDA_ARRAY_INTERFACE_SYNC = False
precision = "single"

CUPY_BIT = cp.uint32
NUMBA_BIT = nb.uint32
NUMPY_BIT = np.uint32
UNIT_FLOAT = np.float128
if precision == "single":
    CUPY_FLOAT = cp.float32
    NUMBA_FLOAT = nb.float32
    NUMPY_FLOAT = np.float32
    CUPY_INT = cp.int32
    NUMBA_INT = nb.int32
    NUMPY_INT = np.int32
    CUPY_UINT = cp.uint32
    NUMBA_UINT = nb.uint32
    NUMPY_UINT = np.uint32
elif precision == "double":
    CUPY_FLOAT = cp.float64
    NUMBA_FLOAT = nb.float64
    NUMPY_FLOAT = np.float64
    CUPY_INT = cp.int64
    NUMBA_INT = nb.int64
    NUMPY_INT = np.int64
    CUPY_UINT = cp.uint64
    NUMBA_UINT = nb.uint64
    NUMPY_UINT = np.uint64
