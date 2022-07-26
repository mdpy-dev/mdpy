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
from mdpy.error import *

nb.config.CUDA_ARRAY_INTERFACE_SYNC = False
precision = "single"

CUPY_BIT = cp.uint32
NUMBA_BIT = nb.uint32
NUMPY_BIT = np.uint32

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


class MDPYEnvironment:
    def __init__(self) -> None:
        self._supported_precisions = ["SINGLE", "DOUBLE"]
        self._default_precisions = "SINGLE"
        self._supproted_platforms = ["CPU", "CUDA"]
        self._default_platforms = "CUDA"
        self.set_precision(self._default_precisions)
        self.set_platform(self._default_platforms)

    def set_precision(self, precision: str):
        precision = precision.upper()
        if precision in self._supported_precisions:
            self._precision = precision
            if precision == "SINGLE":
                self.NUMPY_FLOAT = np.float32
                self.NUMBA_FLOAT = nb.float32
                self.NUMPY_INT = np.int32
                self.NUMBA_INT = nb.int32
            elif precision == "DOUBLE":
                self.NUMPY_FLOAT = np.float64
                self.NUMBA_FLOAT = nb.float64
                self.NUMPY_INT = np.int64
                self.NUMBA_INT = nb.int64
            self.UNIT_FLOAT = np.float128
        else:
            raise EnvironmentVariableError(
                "Precision %s is not supported. " % precision
                + "Check supported precision with `mdpy.env.supported_precisions`"
            )

    def set_platform(self, platform: str):
        platform = platform.upper()
        if platform in self._supproted_platforms:
            self._platform = platform
        else:
            raise EnvironmentVariableError(
                "Platform %s is not supported. " % platform
                + "Check supported platform with `mdpy.env.supported_platforms`"
            )

    def set_default(self):
        self.set_precision(self._default_precisions)
        self.set_platform(self._default_platforms)

    @property
    def supported_presisions(self):
        return self._supported_precisions

    @property
    def default_presisions(self):
        return self._default_precisions

    @property
    def precision(self):
        return self._precision

    @property
    def supported_platforms(self):
        return self._supproted_platforms

    @property
    def default_platform(self):
        return self._default_platforms

    @property
    def platform(self):
        return self._platform


env = MDPYEnvironment()
