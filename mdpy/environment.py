#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : environment.py
created time : 2021/11/05
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import numpy as np
from mdpy.error import *

class MDPYEnvironment:
    def __init__(self) -> None:
        self._supported_precisions = ['SINGLE', 'DOUBLE']
        self._default_precisions = 'SINGLE'
        self._platform = 'CUDA'
        self.set_precision(self._default_precisions)

    def set_precision(self, precision: str):
        precision = precision.upper()
        if precision in self._supported_precisions:
            self._precision = precision
            if precision == 'SINGLE':
                self.NUMPY_FLOAT = np.float32
                self.NUMPY_INT = np.int32
            elif precision == 'DOUBLE':
                self.NUMPY_FLOAT = np.float64
                self.NUMPY_INT = np.int64
            self.UNIT_FLOAT = np.float128
        else:
            raise EnvironmentVariableError(
                'Precision %s is not supported. ' %precision +
                'Check supported precision with `mdpy.env.supported_precisions`'
            )

    @property
    def precision(self):
        return self._precision

env = MDPYEnvironment()
