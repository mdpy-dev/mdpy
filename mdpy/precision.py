#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : precision.py
created time : 2026/07/08
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import numpy as np
from mdpy.error import PrecisionError

class PrecisionConfig:
    def __init__(self) -> None:
        self._supported_precisions = ['SINGLE', 'DOUBLE']
        self._default_precision = 'SINGLE'
        self.UNIT_FLOAT = np.float128
        self.set_precision(self._default_precision)

    def set_precision(self, precision: str):
        precision = precision.upper()
        if precision in self._supported_precisions:
            self._precision = precision
            if precision == 'SINGLE':
                self.FLOAT = np.float32
                self.INT = np.int32
            elif precision == 'DOUBLE':
                self.FLOAT = np.float64
                self.INT = np.int64
        else:
            raise PrecisionError(
                'Precision %s is not supported. ' % precision +
                'Check supported precisions with `mdpy.precision.supported_precisions`'
            )

    @property
    def supported_precisions(self):
        return self._supported_precisions

    @property
    def precision(self):
        return self._precision

precision = PrecisionConfig()
