#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_precision.py
created time : 2026/07/08
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import numpy as np
import pytest

from mdpy.precision import precision, PrecisionConfig
from mdpy.error import PrecisionError


def test_default_precision_is_single():
    assert precision.precision == 'SINGLE'


def test_default_float_dtype():
    assert precision.FLOAT == np.float32


def test_default_int_dtype():
    assert precision.INT == np.int32


def test_unit_float_is_float128():
    assert precision.UNIT_FLOAT == np.float128


def test_set_precision_double_switches_dtypes():
    config = PrecisionConfig()
    config.set_precision('DOUBLE')
    assert config.precision == 'DOUBLE'
    assert config.FLOAT == np.float64
    assert config.INT == np.int64


def test_set_precision_is_case_insensitive():
    config = PrecisionConfig()
    config.set_precision('double')
    assert config.precision == 'DOUBLE'


def test_set_precision_invalid_raises_precision_error():
    config = PrecisionConfig()
    with pytest.raises(PrecisionError):
        config.set_precision('QUAD')


def test_unit_float_independent_of_precision():
    config = PrecisionConfig()
    config.set_precision('DOUBLE')
    assert config.UNIT_FLOAT == np.float128


def test_supported_precisions_property():
    config = PrecisionConfig()
    assert config.supported_precisions == ['SINGLE', 'DOUBLE']
