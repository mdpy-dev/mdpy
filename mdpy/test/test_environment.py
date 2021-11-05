#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_environment.py
created time : 2021/11/05
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest
import numpy as np
import numba as nb
from .. import env
from ..error import *

def test_attributes():
    assert env.precision == 'SINGLE'
    assert env.platform == 'CPU'

def test_exceptions():
    with pytest.raises(EnvironmentVariableError):
        env.set_precision('A')

    with pytest.raises(EnvironmentVariableError):
        env.set_platform('A')