#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_environment.py
created time : 2021/11/05
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import pytest
import numpy as np
import numba as nb
from mdpy import env
from mdpy.error import *


def test_attributes():
    assert env.precision == "SINGLE"
    assert env.platform == "CUDA"


def test_exceptions():
    with pytest.raises(EnvironmentVariableError):
        env.set_precision("A")

    with pytest.raises(EnvironmentVariableError):
        env.set_platform("A")
