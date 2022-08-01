#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_device.py
created time : 2022/08/01
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import pytest
import cupy as cp
from mdpy.device import *
from mdpy.error import *


class TestDevice:
    def setup(self):
        pass

    def teardown(self):
        pass

    def test_attribute(self):
        pass

    def test_exception(self):
        with pytest.raises(DevicePoorDefinedError):
            Device(get_device_count())

    def test_device(self):
        with Device(0):
            a = cp.ones([10, 2])
            assert a.device.id == 0
