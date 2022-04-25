#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_integrator.py
created time : 2021/10/18
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import pytest
from mdpy.integrator import Integrator
from mdpy.unit import *

class TestIntegrator:
    def setup(self):
        pass

    def teardown(self):
        pass

    def test_attributes(self):
        integrator = Integrator(1, 10)
        assert integrator.time_step == 1

        integrator = Integrator(Quantity(1, nanosecond), 10)
        assert integrator.time_step == 1e6

        assert integrator.is_cached == False
        integrator._cur_positions = 1
        assert integrator.is_cached == True

    def test_exceptions(self):
        integrator = Integrator(1, 10)
        with pytest.raises(NotImplementedError):
            integrator.integrate(0, 1)