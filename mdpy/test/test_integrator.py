#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_integrator.py
created time : 2021/10/18
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest
from ..integrator import Integrator
from ..unit import *

class TestIntegrator:
    def setup(self):
        pass

    def teardown(self):
        pass

    def test_attributes(self):
        integrator = Integrator(1)
        assert integrator.time_step == 1

        integrator = Integrator(Quantity(1, nanosecond))
        assert integrator.time_step == 1e6

        assert integrator.is_cached == False
        integrator._cur_positions = 1
        assert integrator.is_cached == True

    def test_exceptions(self):
        integrator = Integrator(1)
        with pytest.raises(NotImplementedError):
            integrator.integrate(0, 1)