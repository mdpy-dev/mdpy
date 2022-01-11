#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_minimizer.py
created time : 2022/01/09
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest
from ..core import Topology
from ..ensemble import Ensemble
from ..minimizer import Minimizer

class TestMinimizer:
    def setup(self):
        self.ensemble = Ensemble(Topology())
        self.minimizer = Minimizer()

    def teardown(self):
        self.minimizer = None

    def test_attributes(self):
        pass

    def test_exceptions(self):
        with pytest.raises(NotImplementedError):
            self.minimizer.minimize(self.ensemble)