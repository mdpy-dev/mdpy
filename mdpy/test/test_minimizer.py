#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_minimizer.py
created time : 2022/01/09
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import pytest
import numpy as np
from mdpy.core import Topology , Ensemble
from mdpy.minimizer import Minimizer

class TestMinimizer:
    def setup(self):
        self.ensemble = Ensemble(Topology(), np.diag([30]*3))
        self.minimizer = Minimizer()

    def teardown(self):
        self.minimizer = None

    def test_attributes(self):
        pass

    def test_exceptions(self):
        with pytest.raises(NotImplementedError):
            self.minimizer.minimize(self.ensemble)