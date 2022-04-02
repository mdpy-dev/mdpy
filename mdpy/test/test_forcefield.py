#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_forcefield.py
created time : 2021/10/16
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import pytest
from mdpy.core import Topology
from mdpy.forcefield import Forcefield

class TestForcefield:
    def setup(self):
        self.topology = Topology()

    def teardown(self):
        pass

    def test_attributes(self):
        pass

    def test_exceptions(self):
        forcefield = Forcefield(self.topology)

        with pytest.raises(NotImplementedError):
            forcefield.set_param_files()

        with pytest.raises(NotImplementedError):
            forcefield.check_parameters()

        with pytest.raises(NotImplementedError):
            forcefield.create_ensemble()