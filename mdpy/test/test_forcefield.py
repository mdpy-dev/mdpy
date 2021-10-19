#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_forcefield.py
created time : 2021/10/16
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest
from ..core import Topology
from ..forcefield import Forcefield

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
            forcefield.check_params()
        
        with pytest.raises(NotImplementedError):
            forcefield.create_ensemble()