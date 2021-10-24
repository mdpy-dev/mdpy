#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_constraint.py
created time : 2021/10/09
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest
from ..constraint import Constraint

class TestConstraint:
    def setup(self):
        self.constraint = Constraint(0, 0)
    
    def teardown(self):
        self.constraint = None

    def test_attributes(self):
        assert isinstance(self.constraint._force_id, int)
        assert isinstance(self.constraint._force_group, int)

    def test_exceptions(self):
        with pytest.raises(NotImplementedError):
            self.constraint.bind_ensemble(1)
        
        with pytest.raises(NotImplementedError):
            self.constraint.update()