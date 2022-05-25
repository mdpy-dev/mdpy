#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_constraint.py
created time : 2021/10/09
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import pytest
from mdpy.constraint import Constraint


class TestConstraint:
    def setup(self):
        self.constraint = Constraint()

    def teardown(self):
        self.constraint = None

    def test_attributes(self):
        assert isinstance(self.constraint._constraint_id, int)

    def test_exceptions(self):
        with pytest.raises(NotImplementedError):
            self.constraint.bind_ensemble(1)

        with pytest.raises(NotImplementedError):
            self.constraint.update()
