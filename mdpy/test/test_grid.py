#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_grid.py
created time : 2022/07/28
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import pytest
import cupy as cp
from mdpy.core import Grid
from mdpy.environment import *
from mdpy.error import *


class TestGrid:
    def setup(self):
        self.grid = Grid(grid_width=0.1, x=[-2, 2], y=[-2, 2], z=[-2, 2])
        self.grid.set_requirement(
            field_name_list=["phi", "epsilon"], constant_name_list=["epsilon0"]
        )

    def teardown(self):
        del self.grid

    def test_attribute(self):
        assert hasattr(self.grid.coordinate, "x")
        assert hasattr(self.grid.coordinate, "y")
        assert hasattr(self.grid.coordinate, "z")

    def test_exception(self):
        with pytest.raises(GridPoorDefinedError):
            self.grid.check_requirement()

    def test_add_field(self):
        self.grid.add_field("phi", self.grid.ones_field())
        assert hasattr(self.grid.field, "phi")

        self.grid.add_field("epsilon", self.grid.ones_field())
        assert hasattr(self.grid.field, "epsilon")

    def test_add_constant(self):
        self.grid.add_constant("epsilon0", 10)
        assert hasattr(self.grid.constant, "epsilon0")
        assert isinstance(self.grid.constant.epsilon0, NUMPY_FLOAT)

    def test_check_requirement(self):
        self.grid.add_field("phi", self.grid.ones_field())
        self.grid.add_field("epsilon", self.grid.ones_field())
        with pytest.raises(GridPoorDefinedError):
            self.grid.check_requirement()

        self.grid.add_constant("epsilon0", 10)
        self.grid.check_requirement()
