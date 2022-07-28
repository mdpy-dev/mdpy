#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_grid.py
created time : 2022/07/28
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import pytest
from mdpy.core import Grid
from mdpy.environment import *
from mdpy.error import *


class TestGrid:
    def setup(self):
        self.grid = Grid(x=[-2, 2, 128], y=[-2, 2, 128], z=[-2, 2, 64])

        self.grid.set_requirement(
            {
                "phi": {"require_gradient": False, "require_curvature": False},
                "epsilon": {"require_gradient": True, "require_curvature": True},
            },
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
        assert not hasattr(self.grid.gradient, "phi")
        assert not hasattr(self.grid.curvature, "phi")

        self.grid.add_field("epsilon", self.grid.ones_field())
        assert hasattr(self.grid.field, "epsilon")
        assert hasattr(self.grid.gradient, "epsilon")
        assert hasattr(self.grid.curvature, "epsilon")

    def test_check_requirement(self):
        self.grid.add_field("phi", self.grid.ones_field())
        self.grid.add_field("epsilon", self.grid.ones_field())
        self.grid.check_requirement()

        del self.grid.curvature.epsilon
        with pytest.raises(GridPoorDefinedError):
            self.grid.check_requirement()
