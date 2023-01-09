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
from mdpy.core.grid import Variable


class TestVariable:
    def setup(self):
        self.variable = Variable()

    def teardown(self):
        del self.variable

    def test_attribute(self):
        assert not self.variable.value is None
        assert not self.variable.boundary is None

    def test_exception(self):
        pass

    def test_add_boundary(self):
        boundary_type = "d"
        boundary_data = {
            "index": cp.array([[1, 2, 3]], CUPY_INT),
            "value": cp.array([1], CUPY_FLOAT),
        }
        self.variable.add_boundary(
            boundary_type=boundary_type, boundary_data=boundary_data
        )
        assert self.variable.boundary["d"]["index"][0, 0] == 1
        self.variable.add_boundary(
            boundary_type=boundary_type, boundary_data=boundary_data
        )
        assert self.variable.boundary["d"]["index"][1, 0] == 1
        assert self.variable.boundary["d"]["index"].shape[0] == 2
        assert self.variable.boundary["d"]["value"].shape[0] == 2


class TestGrid:
    def setup(self):
        self.grid = Grid(grid_width=0.1, x=[-2, 2], y=[-2, 2], z=[-2, 2])
        self.grid.set_requirement(
            variable_name_list=["phi"],
            field_name_list=["epsilon"],
            constant_name_list=["epsilon0"],
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

        with pytest.raises(ArrayDimError):
            self.grid.add_field("a", self.grid.zeros_field()[:-1, :, :])

        with pytest.raises(ArrayDimError):
            variable = self.grid.empty_variable()
            variable.value = variable.value[:-1, :, :]
            self.grid.add_variable("b", variable)

    def test_add_requirement(self):
        self.grid.add_requirement("field", "a")
        assert self.grid.requirement["field"][1] == "a"
        assert len(self.grid.requirement["field"]) == 2
        self.grid.add_requirement("field", "a")
        assert len(self.grid.requirement["field"]) == 2

    def test_add_variable(self):
        self.grid.add_variable("phi", self.grid.empty_variable())
        assert hasattr(self.grid.variable, "phi")

    def test_add_field(self):
        self.grid.add_field("epsilon", self.grid.ones_field())
        assert hasattr(self.grid.field, "epsilon")

    def test_add_constant(self):
        self.grid.add_constant("epsilon0", 10)
        assert hasattr(self.grid.constant, "epsilon0")
        assert isinstance(self.grid.constant.epsilon0, NUMPY_FLOAT)

    def test_check_requirement(self):
        self.grid.add_variable("phi", self.grid.empty_variable())
        self.grid.add_field("epsilon", self.grid.ones_field())
        with pytest.raises(GridPoorDefinedError):
            self.grid.check_requirement()

        self.grid.add_constant("epsilon0", 10)
        self.grid.check_requirement()

        with pytest.raises(GridPoorDefinedError):
            self.grid.add_requirement("field", "a")
            self.grid.check_requirement()
