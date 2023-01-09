#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_grid_parser.py
created time : 2022/07/28
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import os
import pytest
import cupy as cp
from mdpy.core import Grid
from mdpy.error import FileFormatError
from mdpy.io import GridParser
from mdpy.environment import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, "data/grid_parser/")
file_path = os.path.join(data_dir, "test_grid_writer.grid")


class TestGridParser:
    def setup(self):
        self.grid = Grid(grid_width=0.5, x=[-2, 2], y=[-2, 2], z=[-2, 2])
        self.grid.set_requirement(
            variable_name_list=["phi"],
            field_name_list=["epsilon"],
            constant_name_list=["epsilon0"],
        )
        phi = self.grid.empty_variable()
        boundary_type = "d"
        boundary_data = {
            "index": cp.array([[1, 2, 3]], CUPY_INT),
            "value": cp.array([1], CUPY_FLOAT),
        }
        phi.add_boundary(boundary_type=boundary_type, boundary_data=boundary_data)
        self.grid.add_variable("phi", phi)
        self.grid.add_field("epsilon", self.grid.zeros_field())
        self.grid.add_constant("epsilon0", 10)

    def teardown(self):
        pass

    def test_attribute(self):
        pass

    def test_exception(self):
        with pytest.raises(FileFormatError):
            GridParser("test.gri")

    def test_parse(self):
        parser = GridParser(file_path)
        grid = parser.grid
        grid.check_requirement()
        assert grid.num_dimensions == self.grid.num_dimensions
        assert isinstance(grid.variable.phi.value, cp.ndarray)
        assert isinstance(grid.field.epsilon, cp.ndarray)
        for i in range(grid.num_dimensions):
            assert grid.coordinate.x.shape[i] == grid.shape[i]
            assert grid.variable.phi.value.shape[i] == grid.shape[i]
            assert grid.field.epsilon.shape[i] == grid.shape[i]
        assert cp.all(cp.isclose(grid.coordinate.x, self.grid.coordinate.x))
        assert cp.all(cp.isclose(grid.variable.phi.value, self.grid.variable.phi.value))
        assert (
            grid.variable.phi.boundary["d"]["index"].dtype
            == self.grid.variable.phi.boundary["d"]["index"].dtype
        )
        assert cp.all(
            cp.isclose(
                grid.variable.phi.boundary["d"]["index"],
                self.grid.variable.phi.boundary["d"]["index"],
            )
        )
        assert (
            grid.variable.phi.boundary["d"]["value"].dtype
            == self.grid.variable.phi.boundary["d"]["value"].dtype
        )
        assert cp.all(
            cp.isclose(
                grid.variable.phi.boundary["d"]["value"],
                self.grid.variable.phi.boundary["d"]["value"],
            )
        )
        assert cp.all(cp.isclose(grid.field.epsilon, self.grid.field.epsilon))
        assert grid.constant.epsilon0 == self.grid.constant.epsilon0
        assert isinstance(grid.constant.epsilon0, type(self.grid.constant.epsilon0))


if __name__ == "__main__":
    test = TestGridParser()
    test.setup()
    test.test_parse()
    test.teardown()
