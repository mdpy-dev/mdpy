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

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, "data/grid_parser/")
file_path = os.path.join(data_dir, "test_grid_writer.grid")


class TestGridParser:
    def setup(self):
        self.grid = Grid(grid_width=0.5, x=[-2.0, 2.0], y=[-2.0, 2.0], z=[-2.0, 2.0])
        self.grid.set_requirement(
            field_name_list=["phi", "epsilon"], constant_name_list=["epsilon0"]
        )
        self.grid.add_field("phi", self.grid.zeros_field())
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
        assert isinstance(grid.field.phi, cp.ndarray)
        for i in range(grid.num_dimensions):
            assert grid.coordinate.x.shape[i] == grid.shape[i]
            assert grid.field.phi.shape[i] == grid.shape[i]
            assert grid.field.epsilon.shape[i] == grid.shape[i]
        for i in range(grid.num_dimensions):
            assert cp.all(cp.isclose(grid.coordinate.x, self.grid.coordinate.x))
            assert cp.all(cp.isclose(grid.field.phi, self.grid.field.phi))
            assert cp.all(cp.isclose(grid.field.epsilon, self.grid.field.epsilon))
        assert grid.constant.epsilon0 == self.grid.constant.epsilon0
        assert isinstance(grid.constant.epsilon0, type(self.grid.constant.epsilon0))


if __name__ == "__main__":
    test = TestGridParser()
    test.setup()
    test.test_parse()
    test.teardown()
