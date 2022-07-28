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
from mdpy.error import FileFormatError
from mdpy.io import GridParser

cur_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(cur_dir, "out")
file_path = os.path.join(out_dir, "test_grid_writer.grid")


class TestGridParser:
    def setup(self):
        pass

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
        assert grid.num_dimensions == 3
        assert isinstance(grid.field.phi, cp.ndarray)
        for i in range(grid.num_dimensions):
            assert grid.coordinate.x.shape[i] == grid.shape[i]
            assert grid.field.phi.shape[i] == grid.shape[i]
            assert grid.field.epsilon.shape[i] == grid.shape[i]
        assert grid.gradient.epsilon.shape[0] == grid.num_dimensions
        assert grid.curvature.epsilon.shape[0] == grid.num_dimensions
        for i in range(grid.num_dimensions):
            assert grid.gradient.epsilon.shape[i + 1] == grid.inner_shape[i]
            assert grid.curvature.epsilon.shape[i + 1] == grid.inner_shape[i]
        assert grid.gradient.epsilon[0, 0, 0, 0] == 1


if __name__ == "__main__":
    test = TestGridParser()
    test.setup()
    test.test_parse()
    test.teardown()
