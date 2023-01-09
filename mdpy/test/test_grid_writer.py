#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_grid_writer.py
created time : 2022/07/28
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import os
import pytest
import cupy as cp
from mdpy.core import Grid
from mdpy.error import FileFormatError
from mdpy.io import GridWriter
from mdpy.environment import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(cur_dir, "out")
file_path = os.path.join(out_dir, "test_grid_writer.grid")


class TestGridWriter:
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
        del self.grid

    def test_attribute(self):
        pass

    def test_exception(self):
        with pytest.raises(FileFormatError):
            GridWriter("test.gri")

    def test_write(self):
        writer = GridWriter(file_path)
        writer.write(self.grid)
