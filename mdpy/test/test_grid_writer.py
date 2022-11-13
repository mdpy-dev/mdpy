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
from mdpy.core import Grid
from mdpy.error import FileFormatError
from mdpy.io import GridWriter

cur_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(cur_dir, "out")
file_path = os.path.join(out_dir, "test_grid_writer.grid")


class TestGridWriter:
    def setup(self):
        self.grid = Grid(grid_width=0.5, x=[-2, 2], y=[-2, 2], z=[-2, 2])
        self.grid.set_requirement(
            field_name_list=["phi", "epsilon"], constant_name_list=["epsilon0"]
        )
        self.grid.add_field("phi", self.grid.zeros_field())
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
