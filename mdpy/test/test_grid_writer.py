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
        self._grid = Grid(x=[-2, 2, 128], y=[-2, 2, 128], z=[-2, 2, 64])

        self._grid.set_requirement(
            {
                "phi": {"require_gradient": False, "require_curvature": False},
                "epsilon": {"require_gradient": True, "require_curvature": True},
            },
        )
        self._grid.add_field("phi", self._grid.zeros_field())
        self._grid.add_field("epsilon", self._grid.zeros_field())
        self._grid.gradient.epsilon[0, 0, 0, 0] = 1

    def teardown(self):
        del self._grid

    def test_attribute(self):
        pass

    def test_exception(self):
        with pytest.raises(FileFormatError):
            GridWriter("test.gri")

    def test_write(self):
        writer = GridWriter(file_path)
        writer.write(self._grid)