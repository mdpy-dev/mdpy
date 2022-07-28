#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : grid_writer.py
created time : 2022/07/28
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import h5py
from mdpy.core import Grid
from mdpy.error import *


class GridWriter:
    def __init__(self, file_path: str, mode: str = "w") -> None:
        if not file_path.endswith(".grid"):
            raise FileFormatError("The file should end with .hdf5 suffix")
        self._file_path = file_path
        self._mode = mode
        with h5py.File(self._file_path, self._mode) as f:
            f.create_group("information")
            f.create_group("coordinate")
            f.create_group("field")
            f.create_group("gradient")
            f.create_group("curvature")

    def write(self, grid: Grid):
        grid.check_requirement()
        self._write_information(grid)
        self._write_attribute(grid, "coordinate")
        self._write_attribute(grid, "field")
        self._write_attribute(grid, "gradient")
        self._write_attribute(grid, "curvature")

    def _write_information(self, grid: Grid):
        with h5py.File(self._file_path, "a") as h5f:
            del h5f["information"]
            h5f.create_group("information")
            h5f["information/requirement"] = str(grid.requirement)
            h5f["information/num_dimensions"] = grid.num_dimensions
            h5f["information/shape"] = grid.shape
            h5f["information/inner_shape"] = grid.inner_shape
            h5f["information/grid_width"] = grid.grid_width

    def _write_attribute(self, grid: Grid, attribute: str):
        with h5py.File(self._file_path, "a") as h5f:
            del h5f[attribute]
            h5f.create_group(attribute)
            sub_grid = getattr(grid, attribute)
            for key in sub_grid.__dict__:
                if not key.startswith("_SubGrid"):
                    h5f["%s/%s" % (attribute, key)] = getattr(sub_grid, key).get()
