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
            raise FileFormatError("The file should end with .grid suffix")
        self._file_path = file_path
        self._mode = mode
        with h5py.File(self._file_path, self._mode) as f:
            f.create_group("information")
            f.create_group("field")
            f.create_group("constant")

    def write(self, grid: Grid):
        grid.check_requirement()
        self._write_information(grid)
        self._write_field(grid)
        self._write_constant(grid)

    def _write_information(self, grid: Grid):
        with h5py.File(self._file_path, "a") as h5f:
            del h5f["information"]
            h5f.create_group("information")
            h5f["information/requirement"] = str(grid.requirement)
            h5f["information/grid_width"] = grid.grid_width
            h5f["information/coordinate_label"] = grid.coordinate_label
            h5f["information/coordinate_range"] = grid.coordinate_range

    def _write_field(self, grid: Grid):
        attribute = "field"
        with h5py.File(self._file_path, "a") as h5f:
            del h5f[attribute]
            h5f.create_group(attribute)
            sub_grid = getattr(grid, attribute)
            for key in sub_grid.__dict__.keys():
                if not key.startswith("_SubGrid"):
                    h5f["%s/%s" % (attribute, key)] = getattr(sub_grid, key).get()

    def _write_constant(self, grid: Grid):
        attribute = "constant"
        with h5py.File(self._file_path, "a") as h5f:
            del h5f[attribute]
            h5f.create_group(attribute)
            sub_grid = getattr(grid, attribute)
            for key in sub_grid.__dict__.keys():
                if not key.startswith("_SubGrid"):
                    h5f["%s/%s" % (attribute, key)] = getattr(sub_grid, key)
