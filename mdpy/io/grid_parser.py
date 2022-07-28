#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : grid_parser.py
created time : 2022/07/28
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import h5py
import ast
import cupy as cp
from mdpy.core import Grid
from mdpy.environment import *
from mdpy.error import *


class GridParser:
    def __init__(self, file_path: str) -> None:
        if not file_path.endswith(".grid"):
            raise FileFormatError("The file should end with .hdf5 suffix")
        self._file_path = file_path
        self._grid = self._parse_grid()

    def _parse_grid(self) -> Grid:
        grid = Grid()
        with h5py.File(self._file_path, "r") as h5f:
            # Information
            grid.set_requirement(
                ast.literal_eval(bytes.decode(h5f["information/requirement"][()]))
            )
            grid._num_dimensions = h5f["information/num_dimensions"][()]
            grid._shape = list(h5f["information/shape"][()])
            grid._inner_shape = list(h5f["information/inner_shape"][()])
            grid._grid_width = h5f["information/grid_width"][()].astype(NUMPY_FLOAT)
            grid._device_grid_width = cp.array(grid.grid_width, CUPY_FLOAT)
            # Coordinate
            self._parse_attribute(h5f, grid, "coordinate")
            # Field
            self._parse_attribute(h5f, grid, "field")
            # Gradient
            self._parse_attribute(h5f, grid, "gradient")
            # Curvature
            self._parse_attribute(h5f, grid, "curvature")
        grid.check_requirement()
        return grid

    def _parse_attribute(self, handle: h5py.File, grid: Grid, attribute: str):
        sub_grid = getattr(grid, attribute)
        for key in handle[attribute].keys():
            setattr(
                sub_grid,
                key,
                cp.array(handle["%s/%s" % (attribute, key)][()], CUPY_FLOAT),
            )

    @property
    def grid(self) -> Grid:
        return self._grid
