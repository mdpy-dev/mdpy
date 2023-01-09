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
from mdpy.core.grid import Grid, Variable
from mdpy.environment import *
from mdpy.error import *


class GridParser:
    def __init__(self, file_path: str) -> None:
        if not file_path.endswith(".grid"):
            raise FileFormatError("The file should end with .grid suffix")
        self._file_path = file_path
        self._grid = self._parse_grid()

    def _parse_grid(self) -> Grid:
        with h5py.File(self._file_path, "r") as h5f:
            grid_width = h5f["information/grid_width"][()]
            coordinate_label = [
                bytes.decode(i) for i in h5f["information/coordinate_label"][()]
            ]
            coordinate_range = h5f["information/coordinate_range"][()]
            coordinate_dict = {}
            for index, label in enumerate(coordinate_label):
                coordinate_dict[label] = list(coordinate_range[index, :])
            grid = Grid(grid_width=grid_width, **coordinate_dict)
            # Information
            requirement = ast.literal_eval(
                bytes.decode(h5f["information/requirement"][()])
            )
            grid.set_requirement(
                variable_name_list=requirement["variable"],
                field_name_list=requirement["field"],
                constant_name_list=requirement["constant"],
            )
            # Variable
            self._parse_variable(h5f, grid)
            # Field
            self._parse_field(h5f, grid)
            # Constant
            self._parse_constant(h5f, grid)
        grid.check_requirement()
        return grid

    def _parse_variable(self, handle: h5py.File, grid: Grid):
        sub_grid = getattr(grid, "variable")
        for name in handle["variable"].keys():
            group_name = "variable/%s/" % (name)
            variable = grid.empty_variable()
            # Value
            variable.value = cp.array(handle[group_name + "value"][()], CUPY_FLOAT)
            # Boundary
            group_name += "boundary/"  # /variable/name/boundary/
            for boundary_type in handle[group_name].keys():
                boundary_data = {}
                boundary_group_name = group_name + boundary_type + "/"
                for key in handle[boundary_group_name].keys():
                    val = handle[boundary_group_name + key][()]
                    boundary_data[key] = cp.array(val, val.dtype)
                variable.add_boundary(
                    boundary_type=boundary_type, boundary_data=boundary_data
                )
            setattr(
                sub_grid,
                name,
                variable,
            )

    def _parse_field(self, handle: h5py.File, grid: Grid):
        attribute = "field"
        sub_grid = getattr(grid, attribute)
        for key in handle[attribute].keys():
            setattr(
                sub_grid,
                key,
                cp.array(handle["%s/%s" % (attribute, key)][()], CUPY_FLOAT),
            )

    def _parse_constant(self, handle: h5py.File, grid: Grid):
        attribute = "constant"
        sub_grid = getattr(grid, attribute)
        for key in handle[attribute].keys():
            setattr(
                sub_grid,
                key,
                NUMPY_FLOAT(handle["%s/%s" % (attribute, key)][()]),
            )

    @property
    def grid(self) -> Grid:
        return self._grid
