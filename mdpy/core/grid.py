#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : grid.py
created time : 2022/07/18
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import cupy as cp
import numpy as np
from mdpy.environment import *
from mdpy.error import *


class SubGrid:
    def __init__(self, name: str) -> None:
        self.__name = name

    def __getattribute__(self, __name: str):
        try:
            return object.__getattribute__(self, __name)
        except:
            raise AttributeError(
                "Grid.%s.%s has not been defined, please check Grid.requirement"
                % (self.__name, __name)
            )


class Grid:
    def __init__(self, grid_width: float, **coordinate_range) -> None:
        # Input
        self._grid_width = NUMPY_FLOAT(grid_width)
        # Initialize attributes
        self._coordinate = SubGrid("coordinate")
        self._field = SubGrid("field")
        self._constant = SubGrid("constant")
        # Set grid information and coordinate
        self._coordinate_label = list(coordinate_range.keys())
        self._coordinate_range = np.array(list(coordinate_range.values()), NUMPY_FLOAT)
        grid = [
            cp.arange(
                start=value[0],
                stop=value[1] + grid_width,
                step=grid_width,
                dtype=CUPY_FLOAT,
            )
            for value in coordinate_range.values()
        ]
        grid = cp.meshgrid(*grid, indexing="ij")
        self._shape = list(grid[0].shape)
        self._inner_shape = [i - 2 for i in self._shape]
        for index, key in enumerate(self._coordinate_label):
            setattr(
                self._coordinate,
                key,
                grid[index],
            )
        self._num_dimensions = len(self._coordinate_label)
        # Initialize requirement
        self._requirement = {"field": [], "constant": []}

    def set_requirement(
        self, field_name_list: list[str], constant_name_list: list[str]
    ):
        self._requirement["field"] = field_name_list
        self._requirement["constant"] = constant_name_list

    def check_requirement(self):
        is_all_set = True
        exception = "Gird is not all set:\n"
        for key in self._requirement["constant"]:
            is_all_set &= hasattr(self._constant, key)
            exception += "- grid.constant.%s: %s;\n" % (
                key,
                hasattr(self._constant, key),
            )
        for key in self._requirement["field"]:
            is_all_set &= hasattr(self._field, key)
            exception += "- grid.field.%s: %s;\n" % (key, hasattr(self._field, key))
        if not is_all_set:
            raise GridPoorDefinedError(exception[:-1])

    def _check_shape(self, value: cp.ndarray, target_shape: list[int]):
        shape = value.shape
        exception = (
            "Require Array with shape %s, while array with shape %s is provided"
            % (
                tuple(target_shape),
                shape,
            )
        )
        if len(shape) != self._num_dimensions:
            raise ArrayDimError(exception)
        for dim1, dim2 in zip(shape, target_shape):
            if dim1 != dim2:
                raise ArrayDimError(exception)

    def add_field(self, name: str, value: cp.ndarray):
        # Set field
        self._check_shape(value, self._shape)
        setattr(self._field, name, value)

    def add_constant(self, name: str, value: float):
        setattr(self._constant, name, NUMPY_FLOAT(value))

    def zeros_field(self, dtype=CUPY_FLOAT):
        return cp.zeros(self._shape, dtype)

    def ones_field(self, dtype=CUPY_FLOAT):
        return cp.ones(self._shape, dtype)

    @property
    def coordinate_label(self) -> list[str]:
        return self._coordinate_label

    @property
    def coordinate_range(self) -> np.ndarray:
        return self._coordinate_range

    @property
    def requirement(self) -> dict:
        return self._requirement

    @property
    def num_dimensions(self) -> int:
        return self._num_dimensions

    @property
    def shape(self) -> list[int]:
        return self._shape

    @property
    def device_shape(self) -> cp.ndarray:
        return cp.array(self._shape, CUPY_INT)

    @property
    def inner_shape(self) -> list[int]:
        return self._inner_shape

    @property
    def device_inner_shape(self) -> cp.ndarray:
        return cp.array(self._inner_shape, CUPY_INT)

    @property
    def grid_width(self) -> np.ndarray:
        return self._grid_width

    @property
    def coordinate(self) -> object:
        return self._coordinate

    @property
    def field(self) -> object:
        return self._field

    @property
    def constant(self) -> object:
        return self._constant


if __name__ == "__main__":
    grid = Grid(grid_width=0.1, x=[-2, 2], y=[-2, 2], z=[-2, 2])
    grid.set_requirement(
        field_name_list=["phi", "epsilon"], constant_name_list=["epsilon0"]
    )
    print(grid.requirement)
    phi = grid.zeros_field()
    epsilon = grid.zeros_field() + 1

    phi[0, :, :] = 20
    phi[-1, :, :] = 0
    grid.add_field("phi", phi)
    grid.add_field("epsilon", epsilon)
    grid.add_constant("epsilon0", 10)
    grid.check_requirement()
    print(grid.coordinate_range)
