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


class Variable:
    def __init__(self) -> None:
        self._value: cp.ndarray = 0
        self._boundary_index: cp.ndarray = 0
        self._boundary_type: cp.ndarray = 0
        self._boundary_value: cp.ndarray = 0

    @property
    def value(self) -> cp.ndarray:
        return self._value

    @value.setter
    def value(self, value: cp.ndarray):
        try:
            self._value = cp.array(value, CUPY_FLOAT)
        except:
            raise TypeError(
                "numpy.ndarray or cupy.ndarray required, while %s provided"
                % type(value)
            )

    @property
    def boundary_index(self) -> cp.ndarray:
        return self._boundary_index

    @boundary_index.setter
    def boundary_index(self, value):
        try:
            self._boundary_index = cp.array(value, CUPY_INT)
        except:
            raise TypeError(
                "numpy.ndarray or cupy.ndarray required, while %s provided"
                % type(value)
            )

    @property
    def boundary_type(self) -> cp.ndarray:
        return self._boundary_type

    @boundary_type.setter
    def boundary_type(self, value):
        try:
            self._boundary_type = cp.array(value, CUPY_INT)
        except:
            raise TypeError(
                "numpy.ndarray or cupy.ndarray required, while %s provided"
                % type(value)
            )

    @property
    def boundary_value(self) -> cp.ndarray:
        return self._boundary_value

    @boundary_value.setter
    def boundary_value(self, value):
        try:
            self._boundary_value = cp.array(value, CUPY_FLOAT)
        except:
            raise TypeError(
                "numpy.ndarray or cupy.ndarray required, while %s provided"
                % type(value)
            )


class Grid:
    def __init__(self, grid_width: float, **coordinate_range) -> None:
        # Input
        self._grid_width = NUMPY_FLOAT(grid_width)
        # Initialize attributes
        self._coordinate = SubGrid("coordinate")
        self._variable = SubGrid("variable")
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
        self._requirement = {"variable": [], "field": [], "constant": []}

    def set_requirement(
        self,
        variable_name_list: list[str],
        field_name_list: list[str],
        constant_name_list: list[str],
    ):
        self._requirement["variable"] = variable_name_list
        self._requirement["field"] = field_name_list
        self._requirement["constant"] = constant_name_list

    def check_requirement(self):
        is_all_set = True
        exception = "Gird is not all set:\n"
        exception += "variable:\n"
        for key in self._requirement["variable"]:
            is_all_set &= hasattr(self._variable, key)
            exception += "- grid.variable.%s: %s;\n" % (
                key,
                hasattr(self._variable, key),
            )
        exception += "\nconstant:\n"
        for key in self._requirement["constant"]:
            is_all_set &= hasattr(self._constant, key)
            exception += "- grid.constant.%s: %s;\n" % (
                key,
                hasattr(self._constant, key),
            )
        exception += "\nfield:\n"
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

    def add_variable(self, name: str, value: Variable):
        # Set variable
        self._check_shape(value.value, self._shape)
        setattr(self._variable, name, value)

    def empty_variable(self) -> Variable:
        variable = Variable()
        variable.value = self.zeros_field()
        variable.boundary_index = cp.zeros([1, 3], CUPY_INT)
        variable.boundary_type = cp.zeros([1], CUPY_INT)
        variable.boundary_value = cp.zeros([1], CUPY_FLOAT)
        return variable

    def add_field(self, name: str, value: cp.ndarray):
        # Set field
        self._check_shape(value, self._shape)
        setattr(self._field, name, value)

    def zeros_field(self, dtype=CUPY_FLOAT):
        return cp.zeros(self._shape, dtype)

    def ones_field(self, dtype=CUPY_FLOAT):
        return cp.ones(self._shape, dtype)

    def add_constant(self, name: str, value: float):
        setattr(self._constant, name, NUMPY_FLOAT(value))

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
    def coordinate(self) -> SubGrid:
        return self._coordinate

    @property
    def variable(self) -> SubGrid:
        return self._variable

    @property
    def field(self) -> SubGrid:
        return self._field

    @property
    def constant(self) -> SubGrid:
        return self._constant


if __name__ == "__main__":
    grid = Grid(grid_width=0.1, x=[-2, 2], y=[-2, 2], z=[-2, 2])
    grid.set_requirement(
        variable_name_list=["phi"],
        field_name_list=["epsilon"],
        constant_name_list=["epsilon0"],
    )
    print(grid.requirement)
    phi = grid.empty_variable()
    epsilon = grid.zeros_field() + 1

    grid.add_variable("phi", phi)
    grid.add_field("epsilon", epsilon)
    grid.add_constant("epsilon0", 10)
    grid.check_requirement()
    print(grid.coordinate_range)
