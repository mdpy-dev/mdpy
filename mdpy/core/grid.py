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


class Grid:
    def __init__(self, **kwargs) -> None:
        # Set grid information
        self._keys = list(kwargs.keys())
        self._num_dimensions = len(self._keys)
        self._shape = [value[2] + 2 for value in kwargs.values()]
        self._inner_shape = [value[2] for value in kwargs.values()]
        grid = [
            cp.linspace(start=value[0], stop=value[1], num=value[2] + 2, endpoint=True)
            for value in kwargs.values()
        ]
        self._device_grid_width = cp.array([i[1] - i[0] for i in grid], NUMPY_FLOAT)
        self._grid_width = self._device_grid_width.get()
        grid = cp.meshgrid(*grid, indexing="ij")
        for index, key in enumerate(self._keys):
            setattr(
                self,
                key,
                grid[index],
            )

        # Initialize requirement
        self._requirement = {}

        # Initialize attributes
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

        self._field = SubGrid("field")
        self._gradient = SubGrid("gradient")
        self._curvature = SubGrid("curvature")

    def set_requirement(self, requirement: dict):
        for key, value in requirement.items():
            self._requirement[key] = {
                "require_gradient": value["require_gradient"],
                "require_curvature": value["require_curvature"],
            }
            if hasattr(self._field, key):
                if self._requirement[key]["require_gradient"]:
                    setattr(
                        self._gradient,
                        key,
                        self.get_gradient(getattr(self._field, key)),
                    )
                if self._requirement[key]["require_curvature"]:
                    setattr(
                        self._curvature,
                        key,
                        self.get_curvature(getattr(self._field, key)),
                    )

    def check_requirement(self):
        is_all_set = True
        for key, value in self._requirement.items():
            is_all_set &= hasattr(self._field, key)
        if not is_all_set:
            exception = "Gird is not all set:\n"
            for key in self._requirement.keys():
                exception += "- %s: %s\n" % (key, hasattr(self._field, key))
            raise GridPoorDefinedError(exception[:-1])

    def _check_shape(self, value: cp.ndarray, target_shape: list[int]):
        shape = value.shape
        if len(shape) != self._num_dimensions:
            raise ArrayDimError(
                "A %d-D array is required, while %d-D is provided"
                % (self._num_dimensions, len(shape))
            )
        for dim1, dim2 in zip(shape, target_shape):
            if dim1 != dim2:
                raise ArrayDimError(
                    "Require Array in %s, while %s is provided"
                    % (tuple(target_shape), shape)
                )

    def add_field(
        self,
        name: str,
        value: cp.ndarray,
        require_gradient: bool = False,
        require_curvature: bool = False,
    ):
        # Set field
        self._check_shape(value, self._shape)
        setattr(self._field, name, value)
        # Set gradient and curvature
        if not name in self._requirement.keys():  # not required
            self._requirement[name] = {
                "require_gradient": False,
                "require_curvature": False,
            }
        self._requirement[name]["require_gradient"] |= require_gradient
        self._requirement[name]["require_curvature"] |= require_curvature
        if self._requirement[name]["require_gradient"]:
            setattr(
                self._gradient,
                name,
                self.get_gradient(getattr(self._field, name)),
            )
        if self._requirement[name]["require_curvature"]:
            setattr(
                self._curvature,
                name,
                self.get_curvature(getattr(self._field, name)),
            )

    def zeros_field(self, dtype=CUPY_FLOAT):
        return cp.zeros(self._shape, dtype)

    def ones_field(self, dtype=CUPY_FLOAT):
        return cp.ones(self._shape, dtype)

    def get_gradient(self, field: cp.ndarray):
        self._check_shape(field, self._shape)
        gradient = cp.zeros([self._num_dimensions] + self._inner_shape, field.dtype)
        slice_list = [slice(1, -1) for i in range(self.num_dimensions)]
        for i in range(self._num_dimensions):
            slice_list[i] = slice(2, self._shape[i])
            gradient[i, :, :, :] = field[tuple(slice_list)]
            slice_list[i] = slice(0, -2)
            gradient[i, :, :, :] -= field[tuple(slice_list)]
            gradient[i, :, :, :] /= 2 * self._grid_width[i]
            slice_list[i] = slice(1, -1)
        return gradient

    def get_curvature(self, field: cp.ndarray):
        self._check_shape(field, self._shape)
        curvature = cp.zeros([self._num_dimensions] + self._inner_shape, field.dtype)
        slice_list = [slice(1, -1) for i in range(self.num_dimensions)]
        for i in range(self._num_dimensions):
            slice_list[i] = slice(2, self._shape[i])
            curvature[i, :, :, :] = field[tuple(slice_list)]
            slice_list[i] = slice(0, -2)
            curvature[i, :, :, :] += field[tuple(slice_list)]
            slice_list[i] = slice(1, -1)
            curvature[i, :, :, :] -= 2 * field[tuple(slice_list)]
            curvature[i, :, :, :] /= self._grid_width[i] ** 2
        return curvature

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
    def device_grid_width(self) -> cp.ndarray:
        return self._device_grid_width

    @property
    def field(self) -> object:
        return self._field

    @property
    def gradient(self) -> object:
        return self._gradient

    @property
    def curvature(self) -> object:
        return self._curvature


if __name__ == "__main__":
    grid = Grid(x=[-2, 2, 128], y=[-2, 2, 128], z=[-2, 2, 64])
    grid.set_requirement(
        {
            "phi": {"require_gradient": False, "require_curvature": False},
            "epsilon": {"require_gradient": True, "require_curvature": True},
        },
    )
    print(grid.requirement)
    phi = grid.zeros_field()
    epsilon = grid.zeros_field() + 1

    phi[0, :, :] = 20
    phi[-1, :, :] = 0
    grid.add_field("phi", phi)
    grid.add_field("epsilon", epsilon)
    grid.check_requirement()
    print(grid.gradient.phi)
