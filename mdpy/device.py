#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : device.py
created time : 2022/08/01
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import cupy as cp
import numba.cuda as cuda
from mdpy.error import *


def is_device_available():
    return cuda.is_available()


def get_device_count():
    return len(cuda.list_devices())


class Device:
    def __init__(self, device_index: int = 0) -> None:
        self._num_devices = get_device_count()
        self._device_index = self._check_device_index(device_index)

    def _check_device_index(self, device_index: int) -> int:
        if device_index >= self._num_devices:
            raise DevicePoorDefinedError(
                "Device %d is no available, only %d devices are detected"
                % (device_index, self._num_devices)
            )
        return device_index

    def __enter__(self):
        self._cupy_device = cp.cuda.Device(self._device_index)
        cuda.select_device(self._device_index)
        return self

    def __exit__(self, *arg):
        del self._cupy_device
