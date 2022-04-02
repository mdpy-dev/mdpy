#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : dumper.py
created time : 2021/09/28
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

from mdpy.simulation import Simulation
from mdpy.error import *

class Dumper:
    def __init__(self, file_path: str, dump_frequency: int, suffix: str) -> None:
        if not file_path.endswith(suffix):
            raise FileFormatError('The file should end with .%s suffix' %suffix)
        if dump_frequency <= 0:
            raise DumperPoorDefinedError(
                'The dump_frequency of dumper should be a positive integer.'
            )
        self._file_path = file_path
        self._file = open(file_path, 'w')
        self._file.close()
        self._dump_frequency = dump_frequency
        self._num_dumpped_frames = 0

    def dump(self, simulation: Simulation):
        raise NotImplementedError(
            'The subclass of mdpy.dumper.Dumper class should overload update_ensemble method.'
        )

    def _dump_info(self, info):
        with open(self._file_path, 'a') as f:
            print(info, file=f, end='')

    @property
    def file_path(self):
        return self._file_path

    @property
    def dump_frequency(self):
        return self._dump_frequency

    @dump_frequency.setter
    def dump_frequency(self, val: int):
        self._dump_frequency = val