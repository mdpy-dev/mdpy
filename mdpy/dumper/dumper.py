#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : dumper.py
created time : 2021/09/28
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

from ..ensemble import Ensemble

class Dumper:
    def __init__(self, file_path: str, dump_frequency: int) -> None:
        self._file_path = file_path
        self._dump_frequency = dump_frequency

    def dump(self, ensemble: Ensemble):
        raise NotImplementedError(
            'The subclass of mdpy.dumper.Dumper class should overload update_ensemble method'
        )

    @property
    def file_path(self):
        return self._file_path

    @property
    def dump_frequency(self):
        return self._dump_frequency

    @dump_frequency.setter
    def dump_frequency(self, val: int):
        self._dump_frequency = val