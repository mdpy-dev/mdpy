#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : forcefield.py
created time : 2021/10/05
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

from ..core import Topology

class Forcefield:
    def __init__(self, topology: Topology) -> None:
        self._topology = topology
        self._params = None

    def set_param_files(self):
        raise NotImplementedError(
            'The subclass of mdpy.forcefield.Forcefield class should overload set_param_files method'
        )

    def check_params(self):
        raise NotImplementedError(
            'The subclass of mdpy.forcefield.Forcefield class should overload check_params method'
        )

    def create_ensemble(self):
        raise NotImplementedError(
            'The subclass of mdpy.forcefield.Forcefield class should overload create_ensemble method'
        )

    @property
    def params(self):
        pass