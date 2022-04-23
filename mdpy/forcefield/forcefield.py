#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : forcefield.py
created time : 2021/10/05
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

from mdpy.core import Topology, Ensemble

class Forcefield:
    def __init__(self, topology: Topology) -> None:
        self._topology = topology
        self._parameters = None

    def set_parameter_files(self):
        raise NotImplementedError(
            'The subclass of mdpy.forcefield.Forcefield class should overload set_parameter_files method'
        )

    def check_parameters(self):
        raise NotImplementedError(
            'The subclass of mdpy.forcefield.Forcefield class should overload check_parameters method'
        )

    def create_ensemble(self) -> Ensemble:
        raise NotImplementedError(
            'The subclass of mdpy.forcefield.Forcefield class should overload create_ensemble method'
        )

    @property
    def parameters(self):
        return self._parameters