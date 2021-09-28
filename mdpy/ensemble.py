#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : ensemble.py
created time : 2021/09/28
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

from .core import Topology

class Ensemble:
    def __init__(self, positions, topology: Topology) -> None:
        self._topology = topology
        self._positions = []
        self._velocities = []
        self._forces = []
        self._total_energy = []
        self._potential_energy = []
        self._kinetic_energy = []

        self._segments = []
        self._constraints = []

    def add_segment(self, keywords):
        pass

    def add_constraints(self):
        pass

    def update_force(self):
        pass

    def update_energy(self):
        pass