#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : topology.py
created time : 2021/09/28
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

from . import Particle

class Topology:
    def __init__(self) -> None:
        self._bonds = []
        self._angles = []
        self._dihedrals = []
        self._impropers = []

    def add_bonds(self, p1: Particle, p2: Particle):
        self._bonds.append([p1, p2])

    def add_angle(self, p1: Particle, p2: Particle, p3: Particle):
        self._angles.append([p1, p2, p3])

    def add_dihedrals(self, p1: Particle, p2: Particle, p3: Particle, p4: Particle):
        self._dihedrals.append([p1, p2, p3, p4])

    def add_impropers(self, p1: Particle, p2: Particle, p3: Particle, p4: Particle):
        self._impropers.append([p1, p2, p3, p4])

    @property
    def bonds(self):
        return self._bonds

    @property
    def angles(self):
        return self._angles

    @property
    def dihedrals(self):
        return self._dihedrals

    @property
    def improper(self):
        return self._impropers