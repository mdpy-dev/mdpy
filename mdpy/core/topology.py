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
        self._particles = []
        self._bonds = []
        self._angles = []
        self._dihedrals = []
        self._impropers = []

    def add_particle(self, p: Particle):
        self._particles.append(p)

    def add_bond(self, p1: Particle, p2: Particle):
        self._bonds.append([p1, p2])
        if not p1 in self._particles:
            self.add_particle(p1)
        if not p2 in self._particles:
            self.add_particle(p2)

    def add_angle(self, p1: Particle, p2: Particle, p3: Particle):
        self._angles.append([p1, p2, p3])
        if not p1 in self._particles:
            self.add_particle(p1)
        if not p2 in self._particles:
            self.add_particle(p2)
        if not p3 in self._particles:
            self.add_particle(p3)
        
    def add_dihedral(self, p1: Particle, p2: Particle, p3: Particle, p4: Particle):
        self._dihedrals.append([p1, p2, p3, p4])
        if not p1 in self._particles:
            self.add_particle(p1)
        if not p2 in self._particles:
            self.add_particle(p2)
        if not p3 in self._particles:
            self.add_particle(p3)
        if not p4 in self._particles:
            self.add_particle(p4)

    def add_improper(self, p1: Particle, p2: Particle, p3: Particle, p4: Particle):
        self._impropers.append([p1, p2, p3, p4])
        if not p1 in self._particles:
            self.add_particle(p1)
        if not p2 in self._particles:
            self.add_particle(p2)
        if not p3 in self._particles:
            self.add_particle(p3)
        if not p4 in self._particles:
            self.add_particle(p4)

    @property
    def particles(self):
        return self._particles

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