#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : system.py
created time : 2021/09/28
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

from . import Topology

class System:
    def __init__(self, particles, topology: Topology) -> None:
        self._particles = particles
        self._topology = topology

    @property
    def num_particles(self):
        return len(self._particles)

    @property
    def particles(self):
        return self._particles

    def positions(self):
        pass

    def velocities(self):
        pass

    def forces(self):
        pass

    def center_of_mass(self):
        pass

    def center_of_geometry(self):
        pass