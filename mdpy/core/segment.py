#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : segment.py
created time : 2021/09/28
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

class Segment:
    def __init__(self, segment_id, particles) -> None:
        self._segment_id = segment_id
        self._particles = particles
        self._parent_ensemble = None
        self._is_bound = False

    def bind_ensemble(self, ensemble):
        pass

    @property
    def segment_id(self):
        pass

    @property
    def particles(self):
        pass

    @property
    def positions(self):
        pass

    @property
    def forces(self):
        pass

    @property
    def velocities(self):
        pass

    @property
    def total_energy(self):
        pass

    @property
    def potential_energy(self):
        pass

    @property
    def kinetic_energy(self):
        pass

    @property
    def temperature(self):
        pass