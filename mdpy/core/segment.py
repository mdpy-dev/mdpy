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

from ..error import *

class Segment:
    def __init__(self, segment_id, particles) -> None:
        self._segment_id = segment_id
        self._particles = particles
        self._parent_ensemble = None
        
    def __repr__(self):
        return '<Segment object with %d particles at %x>' %(self.num_particles, id(self))
    
    __str__ = __repr__

    def _test_bound_state(self):
        if self._parent_ensemble == None:
            raise NonBoundedError(
                '%s has not been bounded to any Ensemble instance' %self
            )

    @property
    def segment_id(self):
        pass
    
    @property
    def num_particles(self):
        return len(self._particles)

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
    def center_of_mass(self):
        pass

    @property
    def temperature(self):
        pass