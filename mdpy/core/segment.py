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

import numpy as np
from ..unit import *
from ..error import *

class Segment:
    def __init__(self, segment_id, particles) -> None:
        self._segment_id = segment_id
        self._particles = particles
        self._num_particles = len(particles)
        self._particles_id, self._particles_mass, self._segment_mass = [], [], Quantity(0, default_mass_unit)
        for p in self._particles:
            self._particles_id.append(p.particle_id)
            self._particles_mass.append(p.mass.value)
            self._segment_mass += p.mass
        self._particles_mass = np.array(self._particles_mass).reshape([self._num_particles, 1])
        self._parent_ensemble = None
        
    def __repr__(self):
        return '<Segment object with %d particles at %x>' %(self.num_particles, id(self))
    
    __str__ = __repr__

    def _test_bound_state(self):
        if self._parent_ensemble == None:
            raise NonBoundedError(
                '%s has not been bounded to any Ensemble instance' %self
            )

    def bind_to_ensemble(self, ensemble):
        self._parent_ensemble = ensemble
        self.update_segment_info()

    def update_segment_info(self):
        self._test_bound_state()
        self._positions = self._parent_ensemble.positions[self._particles_id, :]
        self._velocities = self._parent_ensemble.velocities[self._particles_id, :]
        self._forces = self._parent_ensemble.forces[self._particles_id, :]

    @property
    def segment_id(self):
        return self._segment_id

    @property
    def particles_id(self):
        return self._particles_id
    
    @property
    def num_particles(self):
        return self._num_particles

    @property
    def segment_mass(self):
        return self._segment_mass

    @property
    def particles_mass(self):
        return self._particles_mass

    @property
    def particles(self):
        return self._particles

    @property
    def positions(self):
        self._test_bound_state()
        return self._positions

    @property
    def velocities(self):
        self._test_bound_state()
        return self._velocities 

    @property
    def forces(self):
        self._test_bound_state()
        return self._forces

    @property
    def center_of_mass(self):
        return np.sum(self._positions * self._particles_mass, 0) / self._segment_mass / self._num_particles

    @property
    def center_of_geomertry(self):
        return np.sum(self._positions, 0) / self._num_particles

    @property
    def temperature(self):
        kinetic_energy = np.sum(self._velocities**2 * self._particles_mass) / self._num_particles / 2
        kinetic_energy *= default_velocity_unit ** 2 * default_mass_unit
        return kinetic_energy * 2 / 3 / KB