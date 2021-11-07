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

import numpy as np

from .core import Topology, State
from .error import *
from .unit import *

class Ensemble:
    def __init__(self, topology: Topology) -> None:
        if not topology.is_joined:
            topology.join()
        # Read input
        self._topology = topology
        self._state = State(self._topology)
        self._matrix_shape = self._state.matrix_shape
        self._forces = np.zeros(self._matrix_shape)

        self._total_energy = 0
        self._potential_energy = 0
        self._kinetic_energy = 0
        self._segments = []
        self._num_segments = 0
        self._constraints = []
        self._num_constraints = 0

    def __repr__(self) -> str:
        return '<mdpy.Ensemble object: %d constraints at %x>' %(
            self._num_constraints, id(self)
        )

    __str__ = __repr__

    def create_segment(self, keywords):
        pass

    def add_constraints(self, *constraints):
        for constraint in constraints:
            if constraint in self._constraints:
                raise ConstraintConflictError(
                    '%s has added twice to %s' 
                    %(constraint, self)
                )
            self._constraints.append(constraint)
            constraint.bind_ensemble(self)
            if constraint.cutoff_radius > self.state.cell_list.cutoff_radius:
                self.state.cell_list.set_cutoff_radius(constraint.cutoff_radius)
            self._num_constraints += 1

    def update(self):
        self._forces = np.zeros(self._matrix_shape)
        self._total_energy, self._kinetic_energy, self._potential_energy = 0, 0, 0
        for constraint in self._constraints:
            constraint.update()
            self._forces += constraint.forces
            self._potential_energy += constraint.potential_energy
        self._update_kinetic_energy()
        self._total_energy = self._potential_energy + self._kinetic_energy
    
    def _update_kinetic_energy(self):
        # Without reshape, the result of the first sum will be a 1d vector
        # , which will be a matrix after multiple with a 2d vector
        self._kinetic_energy = ((
            (self._state.velocities**2).sum(1).reshape(self._topology.num_particles, 1) * self._topology.masses
        ).sum() / 2)
        self._kinetic_energy = Quantity(
            self._kinetic_energy, default_velocity_unit**2 * default_mass_unit
        ).convert_to(default_energy_unit).value
    
    @property
    def topology(self):
        return self._topology

    @property
    def state(self):
        return self._state
    
    @property
    def forces(self):
        return self._forces
    
    @property
    def total_energy(self):
        return self._total_energy
    
    @property
    def potential_energy(self):
        return self._potential_energy
    
    @property
    def kinetic_energy(self):
        return self._kinetic_energy

    @property
    def segments(self):
        return self._segments

    @property
    def num_segments(self):
        return self._num_segments

    @property
    def constraints(self):
        return self._constraints
    
    @property
    def num_constraints(self):
        return self._num_constraints