#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : particle.py
created time : 2021/09/28
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

from ..utils import check_quantity_value
from ..unit import *
from ..error import *

class Particle:
    def __init__(
        self, particle_id=None, particle_type=None, particle_name=None,
        matrix_id=None, molecule_id=None, molecule_type=None, chain_id=None,
        mass=None, charge=None
    ) -> None:
        # Compulsory attributes
        self._particle_id = particle_id
        self._particle_type = particle_type
        self._particle_name = particle_name
        self._matrix_id = matrix_id
        # Optional attributes
        self._molecule_id = molecule_id
        self._molecule_type = molecule_type
        self._chain_id = chain_id
        self._mass = check_quantity_value(mass, default_mass_unit)
        self._charge = check_quantity_value(charge, default_charge_unit)
        # Topology infomation
        self._bonded_particles = []
        self._num_bonded_particles = 0
        self._scaling_particles = []
        self._scaling_factors = []
        self._num_scaling_particles = 0

    def __repr__(self) -> str:
        return '<mdpy.core.Particle object: %s-%d at %x>' %(self._particle_name, self._particle_id, id(self))
        # return '<Particle object: ID %d, Type %s at %x>' %(self._particle_id, self._particle_type, id(self))

    __str__ = __repr__    
    
    def __eq__(self, o) -> bool:
        if id(self) == id(o):
            return True
        return False
    
    def change_matrix_id(self, matrix_id: int):
        # Only used by Topology
        self._matrix_id = matrix_id

    def add_bonded_particle(self, particle_matrix_id):
        if particle_matrix_id in self._bonded_particles:
            raise ParticleConflictError(
                'Particle %d has been added twice to the bonded_particles of Particle %d'
                %(particle_matrix_id, self._matrix_id)
            )
        if particle_matrix_id == self._matrix_id:
            raise ParticleConflictError(
                'Particle itself can not be added to the bonded_particle list.'
            )
        self._bonded_particles.append(particle_matrix_id)
        self._num_bonded_particles += 1
    
    def del_bonded_particle(self, particle_matrix_id):
        if particle_matrix_id in self._bonded_particles:
            self._bonded_particles.remove(particle_matrix_id)
            self._num_bonded_particles -= 1

    def add_scaling_particle(self, particle_matrix_id: int, factor):
        # Benzene give two dihedral with same 1 4 particles
        if particle_matrix_id == self._matrix_id:
            raise ParticleConflictError(
                'Particle itself can not be added to the scaling_particle list.'
            )   
        elif not particle_matrix_id in self._scaling_particles:
            self._scaling_particles.append(particle_matrix_id)
            self._scaling_factors.append(factor)
            self._num_scaling_particles += 1

    def del_scaling_particle(self, particle_matrix_id):
        if particle_matrix_id in self._scaling_particles:
            self._scaling_factors.remove(
                self._scaling_factors[self._scaling_particles.index(particle_matrix_id)]
            )
            self._scaling_particles.remove(particle_matrix_id)
            self._num_scaling_particles -= 1
            
    @property
    def particle_id(self):
        return self._particle_id
    
    @property
    def particle_type(self):
        return self._particle_type

    @property
    def particle_name(self):
        return self._particle_name

    @property
    def matrix_id(self):
        return self._matrix_id

    @property
    def molecule_id(self):
        return self._molecule_id

    @property
    def molecule_type(self):
        return self._molecule_type

    @property
    def chain_id(self):
        return self._chain_id

    @property
    def mass(self):
        return self._mass
    
    @property
    def charge(self):
        return self._charge

    @property
    def bonded_particles(self):
        return self._bonded_particles

    @property
    def num_bonded_particles(self):
        return self._num_bonded_particles

    @property
    def scaling_particles(self):
        return self._scaling_particles

    @property
    def scaling_factors(self):
        return self._scaling_factors

    @property
    def num_scaling_particles(self):
        return self._num_scaling_particles