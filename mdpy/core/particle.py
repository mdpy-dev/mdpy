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

from .. import SPATIAL_DIM
from ..unit import Quantity
from ..unit import default_mass_unit, default_length_unit, default_velocity_unit, default_force_unit
from ..error import *

class Particle:
    def __init__(
        self, particle_id: int, particle_type: str,
        mass=None, position=None, velocity=None, force=None,
        molecule_id=None, molecule_type=None, chain_id=None
    ) -> None:
        # Compulsory attributes
        self._particle_id = particle_id
        self._particle_type = particle_type
        # Optional attributes
        self._molecule_id = molecule_id
        self._molecule_type = molecule_type
        self._chain_id = chain_id
        self._mass = Quantity(mass, default_mass_unit) if mass != None else mass
        
        if not isinstance(position, type(None)):
            self._position = Quantity(position, default_length_unit)
            try:
                self._position.value = self._position.value.reshape(SPATIAL_DIM)
            except: # Prod for (3, 1) or (1, 3) shape
                raise SpatialDimError('The dimension of position should be %d' %SPATIAL_DIM)
        else:
            self._position = None            

        if not isinstance(velocity, type(None)):
            self._velocity = Quantity(velocity, default_velocity_unit)
            try:
                self._velocity.value = self._velocity.value.reshape(SPATIAL_DIM)
            except: 
                raise SpatialDimError('The dimension of velocity should be %d' %SPATIAL_DIM)
        else:
            self._velocity = None   

        if not isinstance(force, type(None)):
            self._force = Quantity(force, default_force_unit)
            try:
                self._force.value = self._force.value.reshape(SPATIAL_DIM)
            except: 
                raise SpatialDimError('The dimension of force should be %d' %SPATIAL_DIM)
        else:
            self._force = None           

    def __repr__(self) -> str:
        return '<Particle object: ID %d, Type %s at %x>' %(self._particle_id, self._particle_type, id(self))

    __str__ = __repr__

    @property
    def particle_id(self):
        return self._particle_id
    
    @property
    def particle_type(self):
        return self._particle_type

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
    def position(self, is_quantity=False):
        return self._position

    @position.setter
    def position(self, val):
        val = Quantity(val, default_length_unit)
        val.value = val.value.reshape(SPATIAL_DIM)
        if val.value.shape[0] == SPATIAL_DIM:
            self._position = val
        else:
            raise SpatialDimError('The dimension of position should be %d' %SPATIAL_DIM)

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, val):
        val = Quantity(val, default_velocity_unit)
        val.value = val.value.reshape(SPATIAL_DIM)
        if val.value.shape[0] == SPATIAL_DIM:
            self._velocity = val
        else:
            raise SpatialDimError('The dimension of velocity should be %d' %SPATIAL_DIM)

    @property
    def force(self):
        return self._force

    @velocity.setter
    def force(self, val):
        val = Quantity(val, default_force_unit)
        val.value = val.value.reshape(SPATIAL_DIM)
        if val.value.shape[0] == SPATIAL_DIM:
            self._force = val
        else:
            raise SpatialDimError('The dimension of force should be %d' %SPATIAL_DIM)



    