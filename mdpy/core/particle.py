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

from ..unit import Quantity
from ..unit import default_mass_unit
from ..error import *

class Particle:
    def __init__(
        self, particle_id: int, particle_type: str,
        mass=None, molecule_id=None, molecule_type=None, chain_id=None
    ) -> None:
        # Compulsory attributes
        self._particle_id = particle_id
        self._particle_type = particle_type
        # Optional attributes
        self._molecule_id = molecule_id
        self._molecule_type = molecule_type
        self._chain_id = chain_id
        self._mass = Quantity(mass, default_mass_unit) if mass != None else mass

    def __repr__(self) -> str:
        return '<Particle %s-%d at %x>' %(self._particle_type, self._particle_id, id(self))
        # return '<Particle object: ID %d, Type %s at %x>' %(self._particle_id, self._particle_type, id(self))

    __str__ = __repr__
    
    def __eq__(self, o) -> bool:
        if id(self) == id(o):
            return True
        return False
    
    def change_particle_id(self, particle_id: int):
        # Only used by Topology
        self._particle_id = particle_id
            
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

    