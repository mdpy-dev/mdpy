#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_particle.py
created time : 2021/09/28
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest
from ..core import Particle
from ..unit import *
from ..error import *

class TestParticle:
    def setup(self):
        pass

    def teardown(self):
        pass

    def test_attributes(self):
        particle = Particle(particle_id=1, particle_type='C')
        assert particle.particle_id == 1
        assert particle.particle_type == 'C'
        assert particle.mass == None

    def test_exceptions(self):
        pass 
    
    def test_mass(self):
        particle = Particle(particle_id=1, particle_type='C', mass=1)
        assert particle.mass == 1

        particle = Particle(mass=Quantity(1, kilogram))
        assert particle.mass == Quantity(1, kilogram).convert_to(default_mass_unit).value
        
        with pytest.raises(UnitDimensionDismatchedError):
            Particle(mass=Quantity(1, default_charge_unit))

    def test_charge(self):
        particle = Particle(particle_id=1, particle_type='C', charge=1)
        assert particle.charge == 1
        
        particle = Particle(charge=Quantity(1, coulomb))
        assert particle.charge == Quantity(1, coulomb).convert_to(default_charge_unit).value

        with pytest.raises(UnitDimensionDismatchedError):
            Particle(charge=Quantity(1, default_energy_unit))