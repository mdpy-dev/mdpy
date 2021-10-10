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
        assert particle.mass.value == 1
        assert particle.mass.unit == default_mass_unit

    def test_charge(self):
        particle = Particle(particle_id=1, particle_type='C', charge=1)
        assert particle.charge.value == 1
        assert particle.charge.unit == default_charge_unit