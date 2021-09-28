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

from functools import partial
from sys import path_importer_cache
import pytest
import numpy as np
from ..core import Particle
from ..unit import default_length_unit, default_time_unit, default_mass_unit
from ..error import *

class TestParticle:
    def setup(self):
        pass

    def teardown(self):
        pass

    def test_attributes(self):
        particle = Particle(1, 'CA')
        assert particle.particle_id == 1
        assert particle.particle_type == 'CA'
        assert particle.mass == None

    def test_exceptions(self):
        pass 
    
    def test_mass(self):
        particle = Particle(1, 'CA', mass=1)
        assert particle.mass.value == 1
        assert particle.mass.unit == default_mass_unit