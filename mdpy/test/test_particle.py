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
from ..moduler import Particle
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
        with pytest.raises(SpatialDimError):
            Particle(1, 'CA', position=[1, 2, 3, 4])

        with pytest.raises(SpatialDimError):
            Particle(1, 'CA', velocity=[1, 2, 3, 4])

    def test_position(self):
        particle = Particle(1, 'CA', position=[1, 2, 3])
        assert particle.position.unit == default_length_unit
        assert particle.position[0].value == 1

        particle = Particle(1, 'CA', position=np.ones([3, 1]))
        assert len(list(particle.position.value.shape)) == 1

        particle.position = [3, 2, 1]
        assert particle.position[0].value == 3
        assert particle.position.unit == default_length_unit
        assert len(list(particle.position.value.shape)) == 1

    def test_velocity(self):
        particle = Particle(1, 'CA', velocity=[1, 2, 3])
        assert particle.velocity.unit == default_length_unit / default_time_unit
        assert particle.velocity[0].value == 1

        particle = Particle(1, 'CA', velocity=np.ones([3, 1]))
        assert len(list(particle.velocity.value.shape)) == 1

        particle.velocity = [3, 2, 1]
        assert particle.velocity[0].value == 3
        assert particle.velocity.unit == default_length_unit / default_time_unit
        assert len(list(particle.velocity.value.shape)) == 1

    def test_mass(self):
        particle = Particle(1, 'CA', mass=1)
        assert particle.mass.value == 1
        assert particle.mass.unit == default_mass_unit