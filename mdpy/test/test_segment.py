#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_segment.py
created time : 2021/10/03
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

# *** Note: current test file only contain test for information without Ensemble
import pytest
from ..core import Particle, Topology, Segment
from ..unit import Quantity, default_mass_unit
from ..error import *

class TestSegment:
    def setup(self):
        self.ps = [
            Particle(0, 'CA', molecule_id=0, molecule_type='ASN', mass=1),
            Particle(1, 'CB', molecule_id=0, molecule_type='ASN', mass=1),
            Particle(2, 'N', molecule_id=0, molecule_type='ASN', mass=1),
            Particle(3, 'H', molecule_id=1, molecule_type='HOH', mass=1),
            Particle(4, 'O', molecule_id=1, molecule_type='HOH', mass=1),
            Particle(5, 'H', molecule_id=1, molecule_type='HOH', mass=1),
            Particle(6, 'CA', molecule_id=2, molecule_type='ASP', mass=1),
            Particle(7, 'CB', molecule_id=2, molecule_type='ASP', mass=1),
            Particle(8, 'N', molecule_id=2, molecule_type='ASP', mass=1),
        ]
        self.topology = Topology()
        self.topology.add_particles(*self.ps)

    def teardown(self):
        pass

    def test_attributes(self):
        segment = Segment(0, self.topology.select_particles('molecule_id=1'))
        assert segment.segment_id == 0
        assert segment.num_particles == 3
        assert segment.segment_mass == Quantity(3, default_mass_unit)
        assert segment.particles_id[0] == 3
        assert segment.particles_mass[2] == Quantity(1, default_mass_unit)

    def test_exceptions(self):
        segment = Segment(0, self.topology.select_particles('molecule_id=1'))
        with pytest.raises(NonBoundedError):
            segment._test_bound_state()