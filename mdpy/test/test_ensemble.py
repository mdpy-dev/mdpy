#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_ensemble.py
created time : 2021/10/09
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

# *** Note: current test file only contain test for information without constrait

import pytest
import numpy as np
from ..core import Particle, Topology
from ..error import *
from ..ensemble import Ensemble
from ..constraint import Constraint

class TestEnsemble:
    def setup(self):
        p1 = Particle(
            particle_id=0, particle_type='C', 
            particle_name='CA', molecule_type='ASN',
            mass=12, charge=0
        )
        p2 = Particle(
            particle_id=1, particle_type='N', 
            particle_name='N', molecule_type='ASN',
            mass=14, charge=0
        )
        p3 = Particle(
            particle_id=2, particle_type='H', 
            particle_name='HA', molecule_type='ASN',
            mass=1, charge=0
        )
        p4 = Particle(
            particle_id=3, particle_type='C', 
            particle_name='CB', molecule_type='ASN',
            mass=12, charge=0
        )
        t = Topology()
        t.add_particles([p1, p2, p3, p4])
        t.add_bond([0, 1])
        t.add_bond([1, 2])
        t.add_bond([2, 3])
        t.add_angle([0, 1, 2])
        t.add_dihedral([0, 1, 2, 3])
        positions = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
        ])
        velocities = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
        ])
        self.ensemble = Ensemble(positions, t)
        self.ensemble.set_velocities(velocities)

    def teardown(self):
        self.ensemble = None

    def test_attributes(self):
        pass

    def test_exceptions(self):
        with pytest.raises(ParticleConflictError):
            self.ensemble.set_positions(np.ones([5, 3]))

        with pytest.raises(SpatialDimError):
            self.ensemble.set_velocities(np.ones([4, 4]))

    def test_add_constraints(self):
        c1, c2 = Constraint(), Constraint()
        self.ensemble.add_constraints(c1, c2)
        assert self.ensemble.num_constraints == 2
        assert c1.force_id == 0
        assert c2.force_id == 1
        
        with pytest.raises(ConstraintConflictError):
            self.ensemble.add_constraints(c1)

    def test_update_kinetic_energy(self):
        self.ensemble._update_kinetic_energy()
        assert self.ensemble.kinetic_energy == 13.5

    def test_update_potential_energy(self):
        self.ensemble._update_potential_energy()
        assert self.ensemble.potential_energy == 0

    def test_update_energy(self):
        self.ensemble.update_energy()
        assert self.ensemble.total_energy == 13.5