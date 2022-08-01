#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_electrostatic_pme_constraint.py
created time : 2022/04/10
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import pytest, os
import numpy as np
from mdpy.constraint import ElectrostaticPMEConstraint
from mdpy.core import Particle, Topology, Ensemble
from mdpy.error import *
from mdpy.unit import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, "data")


class TestElectrostaticPMEConstraint:
    def setup(self):
        p1 = Particle(
            particle_id=0,
            particle_type="C",
            particle_name="CA",
            molecule_type="ASN",
            mass=12,
            charge=1,
        )
        p2 = Particle(
            particle_id=1,
            particle_type="N",
            particle_name="NY",
            molecule_type="ASN",
            mass=14,
            charge=2,
        )
        p3 = Particle(
            particle_id=2,
            particle_type="CA",
            particle_name="CPT",
            molecule_type="ASN",
            mass=1,
            charge=0,
        )
        p4 = Particle(
            particle_id=3,
            particle_type="C",
            particle_name="CA",
            molecule_type="ASN",
            mass=12,
            charge=0,
        )
        self.pbc = np.diag(np.ones(3) * 100)
        t = Topology()
        t.add_particles([p1, p2, p3, p4])
        self.p = np.array([[0, 0, 0], [0, 10, 0], [0, 21, 0], [0, 11, 0]])
        self.ensemble = Ensemble(t, np.eye(3) * 30)
        self.ensemble.tile_list.set_cutoff_radius(12)
        self.ensemble.state.set_positions(self.p)
        self.constraint = ElectrostaticPMEConstraint()

    def teardown(self):
        self.ensemble, self.parameters, self.constraint = None, None, None

    def test_attributes(self):
        pass

    def test_exceptions(self):
        with pytest.raises(NonBoundedError):
            self.constraint._check_bound_state()

    def test_bind_ensemble(self):
        self.ensemble.state.set_pbc_matrix(self.pbc)
        # Non-neutralized system
        with pytest.raises(EnsemblePoorDefinedError):
            self.ensemble.add_constraints(self.constraint)

    def test_get_grid_size(self):
        p1 = Particle(
            particle_id=0,
            particle_type="C",
            particle_name="CA",
            molecule_type="ASN",
            mass=12,
            charge=1,
        )
        p2 = Particle(
            particle_id=1,
            particle_type="N",
            particle_name="NY",
            molecule_type="ASN",
            mass=14,
            charge=-1,
        )
        p3 = Particle(
            particle_id=2,
            particle_type="CA",
            particle_name="CPT",
            molecule_type="ASN",
            mass=1,
            charge=0,
        )
        p4 = Particle(
            particle_id=3,
            particle_type="C",
            particle_name="CA",
            molecule_type="ASN",
            mass=12,
            charge=0,
        )
        t = Topology()
        t.add_particles([p1, p2, p3, p4])
        p = np.array([[0, 0, 0], [0, 10, 0], [0, 21, 0], [0, 11, 0]])
        ensemble = Ensemble(t, np.eye(3) * 30)
        ensemble.add_constraints(self.constraint)
        assert self.constraint._get_grid_size()[0] == 32

        ensemble.state.set_pbc_matrix(np.eye(3) * 65)
        assert self.constraint._get_grid_size()[1] == 96
