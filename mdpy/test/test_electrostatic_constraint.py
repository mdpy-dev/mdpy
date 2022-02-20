#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_electrostatic_constraint.py
created time : 2021/10/21
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest, os
import numpy as np
from .. import env
from ..constraint import ElectrostaticConstraint
from ..core import Particle, Topology
from ..ensemble import Ensemble
from ..utils import get_unit_vec
from ..error import *
from ..unit import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data')

class TestElectrostaticConstraint:
    def setup(self):
        p1 = Particle(
            particle_id=0, particle_type='C', 
            particle_name='CA', molecule_type='ASN',
            mass=12, charge=1
        )
        p2 = Particle(
            particle_id=1, particle_type='N', 
            particle_name='NY', molecule_type='ASN',
            mass=14, charge=2
        )
        p3 = Particle(
            particle_id=2, particle_type='CA', 
            particle_name='CPT', molecule_type='ASN',
            mass=1, charge=0
        )
        p4 = Particle(
            particle_id=3, particle_type='C', 
            particle_name='CA', molecule_type='ASN',
            mass=12, charge=0
        )
        self.pbc = np.diag(np.ones(3) * 100)
        t = Topology()
        t.add_particles([p1, p2, p3, p4])
        self.p = np.array([
            [0, 0, 0], [0, 10, 0], [0, 21, 0], [0, 11, 0]
        ])
        self.ensemble = Ensemble(t)
        self.ensemble.state.set_pbc_matrix(np.eye(3)*30)
        self.ensemble.state.cell_list.set_cutoff_radius(12)
        self.ensemble.state.set_positions(self.p)
        self.constraint = ElectrostaticConstraint()

    def teardown(self):
        self.ensemble, self.params, self.constraint = None, None, None

    def test_attributes(self):
        pass

    def test_exceptions(self):
        with pytest.raises(NonBoundedError):
            self.constraint._check_bound_state()

    def test_bind_ensemble(self):
        self.ensemble.state.set_pbc_matrix(self.pbc)
        self.ensemble.add_constraints(self.constraint)
        assert self.constraint._parent_ensemble.num_constraints == 1

    def test_update(self):
        self.ensemble.state.set_pbc_matrix(self.pbc)
        self.ensemble.add_constraints(self.constraint)
        self.constraint.update()
        
        forces = self.constraint.forces
        assert forces[2, 0] == 0
        assert forces[3, 1] == 0
        k = Quantity(4 * np.pi) * EPSILON0
        force_val = - Quantity(1, e) * Quantity(2, e) / k / Quantity(10, angstrom)**2
        force_vec = get_unit_vec(np.array([0, 10, 0], dtype=env.NUMPY_FLOAT))
        force1 = force_val.convert_to(default_force_unit).value * force_vec
        assert forces[0, 0] == pytest.approx(force1[0])
        assert forces[0, 1] == pytest.approx(force1[1])
        assert forces[0, 2] == pytest.approx(force1[2])
        assert forces[1, 0] == pytest.approx(-force1[0])
        assert forces[1, 1] == pytest.approx(-force1[1])
        assert forces[1, 2] == pytest.approx(-force1[2])

        potential_energy = self.constraint.potential_energy
        k = Quantity(4 * np.pi) * EPSILON0
        energy = Quantity(1, e) * Quantity(2, e) / k / Quantity(10, angstrom)
        assert potential_energy == pytest.approx(
            energy.convert_to(default_energy_unit).value
        )