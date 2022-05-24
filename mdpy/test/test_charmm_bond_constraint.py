#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_charmm_bond_constraint.py
created time : 2021/10/09
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import pytest, os
import numpy as np
from mdpy import env
from mdpy.constraint import CharmmBondConstraint
from mdpy.core import Particle, Topology, Ensemble
from mdpy.io import CharmmTopparParser
from mdpy.utils import get_bond
from mdpy.error import *
from mdpy.unit import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data/simulation/')

class TestCharmmBondConstraint:
    def setup(self):
        p1 = Particle(
            particle_id=0, particle_name='C',
            particle_type='CA', molecule_type='ASN',
            mass=12, charge=0
        )
        p2 = Particle(
            particle_id=1, particle_name='N',
            particle_type='N', molecule_type='ASN',
            mass=14, charge=0
        )
        p3 = Particle(
            particle_id=2, particle_name='H',
            particle_type='HA1', molecule_type='ASN',
            mass=1, charge=0
        )
        p4 = Particle(
            particle_id=3, particle_name='C',
            particle_type='CA', molecule_type='ASN',
            mass=12, charge=0
        )
        t = Topology()
        t.add_particles([p1, p2, p3, p4])
        t.add_bond([0, 3]) # CA   CA    305.000     1.3750
        positions = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
        ])
        velocities = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
        ])
        self.ensemble = Ensemble(t, np.eye(3)*30)
        self.ensemble.tile_list.set_cutoff_radius(12)
        self.ensemble.state.set_positions(positions)
        self.ensemble.state.set_velocities(velocities)

        f1 = os.path.join(data_dir, 'toppar_water_ions_namd.str')
        f2 = os.path.join(data_dir, 'par_all36_prot.prm')
        f3 = os.path.join(data_dir, 'top_all36_na.rtf')
        charmm = CharmmTopparParser(f1, f2, f3)
        self.parameters = charmm.parameters
        self.constraint = CharmmBondConstraint(self.parameters['bond'])

    def teardown(self):
        self.ensemble, self.parameters, self.constraint = None, None, None

    def test_attributes(self):
        pass

    def test_exceptions(self):
        with pytest.raises(NonBoundedError):
            self.constraint._check_bound_state()

    def test_bind_ensemble(self):
        self.ensemble.add_constraints(self.constraint)
        assert self.constraint._parent_ensemble.num_constraints == 1
        assert self.constraint.num_bonds == 1

        # CA   CA    305.000     1.3750
        assert self.constraint._int_parameters[0][0] == 0
        assert self.constraint._int_parameters[0][1] == 3
        assert self.constraint._float_parameters[0][0] == Quantity(305, kilocalorie_permol).convert_to(default_energy_unit).value
        assert self.constraint._float_parameters[0][1] == Quantity(1.3750, angstrom).convert_to(default_length_unit).value

        # No exception
        self.constraint._check_bound_state()

    def test_update(self):
        self.ensemble.add_constraints(self.constraint)
        self.constraint.update()
        forces = self.constraint.forces.get()
        assert forces[1, 0] == 0
        assert forces[2, 1] == 0

        bond_length = get_bond([0, 0, 0], [0, 0, 1])
        k, r0 = self.parameters['bond']['CA-CA']
        force_val = - 2 * k * (bond_length - r0)
        assert forces[0, 0] == 0
        assert forces[0, 1] == 0
        assert forces[0, 2] == - force_val
        assert forces[3, 0] == 0
        assert forces[3, 1] == 0
        assert forces[3, 2] == force_val
        assert forces.sum() == 0

        energy = self.constraint.potential_energy.get()
        bond_length = get_bond([0, 0, 0], [0, 0, 1])
        k, r0 = self.parameters['bond']['CA-CA']
        assert energy == pytest.approx(env.NUMPY_FLOAT(k * (bond_length - r0)**2))
