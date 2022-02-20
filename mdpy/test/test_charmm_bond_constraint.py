#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_charmm_bond_constraint.py
created time : 2021/10/09
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest, os
import numpy as np
from .. import env
from ..constraint import CharmmBondConstraint
from ..core import Particle, Topology
from ..ensemble import Ensemble
from ..file import CharmmParamFile
from ..utils import get_bond
from ..error import *
from ..unit import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data')

class TestCharmmBondConstraint:
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
            particle_name='HA1', molecule_type='ASN',
            mass=1, charge=0
        )
        p4 = Particle(
            particle_id=3, particle_type='C', 
            particle_name='CA', molecule_type='ASN',
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
        self.ensemble = Ensemble(t)
        self.ensemble.state.set_pbc_matrix(np.eye(3)*30)
        self.ensemble.state.cell_list.set_cutoff_radius(12)
        self.ensemble.state.set_positions(positions)
        self.ensemble.state.set_velocities(velocities)

        f1 = os.path.join(data_dir, 'toppar_water_ions_namd.str')
        f2 = os.path.join(data_dir, 'par_all36_prot.prm')
        f3 = os.path.join(data_dir, 'top_all36_na.rtf')
        charmm = CharmmParamFile(f1, f2, f3)
        self.params = charmm.params
        self.constraint = CharmmBondConstraint(self.params['bond'], 0, 0)

    def teardown(self):
        self.ensemble, self.params, self.constraint = None, None, None

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
        assert self.constraint._int_params[0][0] == 0
        assert self.constraint._int_params[0][1] == 3
        assert self.constraint._float_params[0][0] == Quantity(305, kilocalorie_permol).convert_to(default_energy_unit).value
        assert self.constraint._float_params[0][1] == Quantity(1.3750, angstrom).convert_to(default_length_unit).value
        
        # No exception
        self.constraint._check_bound_state()

    def test_update(self):
        self.ensemble.add_constraints(self.constraint)
        self.constraint.update()
        forces = self.constraint.forces
        assert forces[1, 0] == 0
        assert forces[2, 1] == 0 

        bond_length = get_bond([0, 0, 0], [0, 0, 1])
        k, r0 = self.params['bond']['CA-CA']
        force_val = - 2 * k * (bond_length - r0)
        assert forces[0, 0] == 0
        assert forces[0, 1] == 0
        assert forces[0, 2] == - force_val
        assert forces[3, 0] == 0
        assert forces[3, 1] == 0
        assert forces[3, 2] == force_val
        assert forces.sum() == 0

        energy = self.constraint.potential_energy
        bond_length = get_bond([0, 0, 0], [0, 0, 1])
        k, r0 = self.params['bond']['CA-CA']
        assert energy == pytest.approx(env.NUMPY_FLOAT(k * (bond_length - r0)**2))
