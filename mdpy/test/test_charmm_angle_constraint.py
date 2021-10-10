#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_charmm_angle_constraint.py
created time : 2021/10/10
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest, os
import numpy as np
from ..constraint import CharmmAngleConstraint
from ..core import Particle, Topology
from ..ensemble import Ensemble
from ..file import CharmmParamFile
from ..math import get_angle
from ..error import *
from ..unit import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data')

class TestCharmmAngleConstraint:
    def setup(self):
        p1 = Particle(
            particle_id=0, particle_type='C', 
            particle_name='CA', molecule_type='ASN',
            mass=12, charge=0
        )
        p2 = Particle(
            particle_id=1, particle_type='C', 
            particle_name='CA', molecule_type='ASN',
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
        t.add_angle([0, 1, 3]) # CA   CA   CA    40.000    120.00
        positions = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
        ])
        velocities = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
        ])
        self.ensemble = Ensemble(positions, t)
        self.ensemble.set_velocities(velocities)

        f1 = os.path.join(data_dir, 'toppar_water_ions_namd.str')
        f2 = os.path.join(data_dir, 'par_all36_prot.prm')
        f3 = os.path.join(data_dir, 'top_all36_na.rtf')
        charmm = CharmmParamFile(f1, f2, f3)
        self.params = charmm.params
        self.constraint = CharmmAngleConstraint(0, 0)

    def teardown(self):
        self.ensemble, self.params, self.constraint = None, None, None

    def test_attributes(self):
        pass

    def test_exceptions(self):
        with pytest.raises(NonBoundedError):
            self.constraint._test_bound_state()

    def test_bind_ensemble(self):
        self.constraint.bind_ensemble(self.ensemble)
        assert self.constraint._parent_ensemble.num_constraints == 1

        assert self.constraint._angle_type[0] == 'CA-CA-CA'
        assert self.constraint._angle_matrix_id[0][0] == 0
        assert self.constraint._angle_matrix_id[0][1] == 1
        assert self.constraint._angle_matrix_id[0][2] == 3
        assert self.constraint.num_angles == 1

        # No exception
        self.constraint._test_bound_state()

    def test_set_params(self):
        # CA   CA   CA    40.000    120.00
        self.constraint.bind_ensemble(self.ensemble)
        self.constraint.set_params(self.params['angle'])
        assert self.constraint._angle_info[0][0] == 0
        assert self.constraint._angle_info[0][1] == 1
        assert self.constraint._angle_info[0][2] == 3
        assert self.constraint._angle_info[0][3] == Quantity(40, kilocalorie_permol).convert_to(default_energy_unit).value
        assert self.constraint._angle_info[0][4] == Quantity(120).value

    def test_get_forces(self):
        pass

    def test_potential_energy(self):
        self.constraint.bind_ensemble(self.ensemble)
        self.constraint.set_params(self.params['angle'])
        energy = self.constraint.get_potential_energy()
        k, theta0 = self.params['angle']['CA-CA-CA']
        angle = get_angle([0, 0, 0], [1, 0, 0], [0, 0, 1], is_angular=False)
        assert energy == k * (angle - theta0)**2