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
from ..math import get_angle, get_unit_vec
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
        self.ensemble = Ensemble(t)
        self.ensemble.state.set_positions(positions)
        self.ensemble.state.set_velocities(velocities)

        f1 = os.path.join(data_dir, 'toppar_water_ions_namd.str')
        f2 = os.path.join(data_dir, 'par_all36_prot.prm')
        f3 = os.path.join(data_dir, 'top_all36_na.rtf')
        charmm = CharmmParamFile(f1, f2, f3)
        self.params = charmm.params
        self.constraint = CharmmAngleConstraint(self.params['angle'], 0, 0)

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
        assert self.constraint.num_angles == 1

        assert self.constraint._angle_info[0][0] == 0
        assert self.constraint._angle_info[0][1] == 1
        assert self.constraint._angle_info[0][2] == 3
        assert self.constraint._angle_info[0][3] == Quantity(40, kilocalorie_permol).convert_to(default_energy_unit).value
        assert self.constraint._angle_info[0][4] == Quantity(120).value

        # No exception
        self.constraint._check_bound_state()

    def test_update(self):
        self.ensemble.add_constraints(self.constraint)
        self.ensemble.state.set_pbc_matrix(np.diag(np.ones(3)*10))
        self.constraint.update()

        forces = self.constraint.forces
        k, theta0 = self.params['angle']['CA-CA-CA']
        theta = get_angle([0, 0, 0], [1, 0, 0], [0, 0, 1], is_angular=False)
        theta_rad = np.pi / 4
        force_val = 2 * k * (45 - theta0) / np.abs(np.sin(theta_rad))
        vec0, vec1 = np.array([-1, 0, 0]), get_unit_vec(np.array([-1, 0, 1]))
        force_vec0 = (vec1 - vec0 * np.cos(theta_rad)) / 1
        force_vec2 = (vec0 - vec1 * np.cos(theta_rad)) / np.sqrt(2)
        force0, force2 = force_val * force_vec0, force_val * force_vec2
        assert forces[0, 0] == pytest.approx(force0[0], abs=1e-10)
        assert forces[0, 1] == pytest.approx(force0[1])
        assert forces[0, 2] == pytest.approx(force0[2])
        assert forces[3, 0] == pytest.approx(force2[0])
        assert forces[3, 1] == pytest.approx(force2[1])
        assert forces[3, 2] == pytest.approx(force2[2])
        assert forces.sum() == pytest.approx(0, abs=1e-10)

        vec1 = np.array([-1, 0, 1])
        torque = np.cross(vec0, forces[0, :]) + np.cross(vec1, forces[3, :])
        assert torque[0] == pytest.approx(0, abs=1e-11)
        assert torque[1] == pytest.approx(0, abs=1e-11)
        assert torque[2] == pytest.approx(0, abs=1e-11)

        energy = self.constraint.potential_energy
        k, theta0 = self.params['angle']['CA-CA-CA']
        theta = get_angle([0, 0, 0], [1, 0, 0], [0, 0, 1], is_angular=False)
        assert energy == k * (theta - theta0)**2
        