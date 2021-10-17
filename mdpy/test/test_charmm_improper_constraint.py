#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_charmm_improper_constraint.py
created time : 2021/10/12
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest, os
import numpy as np
from ..constraint import CharmmImproperConstraint
from ..core import Particle, Topology
from ..ensemble import Ensemble
from ..file import CharmmParamFile
from ..math import *
from ..error import *
from ..unit import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data')

class TestCharmmDihedralConstraint:
    def setup(self):
        p1 = Particle(
            particle_id=0, particle_type='H', 
            particle_name='HE2', molecule_type='ASN',
            mass=12, charge=0
        )
        p2 = Particle(
            particle_id=1, particle_type='H', 
            particle_name='HE2', molecule_type='ASN',
            mass=14, charge=0
        )
        p3 = Particle(
            particle_id=2, particle_type='C', 
            particle_name='CE2', molecule_type='ASN',
            mass=1, charge=0
        )
        p4 = Particle(
            particle_id=3, particle_type='C', 
            particle_name='CE2', molecule_type='ASN',
            mass=12, charge=0
        )
        t = Topology()
        t.add_particles([p1, p2, p3, p4])
        # CA   NY   CPT  CA       3.0000  2   180.00
        t.add_improper([0, 1, 2, 3])
        positions = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 0.5, 1]
        ])
        velocities = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
        ])
        self.ensemble = Ensemble(t)
        self.ensemble.set_positions(positions)
        self.ensemble.set_velocities(velocities)

        f1 = os.path.join(data_dir, 'toppar_water_ions_namd.str')
        f2 = os.path.join(data_dir, 'par_all36_prot.prm')
        f3 = os.path.join(data_dir, 'top_all36_na.rtf')
        charmm = CharmmParamFile(f1, f2, f3)
        self.params = charmm.params
        self.constraint = CharmmImproperConstraint(self.params['improper'], 0, 0)

    def teardown(self):
        self.ensemble, self.params, self.constraint = None, None, None

    def test_attributes(self):
        pass

    def test_exceptions(self):
        with pytest.raises(NonBoundedError):
            self.constraint._check_bound_state()

    def test_bind_ensemble(self):
        self.constraint.bind_ensemble(self.ensemble)
        assert self.constraint._parent_ensemble.num_constraints == 1

        assert self.constraint._improper_type[0] == 'HE2-HE2-CE2-CE2'
        assert self.constraint._improper_matrix_id[0][0] == 0
        assert self.constraint._improper_matrix_id[0][1] == 1
        assert self.constraint._improper_matrix_id[0][2] == 2
        assert self.constraint._improper_matrix_id[0][3] == 3
        assert self.constraint.num_impropers == 1

        # HE2  HE2  CE2  CE2     3.0            0      0.00   
        assert self.constraint._improper_info[0][0] == 0
        assert self.constraint._improper_info[0][1] == 1
        assert self.constraint._improper_info[0][2] == 2
        assert self.constraint._improper_info[0][3] == 3
        assert self.constraint._improper_info[0][4] == Quantity(3, kilocalorie_permol).convert_to(default_energy_unit).value
        assert self.constraint._improper_info[0][5] == Quantity(0).value

        # No exception
        self.constraint._check_bound_state()

    def test_get_forces(self):
        self.constraint.bind_ensemble(self.ensemble)
        forces = self.constraint.get_forces()
        k, psi0 = self.params['improper']['HE2-HE2-CE2-CE2']
        psi = get_dihedral([0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 0.5, 1], is_angular=False)
        assert forces.sum() == pytest.approx(0)

        positions = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 0.5, 1]
        ])
        vec_bc = positions[2, :] - positions[1, :]
        vec_oc = vec_bc / 2
        vec_ob = - vec_bc / 2
        vec_oa = vec_ob + positions[0, :] - positions[1, :] # Vec ba
        vec_od = vec_oc + positions[3, :] - positions[2, :] # Vec cd
        res = (
            np.cross(vec_oa, forces[0, :]) +
            np.cross(vec_ob, forces[1, :]) +
            np.cross(vec_oc, forces[2, :]) + 
            np.cross(vec_od, forces[3, :])
        )
        assert res[0] == pytest.approx(0)
        assert res[1] == pytest.approx(0)
        assert res[2] == pytest.approx(0)

        force_val = 2 * k * (90 - psi0)
        vec_ab = positions[1, :] - positions[0, :]
        theta_abc = get_angle(
            positions[0, :], positions[1, :], positions[2, :]
        )
        force_a = force_val / (np.linalg.norm(vec_ab) * np.sin(theta_abc)) * get_unit_vec(np.cross(-vec_ab, vec_bc))
        assert forces[0, 0] == force_a[0]
        assert forces[0, 1] == force_a[1]
        assert forces[0, 2] == force_a[2]

    def test_get_potential_energy(self):
        self.constraint.bind_ensemble(self.ensemble)
        energy = self.constraint.get_potential_energy()
        k, psi0 = self.params['improper']['HE2-HE2-CE2-CE2']
        psi = get_dihedral([0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 0.5, 1], is_angular=False)
        assert energy == k * (90 - psi0)**2