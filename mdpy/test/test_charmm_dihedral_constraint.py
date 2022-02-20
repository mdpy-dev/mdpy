#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_charmm_dihedral_constraint.py
created time : 2021/10/11
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest, os
import numpy as np
from .. import env
from ..constraint import CharmmDihedralConstraint
from ..core import Particle, Topology
from ..ensemble import Ensemble
from ..file import CharmmParamFile
from ..utils import get_dihedral
from ..error import *
from ..unit import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data')

class TestCharmmDihedralConstraint:
    def setup(self):
        p1 = Particle(
            particle_id=0, particle_type='C', 
            particle_name='CA', molecule_type='ASN',
            mass=12, charge=0
        )
        p2 = Particle(
            particle_id=1, particle_type='N', 
            particle_name='NY', molecule_type='ASN',
            mass=14, charge=0
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
        t = Topology()
        t.add_particles([p1, p2, p3, p4])
        # CA   NY   CPT  CA       3.0000  2   180.00
        t.add_dihedral([0, 1, 2, 3])
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
        self.constraint = CharmmDihedralConstraint(self.params['dihedral'], 0, 0)

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
        assert self.constraint.num_dihedrals == 1

        # CA   NY   CPT  CA       3.0000  2   180.00
        assert self.constraint._int_params[0][0] == 0
        assert self.constraint._int_params[0][1] == 1
        assert self.constraint._int_params[0][2] == 2
        assert self.constraint._int_params[0][3] == 3
        assert self.constraint._float_params[0][0] == Quantity(3, kilocalorie_permol).convert_to(default_energy_unit).value
        assert self.constraint._float_params[0][1] == Quantity(2).value
        assert self.constraint._float_params[0][2] == np.deg2rad(Quantity(180).value)

        # No exception
        self.constraint._check_bound_state() 

    def test_update(self):    
        self.ensemble.add_constraints(self.constraint)
        self.constraint.update()
        
        forces = self.constraint.forces
        k, n, delta = self.params['dihedral']['CA-NY-CPT-CA'][0]
        theta = get_dihedral([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], is_angular=False)
        assert forces.sum() == pytest.approx(0, abs=1e-8)

        positions = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
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
        assert res[0] == pytest.approx(0, abs=1e-8)
        assert res[1] == pytest.approx(0, abs=1e-8)
        assert res[2] == pytest.approx(0, abs=1e-8)

        energy = self.constraint.potential_energy
        k, n, delta = self.params['dihedral']['CA-NY-CPT-CA'][0]
        theta = get_dihedral([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])
        assert energy == pytest.approx(env.NUMPY_FLOAT(k * (1 + np.cos(n*theta - delta))))
