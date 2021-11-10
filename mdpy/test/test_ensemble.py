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

import pytest, os
import numpy as np
from ..core import Particle, Topology
from ..ensemble import Ensemble
from ..file import CharmmParamFile, PSFFile
from ..constraint import *
from ..error import *
from ..unit import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data')

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
        self.ensemble = Ensemble(t)
        self.ensemble.state.set_pbc_matrix(np.eye(3)*30)
        self.ensemble.state.cell_list.set_cutoff_radius(12)
        self.ensemble.state.set_positions(positions)
        self.ensemble.state.set_velocities(velocities)

    def teardown(self):
        self.ensemble = None

    def test_attributes(self):
        pass

    def test_exceptions(self):
        pass

    def test_add_constraints(self):
        f2 = os.path.join(data_dir, 'par_all36_prot.prm')
        f3 = os.path.join(data_dir, 'toppar_water_ions_namd.str')
        psf_file_path = os.path.join(data_dir, '1M9Z.psf')
        charmm_file = CharmmParamFile(f2, f3)
        params = charmm_file.params
        topology = PSFFile(psf_file_path).create_topology()
        ensemble = Ensemble(topology)
        ensemble.state.set_pbc_matrix(np.diag(np.ones(3)*30))
        constraint = CharmmNonbondedConstraint(params['nonbonded'])
        ensemble.add_constraints(constraint)
        assert ensemble.num_constraints == 1
        
        with pytest.raises(ConstraintConflictError):
            ensemble.add_constraints(constraint)

    def test_update_kinetic_energy(self):
        self.ensemble._update_kinetic_energy()
        assert self.ensemble.kinetic_energy == Quantity(
            13.5, default_velocity_unit**2*default_mass_unit
        ).convert_to(default_energy_unit).value

    def test_update_potential_energy(self):
        self.ensemble.update()
        assert self.ensemble.potential_energy == 0

    def test_update_energy(self):
        self.ensemble.update()
        assert self.ensemble.total_energy == Quantity(
            13.5, default_velocity_unit**2*default_mass_unit
        ).convert_to(default_energy_unit).value