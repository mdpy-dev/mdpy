#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_charmm_nonbonded_constraint.py
created time : 2021/10/12
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest, os
import numpy as np
from .. import env
from ..constraint import CharmmNonbondedConstraint
from ..core import Particle, Topology
from ..ensemble import Ensemble
from ..file import CharmmParamFile
from ..file.charmm_param_file import RMIN_TO_SIGMA_FACTOR
from ..utils import get_unit_vec
from ..error import *
from ..unit import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data')

class TestCharmmNonbondedConstraint:
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
        self.pbc = np.diag(np.ones(3) * 30).astype(env.NUMPY_FLOAT)
        t = Topology()
        t.add_particles([p1, p2, p3, p4])
        self.p = np.array([
            [0, 0, 0], [0, 10, 0], [0, 21, 0], [0, 11, 0]
        ]).astype(env.NUMPY_FLOAT)
        velocities = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]
        ]).astype(env.NUMPY_FLOAT)
        self.ensemble = Ensemble(t)
        self.ensemble.state.set_pbc_matrix(np.eye(3)*30)
        self.ensemble.state.cell_list.set_cutoff_radius(5)
        self.ensemble.state.set_positions(self.p)
        self.ensemble.state.set_velocities(velocities)

        f1 = os.path.join(data_dir, 'toppar_water_ions_namd.str')
        f2 = os.path.join(data_dir, 'par_all36_prot.prm')
        f3 = os.path.join(data_dir, 'top_all36_na.rtf')
        charmm = CharmmParamFile(f1, f2, f3)
        self.params = charmm.params
        self.constraint = CharmmNonbondedConstraint(self.params['nonbonded'])

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

        # CA     0.000000  -0.070000     1.992400
        # NY     0.000000  -0.200000     1.850000 
        # CPT    0.000000  -0.099000     1.860000
        self.ensemble.state.set_pbc_matrix(self.pbc)
        assert self.constraint._params_list[0, 0] == Quantity(0.07, kilocalorie_permol).convert_to(default_energy_unit).value
        assert self.constraint._params_list[1, 1] == pytest.approx(env.NUMPY_FLOAT(1.85 * RMIN_TO_SIGMA_FACTOR * 2))

        # No exception
        self.constraint._check_bound_state()

    def test_update(self):
        self.ensemble.state.set_pbc_matrix(self.pbc)
        self.constraint.set_cutoff_radius(Quantity(0.91, nanometer))
        self.ensemble.add_constraints(self.constraint)
        self.ensemble.state.cell_list.update(self.ensemble.state.positions)
        # CA     0.000000  -0.070000     1.992400
        # NY     0.000000  -0.200000     1.850000 
        # CPT    0.000000  -0.099000     1.860000
        self.constraint.update()
        forces = self.constraint.forces
        assert forces.sum() == pytest.approx(0, abs=1e-8)
        
        epsilon = np.sqrt(
            Quantity(0.07, kilocalorie_permol).convert_to(default_energy_unit).value *
            Quantity(0.099, kilocalorie_permol).convert_to(default_energy_unit).value
        )
        sigma = (1.9924 + 1.86) * RMIN_TO_SIGMA_FACTOR
        r = 9
        scaled_r = sigma / r
        force_val = - 24 * epsilon / r * (2 * scaled_r**12 - scaled_r**6)
        force_vec = - get_unit_vec(self.p[2, :] - self.p[0, :]) # Manually PBC Wrap
        force = force_val * force_vec
        assert forces[0, 0] == pytest.approx(force[0], abs=1e-8)
        assert forces[0, 1] == pytest.approx(force[1], abs=1e-8)
        assert forces[0, 2] == pytest.approx(force[2], abs=1e-8)

        energy = self.constraint.potential_energy
        epsilon = np.sqrt(
            Quantity(0.07, kilocalorie_permol).convert_to(default_energy_unit).value *
            Quantity(0.2, kilocalorie_permol).convert_to(default_energy_unit).value
        )
        sigma = (1.9924 + 1.85) * RMIN_TO_SIGMA_FACTOR
        scaled_r = sigma / 1
        energy_ref = 4 * epsilon * (scaled_r**12 - scaled_r**6) 
        assert energy == pytest.approx(env.NUMPY_FLOAT(energy_ref), abs=1e-3)