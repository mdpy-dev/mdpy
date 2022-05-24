#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_verlet_integrator.py
created time : 2021/10/18
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import pytest, os
import numpy as np
from mdpy.io import PDBParser, PSFParser
from mdpy.forcefield import CharmmForcefield
from mdpy.integrator import VerletIntegrator

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data/simulation/')
out_dir = os.path.join(cur_dir, 'out')

class TestVerletIntegrator:
    def setup(self):
        pass

    def teardown(self):
        pass

    def test_attributes(self):
        pass

    def test_exceptions(self):
        pass

    def test_integrate(self):
        pdb = PDBParser(os.path.join(data_dir, '6PO6.pdb'))
        topology = PSFParser(os.path.join(data_dir, '6PO6.psf')).topology

        forcefield = CharmmForcefield(topology, np.diag(np.ones(3)*100), long_range_solver='CUTOFF')
        forcefield.set_parameter_files(os.path.join(data_dir, 'par_all36_prot.prm'))
        ensemble = forcefield.create_ensemble()
        ensemble.tile_list.set_cutoff_radius(12)
        ensemble.state.set_positions(pdb.positions)
        ensemble.state.set_velocities_to_temperature(300)
        ensemble.update_tile_list()
        integrator = VerletIntegrator(1, 10)
        integrator.integrate(ensemble, 1)

        # ATOM      1  N   VAL A   1       2.347  -0.970   3.962  1.00  0.00      A    N
        assert ensemble.state.positions.get()[0, 1] == pytest.approx(-0.970, abs=0.01)
        assert ensemble.state.positions.get()[0, 0] == pytest.approx(2.347, abs=0.01)
