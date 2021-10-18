#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_verlet_integrator.py
created time : 2021/10/18
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest, os
import numpy as np 
from ..file import PDBFile, PSFFile 
from ..forcefield import CharmmForcefield
from ..integrator import VerletIntegrator

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data')
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

    def test_step(self):
        pdb = PDBFile(os.path.join(data_dir, '2BQJ.pdb'))
        topology = PSFFile(os.path.join(data_dir, '2BQJ.psf')).create_topology()
        topology.set_pbc_matrix(np.diag(np.ones(3) * 40))

        forcefield = CharmmForcefield(topology)
        forcefield.set_param_files(os.path.join(data_dir, 'par_all36_prot.prm'))
        ensemble = forcefield.create_ensemble()
        ensemble.state.set_positions(pdb.positions)
        ensemble.state.set_velocities_to_temperature(300)
        integrator = VerletIntegrator(1)
        integrator.step(ensemble, 1)

        # ATOM      1  N   LYS A   1     -12.138   4.446  -6.361  1.00  0.00      A    N
        assert ensemble.state.positions[0, 1] == pytest.approx(4.446, abs=0.01)
        assert ensemble.state.positions[0, 0] == pytest.approx(-12.138, abs=0.01)