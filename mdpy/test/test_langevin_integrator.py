#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_langevin_integrator.py
created time : 2021/10/31
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest, os
import numpy as np 
from ..file import PDBFile, PSFFile 
from ..forcefield import CharmmForcefield
from ..integrator import LangevinIntegrator
from ..unit import *

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

    def test_integrate(self):
        pdb = PDBFile(os.path.join(data_dir, '6PO6.pdb'))
        topology = PSFFile(os.path.join(data_dir, '6PO6.psf')).create_topology()

        forcefield = CharmmForcefield(topology)
        forcefield.set_param_files(os.path.join(data_dir, 'par_all36_prot.prm'))
        ensemble = forcefield.create_ensemble()
        ensemble.state.set_pbc_matrix(np.diag(np.ones(3)*100))
        ensemble.state.cell_list.set_cutoff_radius(12)
        ensemble.state.set_positions(pdb.positions)
        ensemble.state.set_velocities_to_temperature(300)
        integrator = LangevinIntegrator(1, 300, Quantity(1, 1/picosecond))
        integrator.integrate(ensemble, 1)

        # ATOM      1  N   VAL A   1       2.347  -0.970   3.962  1.00  0.00      A    N
        assert ensemble.state.positions[0, 1] == pytest.approx(-0.970, abs=0.01)
        assert ensemble.state.positions[0, 0] == pytest.approx(2.347, abs=0.01)