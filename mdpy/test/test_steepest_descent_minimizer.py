#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_steepest_descent_minimizer.py
created time : 2022/01/09
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest, os
import numpy as np 
from ..file import PDBFile, PSFFile 
from ..forcefield import CharmmForcefield
from ..minimizer import SteepestDescentMinimizer
from ..unit import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data')
out_dir = os.path.join(cur_dir, 'out')

class TestSteepestDescentMinimizer:
    def setup(self):
        pass

    def teardown(self):
        pass

    def test_attributes(self):
        pass

    def test_exceptions(self):
        pass

    def test_minimize(self):
        pdb = PDBFile(os.path.join(data_dir, '6PO6.pdb'))
        topology = PSFFile(os.path.join(data_dir, '6PO6.psf')).create_topology()

        forcefield = CharmmForcefield(topology)
        forcefield.set_param_files(os.path.join(data_dir, 'par_all36_prot.prm'))
        ensemble = forcefield.create_ensemble()
        ensemble.state.set_pbc_matrix(np.diag(np.ones(3)*100))
        ensemble.state.cell_list.set_cutoff_radius(12)
        ensemble.state.set_positions(pdb.positions)
        ensemble.state.set_velocities_to_temperature(300)
        ensemble.update()
        pre_energy = ensemble.potential_energy
        minimizer = SteepestDescentMinimizer()
        minimizer.minimize(ensemble, 0.01, 10)
        assert ensemble.potential_energy < pre_energy