#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_steepest_descent_minimizer.py
created time : 2022/01/09
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import pytest, os
import numpy as np
from mdpy.io import PDBParser, PSFParser
from mdpy.recipe import CharmmRecipe
from mdpy.minimizer import SteepestDescentMinimizer
from mdpy.unit import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, "data/simulation/")
out_dir = os.path.join(cur_dir, "out")


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
        pdb = PDBParser(os.path.join(data_dir, "6PO6.pdb"))
        topology = PSFParser(os.path.join(data_dir, "6PO6.psf")).topology

        recipe = CharmmRecipe(
            topology, np.diag(np.ones(3) * 100), long_range_solver="CUTOFF"
        )
        recipe.set_parameter_files(os.path.join(data_dir, "par_all36_prot.prm"))
        ensemble = recipe.create_ensemble()
        ensemble.tile_list.set_cutoff_radius(12)
        ensemble.state.set_positions(pdb.positions)
        ensemble.state.set_velocities_to_temperature(300)
        ensemble.update_tile_list()
        ensemble.update_constraints()
        pre_energy = ensemble.potential_energy
        minimizer = SteepestDescentMinimizer()
        minimizer.minimize(ensemble, 0.01, 10)
        assert ensemble.potential_energy < pre_energy
