#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_hdf5_dumper.py
created time : 2022/02/25
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest, os
import numpy as np
import h5py
from .. import env
from ..io import PDBParser, PSFParser
from ..ensemble import Ensemble
from ..integrator import Integrator
from ..simulation import Simulation
from ..dumper import HDF5Dumper

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data')
out_dir = os.path.join(cur_dir, 'out')

class TestPDBDumper:
    def setup(self):
        self.topology = PSFParser(os.path.join(data_dir, '6PO6.psf')).topology
        self.ensemble = Ensemble(self.topology)
        self.ensemble.state.set_pbc_matrix(np.diag(np.ones(3)*100))
        self.ensemble.state.cell_list.set_cutoff_radius(12)
        positions = PDBParser(os.path.join(data_dir, '6PO6.pdb')).positions
        self.ensemble.state.set_positions(np.ascontiguousarray(positions).astype(env.NUMPY_FLOAT))
        self.integrator = Integrator(1)
        self.simulation = Simulation(self.ensemble, self.integrator)
        self.file_path = os.path.join(out_dir, 'test_hdf5_dumper.hdf5')

    def teardown(self):
        self.particles = None
        self.topology = None
        self.ensemble = None
        self.simulation = None

    def test_attributes(self):
        pass

    def test_exceptions(self):
        pass

    def test_dump(self):
        dumper = HDF5Dumper(self.file_path, 1)
        dumper.dump(self.simulation)
        new_position = self.simulation.ensemble.state.positions
        new_position[:, 0] += 1
        self.simulation.ensemble.state.set_positions(new_position)
        dumper.dump(self.simulation)

        with h5py.File(self.file_path, 'r') as f:
            assert f['topology/num_particles'][()] == self.ensemble.topology.num_particles
            assert f['positions/num_frames'][()] == 2