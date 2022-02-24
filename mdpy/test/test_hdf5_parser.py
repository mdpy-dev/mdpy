#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_hdf5_parser.py
created time : 2022/02/24
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import os
import pytest
import numpy as np
from ..io import HDF5Parser
from ..unit import *
from ..error import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data')
out_dir = os.path.join(cur_dir, 'out')
file_path = os.path.join(data_dir, 'test_hdf5_parser.hdf5')

class TestHDF5Writer:
    def setup(self):
        pass

    def teardown(self):
        pass
    
    def test_attributes(self):
        parser = HDF5Parser(file_path)
        # Topology
        assert parser.topology.num_particles == 49
        assert parser.topology.num_bonds == 49
        # Positions
        assert parser.positions.shape[0] == 2
        assert parser.positions.shape[1] == 49
        # PBC Matrix
        assert parser.pbc_matrix[0, 0] == 10
        # Trajectory
        assert parser.trajectory.num_frames == 2


    def test_exceptions(self):
        with pytest.raises(FileFormatError):
            HDF5Parser('te.hd')
