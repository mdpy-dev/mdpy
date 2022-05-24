#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_hdf5_writer.py
created time : 2022/02/24
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import os
import pytest
import numpy as np
from mdpy.io import HDF5Writer, PSFParser, PDBParser
from mdpy.unit import *
from mdpy.error import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data/simulation')
out_dir = os.path.join(cur_dir, 'out')
file_path = os.path.join(out_dir, 'test_hdf5_writer.hdf5')

class TestHDF5Writer:
    def setup(self):
        self.topology = PSFParser(os.path.join(data_dir, '6PO6.psf')).topology
        self.positions = PDBParser(os.path.join(data_dir, '6PO6.pdb')).positions
        self.pbc_matrix = Quantity(np.diag([1]*3), nanometer)

    def teardown(self):
        self.topology = None
        self.positions = None
        self.pbc_matrix = None

    def test_attributes(self):
        pass

    def test_exceptions(self):
        with pytest.raises(FileFormatError):
            HDF5Writer('te.hd')

    def test_write(self):
        writer = HDF5Writer(file_path, topology=self.topology, pbc_matrix=self.pbc_matrix)
        writer.write(self.positions)
        writer.write(self.positions)