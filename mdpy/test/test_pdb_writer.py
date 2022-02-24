#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_pdb_writer.py
created time : 2022/02/24
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import os
import pytest
import numpy as np
from .. import SPATIAL_DIM
from ..io import PDBWriter, PDBParser, PSFParser
from ..error import *
from ..unit import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data')
out_dir = os.path.join(cur_dir, 'out')

class TestPDBWriter:
    def setup(self):
        self.topology = PSFParser(os.path.join(data_dir, '6PO6.psf')).topology
        self.positions = PDBParser(os.path.join(data_dir, '6PO6.pdb')).positions

    def teardown(self):
        self.topology = None

    def test_attributes(self):
        pass

    def test_exceptions(self):
        with pytest.raises(FileFormatError):
            PDBWriter('a.pdd', 'a', self.topology)
        
        writer = PDBWriter(
            os.path.join(out_dir, 'test_pdb_writer.pdb'), 'w',
            self.topology, Quantity(np.diag([1]*3), nanometer)
        )
        with pytest.raises(ArrayDimError):
            writer.write(np.ones([5, 1]))

        with pytest.raises(ArrayDimError):
            writer.write(np.ones([1, 5, 1]))

        with pytest.raises(ArrayDimError):
            writer.write(np.ones([1, 1, 5, 1]))

    def test_write(self):
        writer = PDBWriter(
            os.path.join(out_dir, 'test_pdb_writer.pdb'), 'w',
            self.topology, Quantity(np.diag([1]*3), nanometer)
        )
        num_frames = 5
        positions = np.ones([num_frames, self.topology.num_particles, SPATIAL_DIM])
        writer.write(positions)

        writer.write(self.positions)
        writer.close()