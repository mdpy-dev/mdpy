#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_hdf5_parser.py
created time : 2022/02/24
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import os
import pytest
import numpy as np
from mdpy.io import HDF5Parser
from mdpy.unit import *
from mdpy.error import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, "data/hdf5_parser")
out_dir = os.path.join(cur_dir, "out")
file_path = os.path.join(data_dir, "test_hdf5_parser.hdf5")


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
            HDF5Parser("te.hd")

        with pytest.raises(IOPoorDefinedError):
            HDF5Parser(file_path, is_parse_all=False).positions

        with pytest.raises(IOPoorDefinedError):
            HDF5Parser(file_path, is_parse_all=False).trajectory

        with pytest.raises(ArrayDimError):
            HDF5Parser(file_path, is_parse_all=False).get_positions(100)

        with pytest.raises(ArrayDimError):
            HDF5Parser(file_path, is_parse_all=False).get_positions(100, 101)
