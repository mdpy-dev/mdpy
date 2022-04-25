#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_cell_list.py
created time : 2021/10/28
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import pytest, os
import numpy as np
from mdpy.core import CellList
from mdpy.io import PDBParser
from mdpy.unit import *
from mdpy.error import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data')

class TestCellList:
    def setup(self):
        self.cell_list = CellList(np.diag([30, 30, 30]))
        self.cell_list.set_cutoff_radius(Quantity(9, angstrom))

    def teardown(self):
        self.cell_list = None

    def test_attributes(self):
        assert self.cell_list.cutoff_radius == 9
        assert self.cell_list.pbc_matrix[0, 0] == 30

    def test_exceptions(self):
        with pytest.raises(UnitDimensionDismatchedError):
            self.cell_list.set_cutoff_radius(Quantity(1, second))

        with pytest.raises(NeighborListPoorDefinedError):
            self.cell_list.set_cutoff_radius(0)

        with pytest.raises(NeighborListPoorDefinedError):
            self.cell_list.set_cutoff_radius(24)

    def test_update(self):
        pdb = PDBParser(os.path.join(data_dir, '6PO6.pdb'))
        self.cell_list.update(pdb.positions)
        cell_id = np.floor(np.dot(pdb.positions - pdb.positions.min(0), self.cell_list.cell_inv))
        num_particles = pdb.positions.shape[0]
        particles = []
        for particle in range(num_particles):
            if (cell_id[particle, :] == 0).all():
                particles.append(particle)
        assert len(particles) != 0

        # -15,-4.7 -5, 5, 5, 15
        # ATOM     24  HB1 PHE A   2      -2.752  -0.222   1.686  1.00  0.00      A
        assert 23 in self.cell_list.cell_list[1, 1, 1, :]
        # ATOM     32  HZ  PHE A   2      -5.353  -0.632  -3.449  1.00  0.00      A
        assert 31 in self.cell_list.cell_list[0, 1, 1, :]