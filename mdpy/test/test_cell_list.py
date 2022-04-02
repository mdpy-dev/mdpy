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
        self.cell_list = CellList(np.diag([30, 30, 30]), Quantity(9, angstrom))

    def teardown(self):
        self.cell_list = None

    def test_attributes(self):
        assert self.cell_list.cutoff_radius == 9
        assert self.cell_list.pbc_matrix[0, 0] == 30

    def test_exceptions(self):
        with pytest.raises(UnitDimensionDismatchedError):
            self.cell_list.set_cutoff_radius(Quantity(1, second))

        with pytest.raises(CellListPoorDefinedError):
            self.cell_list.set_cutoff_radius(0)

        with pytest.raises(CellListPoorDefinedError):
            self.cell_list.set_cutoff_radius(24)

    def test_update(self):
        pdb = PDBParser(os.path.join(data_dir, '6PO6.pdb'))
        self.cell_list.update(pdb.positions)
        cell_id = np.floor(np.dot(pdb.positions - pdb.positions.min(0), self.cell_list.cell_inv))
        # cell_id -= cell_id.min(0)
        num_atoms = pdb.positions.shape[0]
        atoms = []
        for atom in range(num_atoms):
            if (cell_id[atom, :] == 0).all():
                atoms.append(atom)
        assert len(atoms) != 0
        for index, atom in enumerate(atoms):
            assert atom == self.cell_list.cell_list[0, 0, 0, index]

        # ATOM     24  HB1 PHE A   2      -2.752  -0.222   1.686  1.00  0.00      A
        assert 23 in self.cell_list[-1, -1, 0]
        # ATOM     39  N   ALA A   3      -0.248  -0.263  -1.755  1.00  0.00      A    N
        assert 38 in self.cell_list[-1, -1, -1]

