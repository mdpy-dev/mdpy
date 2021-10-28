#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_cell_list.py
created time : 2021/10/28
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest, os
import numpy as np
from ..core import CellList
from ..file import PDBFile
from ..unit import *
from ..error import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data')

class TestCellList:
    def setup(self):
        pass

    def teardown(self):
        pass

    def test_attributes(self):
        cell_list = CellList(Quantity(13, angstrom), np.diag([100, 100, 100]))
        assert cell_list.cutoff_radius == 13
        assert cell_list.pbc_matrix[0, 0] == 100

    def test_exceptions(self):
        with pytest.raises(UnitDimensionDismatchedError):
            CellList(Quantity(1, second), 0)

    def test_get_cell_index(self):
        cell_list = CellList(Quantity(13, angstrom), np.diag([100, 100, 100]))
        assert cell_list._get_cell_index(0, 0, 0) == 0
        assert cell_list._get_cell_index(0., 0., 1.) == 1
        assert cell_list._get_cell_index(7, 0, 0) == 0
        assert cell_list._get_cell_index(-1, 0, 0) == 6 * 7 * 7
        assert cell_list._get_cell_index(0, 0, 6) == 6 
        assert cell_list._get_cell_index(0, 1, 9) == 1 * 7 + 2

    def test_update(self):
        pdb = PDBFile(os.path.join(data_dir, '6PO6.pdb'))
        cell_list = CellList(Quantity(9, angstrom), np.diag([30, 30, 30]))
        cell_list.update(pdb.positions)
        cell_id = np.floor(np.dot(pdb.positions, cell_list.cell_inv))
        num_atoms = pdb.positions.shape[0]
        atoms = []
        for atom in range(num_atoms):
            if (cell_id[atom, :] == 0).all():
                atoms.append(atom)
        assert len(atoms) != 0
        for index, atom in enumerate(atoms):
            assert atom == cell_list[0, 0, 0][index]

        # ATOM     24  HB1 PHE A   2      -2.752  -0.222   1.686  1.00  0.00      A     
        assert 23 in cell_list[-1, -1, 0]
        # ATOM     39  N   ALA A   3      -0.248  -0.263  -1.755  1.00  0.00      A    N
        assert 38 in cell_list[-1, -1, -1]

    def test_get_neighbor(self):
        pdb = PDBFile(os.path.join(data_dir, '6PO6.pdb'))
        cell_list = CellList(Quantity(2, angstrom), np.diag([30, 30, 30]))
        cell_list.update(pdb.positions)
        matrix_ids = cell_list.get_neighbors(np.array([0, 0, 0]))
        atoms = []
        num_atoms = pdb.positions.shape[0]
        for atom in range(num_atoms):
            position = pdb.positions[atom, :]
            is_in_cell = True
            for p in position:
                if p > 4:
                    is_in_cell = False
                    break
                elif p < -2:
                    is_in_cell = False
                    break
            if is_in_cell:
                atoms.append(atom)
        assert len(matrix_ids) == len(atoms)

    