#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_neighbor_list.py
created time : 2022/04/25
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import pytest, os
import numpy as np
from mdpy.core import NeighborList
from mdpy.io import PDBParser
from mdpy.unit import *
from mdpy.error import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data')

class TestNeighborList:
    def setup(self):
        self.neighbor_list = NeighborList(np.diag([30, 30, 30]))
        self.neighbor_list.set_cutoff_radius(Quantity(3, angstrom))

    def teardown(self):
        self.neighbor_list = None

    def test_attributes(self):
        assert self.neighbor_list.cutoff_radius == 3
        assert self.neighbor_list.pbc_matrix[0, 0] == 30

    def test_exceptions(self):
        with pytest.raises(UnitDimensionDismatchedError):
            self.neighbor_list.set_cutoff_radius(Quantity(1, second))

        with pytest.raises(NeighborListPoorDefinedError):
            self.neighbor_list.set_cutoff_radius(0)

        with pytest.raises(NeighborListPoorDefinedError):
            self.neighbor_list.set_cutoff_radius(24)

    def test_update(self):
        pdb = PDBParser(os.path.join(data_dir, '6PO6.pdb'))
        self.neighbor_list.update(pdb.positions)
        neighbor_list = self.neighbor_list.device_neighbor_list.get()
        for i in range(10):
            r = ((pdb.positions[0, :] - pdb.positions)**2).sum(1)
            r = np.sqrt(r)
            for i in np.argwhere(r <= 4).flatten():
                if i != 0:
                    assert i in list(neighbor_list[0, :])

if __name__ == '__main__':
    test = TestNeighborList()
    test.setup()
    test.test_update()