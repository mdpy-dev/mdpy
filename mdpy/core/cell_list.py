#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : cell_list.py
created time : 2021/10/27
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np
from .. import SPATIAL_DIM
from ..math import *
from ..unit import *

class CellList:
    def __init__(self, cutoff_radius, pbc_matrix: np.ndarray) -> None:
        self._cutoff_radius = check_quantity_value(cutoff_radius, default_length_unit)
        self._cell_matrix = np.ones(SPATIAL_DIM) * self._cutoff_radius
        self._pbc_diag = pbc_matrix.diagonal()
        self._num_cells = np.floor(self._pbc_diag / self._cell_matrix)
        self._cell_matrix = np.diag(self._pbc_diag / self._num_cells)
        self._cell_inv = np.linalg.inv(self._cell_matrix)
        self._num_atoms = 0
        self._num_cells = [int(i) for i in self._num_cells]
        self._cell_list = [[] for _ in range(np.prod(self._num_cells))]
        self._cell_id_min = np.array([0, 0, 0])

    def _get_cell_index(self, x: int, y: int, z: int):
        x -= self._cell_id_min[0]
        y -= self._cell_id_min[1]
        z -= self._cell_id_min[2]
        cell_index = [int(x), int(y), int(z)]
        for i, j in enumerate(cell_index):
            if j < 0:
                cell_index[i] = self._num_cells[i] + j
            elif j >= self._num_cells[i]:
                cell_index[i] -= self._num_cells[i]
        x, y, z = cell_index
        index = x * self._num_cells[1] * self._num_cells[2] + y * self._num_cells[2] + z       
        return index

    def __getitem__(self, keys):
        x, y, z = keys
        return self._cell_list[self._get_cell_index(x, y, z)]

    def get_neighbors(self, position: np.ndarray):
        matrix_ids = []
        cell_id = np.floor(np.dot(position, self._cell_inv))
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    matrix_ids.extend(
                        self[cell_id[0]+i, cell_id[1]+j, cell_id[2]+k]
                    )
        return matrix_ids

    def set_pbc_matrix(self, pbc_matrix: np.ndarray):
        self._pbc_diag = pbc_matrix.diagonal()
        self._num_cells = np.floor(self._pbc_diag / self._cell_matrix)
        self._cell_matrix = np.diag(self._pbc_diag / self._num_cells)
        self._cell_inv = np.linalg.inv(self._cell_matrix)
        self._num_atoms = 0
        self._num_cells = [int(i) for i in self._num_cells]
        self._cell_list = [[] for i in range(np.prod(self._num_cells))]
        self._cell_id_min = np.array([0, 0, 0])

    def update(self, positions: np.ndarray):
        cell_id = np.floor(np.dot(positions, self._cell_inv))
        self._num_atoms = cell_id.shape[0]
        self._cell_id_min = cell_id.min(0)
        for atom in range(self._num_atoms):
            self._cell_list[
                self._get_cell_index(*cell_id[atom])
            ].append(atom)

    @property
    def cutoff_radius(self):
        return self._cutoff_radius

    @property
    def pbc_matrix(self):
        return np.diag(self._pbc_diag)

    @property
    def cell_matrix(self):
        return self._cell_matrix

    @property
    def cell_inv(self):
        return self._cell_inv