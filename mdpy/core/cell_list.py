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
from .. import SPATIAL_DIM, env
from ..math import *
from ..unit import *
from ..error import *

class CellList:
    def __init__(self) -> None:
        self._cutoff_radius = env.NUMPY_FLOAT(0)
        self._pbc_diag = np.zeros(SPATIAL_DIM, env.NUMPY_FLOAT)

    def _get_cell_index(self, x: int, y: int, z: int):
        x -= self._cell_id_min[0]
        y -= self._cell_id_min[1]
        z -= self._cell_id_min[2]
        cell_index = [int(x), int(y), int(z)]
        for i, j in enumerate(cell_index):
            if j < 0:
                cell_index[i] = self._num_cells_vec[i] + j
            elif j >= self._num_cells_vec[i]:
                cell_index[i] -= self._num_cells_vec[i]
        x, y, z = cell_index
        index = x * self._num_cells_vec[1] * self._num_cells_vec[2] + y * self._num_cells_vec[2] + z       
        return index

    def __getitem__(self, keys):
        x, y, z = keys
        return self._cell_list[self._get_cell_index(x, y, z)]

    def get_neighbors(self, position: np.ndarray):
        if self._num_cells <= 27:
            return self._particles
        else:
            matrix_ids = []
            cell_id = np.floor(np.dot(position, self._cell_inv))
            for i in range(-1, 2):
                for j in range(-1, 2):
                    for k in range(-1, 2):
                        matrix_ids.extend(
                            self[cell_id[0]+i, cell_id[1]+j, cell_id[2]+k]
                        )
        return matrix_ids

    def _is_poor_defined(self, verbose=False):
        if self._cutoff_radius == 0:
            if verbose:
                print(
                    'Cutoff radius is poor defined, current value %.3f' 
                    %(self._cutoff_radius)
                )
            return True
        elif (self._pbc_diag == 0).all():
            if verbose:
                print(
                    'PBC is poor defined, current diag value is %s'
                    %(self._pbc_diag)
                )
            return True
        return False

    def _update_attributes(self):
        self._cell_matrix = np.ones(SPATIAL_DIM) * self._cutoff_radius
        self._num_cells_vec = np.floor(self._pbc_diag / self._cell_matrix)
        for i in self._num_cells_vec:
            if i == 0:
                raise CellListPoorDefinedError(
                    'The cutoff_radius is too large compared to the pbc box size'
                )
        self._num_cells = env.NUMPY_INT(np.prod(self._num_cells_vec))
        self._cell_matrix = np.diag(self._pbc_diag / self._num_cells_vec)
        self._cell_inv = np.linalg.inv(self._cell_matrix)
        self._num_particles = 0
        self._num_cells_vec = [int(i) for i in self._num_cells_vec]
        self._cell_list = [[] for _ in range(self._num_cells)]
        self._cell_id_min = np.array([0, 0, 0])

    def set_pbc_matrix(self, pbc_matrix: np.ndarray):
        self._pbc_diag = pbc_matrix.diagonal()
        if not self._is_poor_defined():
            self._update_attributes()

    def set_cutoff_radius(self, cutoff_radius):
        self._cutoff_radius = check_quantity_value(cutoff_radius, default_length_unit)
        if not self._is_poor_defined():
            self._update_attributes()

    def update(self, positions: np.ndarray):
        if not self._is_poor_defined():
            self._cell_list = [[] for _ in range(self._num_cells)]
            cell_id = np.floor(np.dot(positions, self._cell_inv))
            self._num_particles = cell_id.shape[0]
            self._particles = list(range(self._num_particles))
            self._cell_id_min = cell_id.min(0)
            for particle in range(self._num_particles):
                self._cell_list[
                    self._get_cell_index(*cell_id[particle])
                ].append(particle)
        else:
            self._is_poor_defined(verbose=True)
            raise CellListPoorDefinedError(
                'Cell list is poorly defined, please check CellList._is_poor_deined(verbose=True)'
            )

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