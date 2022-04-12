#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : cell_list.py
created time : 2021/10/27
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import numpy as np
import numba as nb
from mdpy import SPATIAL_DIM, env
from mdpy.utils import *
from mdpy.unit import *
from mdpy.error import *

CELL_LIST_SKIN = Quantity(1, angstrom)

class CellList:
    def __init__(self, pbc_matrix: np.ndarray) -> None:
        # Read input
        self.set_pbc_matrix(pbc_matrix, is_update=False)
        self.set_cutoff_radius(Quantity(1, angstrom), is_update=False)
        # Skin for particles on boundary to be included in cell_list
        self._cell_list_skin = check_quantity_value(CELL_LIST_SKIN, default_length_unit)
        # Cell list attributes
        self._num_particles = 0
        self._particle_cell_index = None # N x 3
        self._cell_list = None # n x n x n x Nb
        self._num_particles_per_cell = 0
        self._update_attributes()
        # Cell list construction kernel
        self._kernel = nb.njit(
            (env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[:, ::1], env.NUMBA_INT[::1])
        )(self.kernel)

    def __repr__(self) -> str:
        x, y, z, _ = self._cell_list.shape
        return '<mdpy.core.CellList object with %d x %d x %d cells at %x>' %(
            x, y, z, id(self)
        )

    __str__ = __repr__

    def __getitem__(self, matrix_id):
        x, y, z = self._particle_cell_index[matrix_id, :]
        return self._cell_list[x, y, z, :]

    def set_pbc_matrix(self, pbc_matrix: np.ndarray, is_update=True):
        pbc_matrix = check_quantity_value(pbc_matrix, default_length_unit)
        self._pbc_matrix = check_pbc_matrix(pbc_matrix)
        self._pbc_diag = self._pbc_matrix.diagonal()
        if is_update:
            self._update_attributes()

    def set_cutoff_radius(self, cutoff_radius, is_update=True):
        cutoff_radius = check_quantity_value(cutoff_radius, default_length_unit)
        if cutoff_radius == 0:
            raise CellListPoorDefinedError(
                'Cutoff radius is poor defined, current value %.3f'
                %(cutoff_radius)
            )
        cell_matrix = np.ones(SPATIAL_DIM) * cutoff_radius
        num_cells_vec = np.floor(self._pbc_diag / cell_matrix).astype(env.NUMPY_INT)
        for i in num_cells_vec:
            if i < 2:
                raise CellListPoorDefinedError(
                    'The cutoff_radius is too large to create cell list'
                )
        self._cutoff_radius = cutoff_radius
        if is_update:
            self._update_attributes()

    def _update_attributes(self):
        self._cell_matrix = np.ones(SPATIAL_DIM) * self._cutoff_radius
        self._num_cells_vec = np.floor((self._pbc_diag + self._cell_list_skin) / self._cell_matrix).astype(env.NUMPY_INT)
        # Construct at least 27 cells
        self._num_cells_vec[self._num_cells_vec < 3] = 3
        self._num_cells = env.NUMPY_INT(np.prod(self._num_cells_vec))
        self._cell_matrix = np.diag((self._pbc_diag + self._cell_list_skin) / self._num_cells_vec).astype(env.NUMPY_FLOAT)
        self._cell_inv = np.linalg.inv(self._cell_matrix)

    def update(self, positions: np.ndarray):
        # Set the position to positive value
        # Ensure the id calculated by matrix dot corresponds to the cell_list index
        positive_position = positions + self._pbc_diag / 2
        self._particle_cell_index, self._cell_list = self._kernel(
            positive_position, self._cell_inv, self._num_cells_vec
        )
        self._num_particles_per_cell = self._cell_list.shape[3]

    @staticmethod
    def kernel(positions: np.ndarray, cell_inv: np.ndarray, num_cell_vec: np.ndarray):
        # Read input
        int_type = num_cell_vec.dtype
        num_particles = positions.shape[0]
        # Set variables
        num_x, num_y, num_z = num_cell_vec
        num_cell_particles = np.zeros((num_x, num_y, num_z), dtype=int_type) # Number of particles in each cells
        # Assign particles
        particle_cell_index = np.floor(np.dot(positions, cell_inv)).astype(int_type) # The cell index of each particle
        for particle in range(num_particles):
            x, y, z = particle_cell_index[particle]
            num_cell_particles[x, y, z] += 1
        # Build cell list
        max_num_cell_particles = num_cell_particles.max() # The number of particles of cell that contain the most particles
        cell_list = np.ones((
            num_cell_vec[0], num_cell_vec[1],
            num_cell_vec[2], max_num_cell_particles
        ), dtype=int_type) * -1
        cur_cell_flag = np.zeros_like(num_cell_particles)
        for particle in range(num_particles):
            x, y, z = particle_cell_index[particle]
            cell_list[x, y, z, cur_cell_flag[x, y, z]] = particle
            cur_cell_flag[x, y, z] += 1
        return particle_cell_index.astype(int_type), cell_list.astype(int_type)

    @property
    def cutoff_radius(self):
        return self._cutoff_radius

    @property
    def pbc_matrix(self):
        return self._pbc_matrix

    @property
    def cell_matrix(self):
        return self._cell_matrix

    @property
    def cell_inv(self):
        return self._cell_inv

    @property
    def particle_cell_index(self):
        return self._particle_cell_index

    @property
    def cell_list(self):
        return self._cell_list

    @property
    def num_cell_vec(self):
        return self._num_cells_vec

    @property
    def num_particles_per_cell(self):
        return self._num_particles_per_cell