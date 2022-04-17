#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : cell_list.py
created time : 2021/10/27
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import math
import numpy as np
import numba as nb
import cupy as cp
import numba.cuda as cuda
from mdpy import SPATIAL_DIM
from mdpy.environment import *
from mdpy.utils import *
from mdpy.unit import *
from mdpy.error import *

CELL_LIST_SKIN = Quantity(1, angstrom)
THREAD_PER_BLOCK = 64

class CellList:
    def __init__(self, pbc_matrix: np.ndarray) -> None:
        # Read input
        self.set_pbc_matrix(pbc_matrix, is_update=False)
        self.set_cutoff_radius(Quantity(1, angstrom), is_update=False)
        # Skin for particles on boundary to be included in cell_list
        self._cell_list_skin = check_quantity_value(CELL_LIST_SKIN, default_length_unit)
        # Cell list attributes
        self._particle_cell_index = None # N x 3
        self._cell_list = None # n x n x n x Nb
        self._num_particles_per_cell = 0
        self._update_attributes()
        # Device arrays
        self._device_particle_cell_index = None
        self._device_cell_list = None
        self._device_num_particles_per_cell = None
        self._device_cell_size = None
        # Cell list construction kernel
        self._kernel = nb.njit((
            NUMBA_FLOAT[:, ::1], # positions
            NUMBA_FLOAT[:, ::1], # cell_inv
            NUMBA_INT[::1] # num_cell_vec
        ))(self.kernel)
        self._update = cuda.jit(nb.void(
            NUMBA_FLOAT[:, ::1], # positions
            NUMBA_FLOAT[:, ::1], # pbc_matrix
            NUMBA_INT[::1], # cell_size
            NUMBA_INT[:, ::1], # particle_cell_index
            NUMBA_INT[:, :, ::1], # cell_num_particles
            NUMBA_INT[:, :, :, ::1]# cell_list
        ))(self._update_kernel)

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
        self._device_pbc_matrix = cp.array(self._pbc_matrix, CUPY_FLOAT)
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
        cell_size = np.floor(self._pbc_diag / cell_matrix).astype(NUMPY_INT)
        for i in cell_size:
            if i < 2:
                raise CellListPoorDefinedError(
                    'The cutoff_radius is too large to create cell list'
                )
        self._cutoff_radius = cutoff_radius
        if is_update:
            self._update_attributes()

    def _update_attributes(self):
        self._cell_matrix = np.ones(SPATIAL_DIM) * self._cutoff_radius
        self._cell_size = np.floor((self._pbc_diag + self._cell_list_skin) / self._cell_matrix).astype(NUMPY_INT)
        # Construct at least 27 cells
        self._cell_size[self._cell_size < 3] = 3
        self._num_cells = NUMPY_INT(np.prod(self._cell_size))
        self._cell_matrix = np.diag((self._pbc_diag + self._cell_list_skin) / self._cell_size).astype(NUMPY_FLOAT)
        self._cell_inv = np.linalg.inv(self._cell_matrix)
        # Device array
        self._device_cell_size = cp.array(self._cell_size, CUPY_INT)

    @staticmethod
    def _update_kernel(
        positions,
        pbc_matrix,
        cell_size,
        particle_cell_index,
        cell_num_particles,
        cell_list
    ):
        particle_id = cuda.grid(1)
        if particle_id >= positions.shape[0]:
            return None
        # Shared array
        thread_x = cuda.threadIdx.x
        # PBC matrix
        shared_pbc_matrix = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        shared_half_pbc_matrix = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        # num_cell_vec
        shared_cell_size = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_INT)
        if thread_x == 0:
            shared_pbc_matrix[0] = pbc_matrix[0, 0]
            shared_pbc_matrix[1] = pbc_matrix[1, 1]
            shared_pbc_matrix[2] = pbc_matrix[2, 2]
            shared_half_pbc_matrix[0] = shared_pbc_matrix[0] / 2
            shared_half_pbc_matrix[1] = shared_pbc_matrix[1] / 2
            shared_half_pbc_matrix[2] = shared_pbc_matrix[2] / 2
        elif thread_x == 1:
            shared_cell_size[0] = cell_size[0]
            shared_cell_size[1] = cell_size[1]
            shared_cell_size[2] = cell_size[2]
        cuda.syncthreads()

        position_x = positions[particle_id, 0] + shared_half_pbc_matrix[0]
        position_y = positions[particle_id, 1] + shared_half_pbc_matrix[1]
        position_z = positions[particle_id, 2] + shared_half_pbc_matrix[2]
        cell_index = cuda.local.array((SPATIAL_DIM), NUMBA_INT)
        cell_index[0] = math.floor(position_x / shared_pbc_matrix[0] * shared_cell_size[0])
        cell_index[1] = math.floor(position_y / shared_pbc_matrix[1] * shared_cell_size[1])
        cell_index[2] = math.floor(position_z / shared_pbc_matrix[2] * shared_cell_size[2])

        cuda.atomic.compare_and_swap(
            cell_list[cell_index[0], cell_index[1], cell_index[2], :],
            -1, particle_id
        )
        cuda.atomic.add(particle_cell_index, (particle_id, 0), cell_index[0])
        cuda.atomic.add(particle_cell_index, (particle_id, 1), cell_index[1])
        cuda.atomic.add(particle_cell_index, (particle_id, 2), cell_index[2])
        cuda.atomic.add(cell_num_particles, (cell_index[0], cell_index[1], cell_index[2]), 1)
        cuda.atomic.add(particle_cell_index, (particle_id, 3), cell_num_particles[cell_index[0], cell_index[1], cell_index[2]])

    def update(self, positions: np.ndarray):
        # Set the position to positive value
        # Ensure the id calculated by matrix dot corresponds to the cell_list index

        # num_particles = positions.shape[0]
        # particle_cell_index = cp.zeros([num_particles, SPATIAL_DIM + 1], CUPY_INT)
        # cell_num_particles = cp.zeros(self._cell_size, CUPY_INT)
        # cell_list = cp.ones([
        #     self._cell_size[0], self._cell_size[1], self._cell_size[2],
        #     num_particles // self._num_cells * 2
        # ], CUPY_INT) * -1
        # block_per_grid = int(np.ceil(num_particles // THREAD_PER_BLOCK))
        # self._update[block_per_grid, THREAD_PER_BLOCK](
        #     cp.array(positions, CUPY_FLOAT),
        #     self._device_pbc_matrix,
        #     self._device_cell_size,
        #     particle_cell_index,
        #     cell_num_particles,
        #     cell_list
        # )
        # self._num_particles_per_cell = cell_num_particles.max()
        # # Device array
        # self._device_particle_cell_index = particle_cell_index
        # self._device_cell_list = cell_list[:, :, :self._num_particles_per_cell]

        positive_position = positions + self._pbc_diag / 2
        self._particle_cell_index, self._cell_list = self._kernel(
            positive_position, self._cell_inv, self._cell_size
        )
        # self._num_particles_per_cell = self._cell_list.shape[3]
        # print(cell_list[0, 0, 0, :])
        # print(self._cell_list[0, 0, 0, :])

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
            x, y, z = particle_cell_index[particle, :]
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
    def device_particle_cell_index(self):
        return self._device_particle_cell_index

    @property
    def cell_list(self):
        return self._cell_list

    @property
    def device_cell_list(self):
        return self._device_cell_list

    @property
    def num_cell_vec(self):
        return self._cell_size

    @property
    def device_num_cell_vec(self):
        return self._device_cell_size

    @property
    def num_particles_per_cell(self):
        return self._num_particles_per_cell

    @property
    def device_num_particles_per_cell(self):
        return self._device_num_particles_per_cell