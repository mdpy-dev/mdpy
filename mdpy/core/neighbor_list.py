#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : neighbor_list.py
created time : 2022/04/25
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

CELL_WIDTH = Quantity(3, angstrom)
SKIN_WIDTH = Quantity(1, angstrom)
NUM_MIN_NEIGHBORS = 300
THREAD_PER_BLOCK = 64

class NeighborList:
    def __init__(self, pbc_matrix: np.ndarray, skin_width=SKIN_WIDTH, cell_width=CELL_WIDTH) -> None:
        pbc_matrix = check_quantity_value(pbc_matrix, default_length_unit)
        self._pbc_matrix = check_pbc_matrix(pbc_matrix)
        self._pbc_diag = np.ascontiguousarray(self._pbc_matrix.diagonal(), dtype=NUMPY_FLOAT)
        self._pbc_volume = np.prod(self._pbc_diag)
        self._skin_width = check_quantity_value(skin_width, default_length_unit)
        self._cell_width = check_quantity_value(cell_width, default_length_unit)
        self.set_cutoff_radius(Quantity(1, angstrom))
        # Attributes
        self._num_cells_vec = np.ceil(self._pbc_diag / self._cell_width).astype(NUMPY_INT)
        self._cell_list = np.zeros([
            self._num_cells_vec[0], self._num_cells_vec[1], self._num_cells_vec[1],
        ], dtype=NUMPY_INT)
        self._particle_cell_index = None
        self._neighbor_ceil_shift = None
        self._num_neighbor_cells = None
        self._neighbor_list = None
        self._neighbor_vec_list = None
        # Device attribute
        self._device_skin_width = cp.array([self._skin_width], CUPY_FLOAT)
        self._device_pbc_matrix = cp.array(self._pbc_matrix, CUPY_FLOAT)
        self._device_num_cells_vec = cp.array(self._num_cells_vec, CUPY_INT)
        # Kernels
        self._update_cell_list = nb.njit((
            NUMBA_FLOAT[:, ::1], # positions
            NUMBA_FLOAT[::1], # pbc_diag
            NUMBA_INT[::1], # num_cells_vec
            NUMBA_FLOAT, # cell_width
        ))(self._update_cell_list_kernel)
        self._update_neighbor_list = cuda.jit(nb.void(
            NUMBA_FLOAT[:, ::1], # positions
            NUMBA_FLOAT[:, ::1], # pbc_matrix
            NUMBA_INT[:, ::1], # particle_cell_index
            NUMBA_INT[:, :, :, ::1], # cell_list
            NUMBA_INT[::1], # num_cell_vec
            NUMBA_FLOAT[::1], # cutoff_radius
            NUMBA_FLOAT[::1], # skin_width
            NUMBA_INT[::1], # neighbor_ceil_shift
            NUMBA_INT[:, ::1], # neighbor_list
            NUMBA_FLOAT[:, :, ::1], # neighbor_vec_list
        ))(self._update_neighbor_list_kernel)
        self._update_neighbor_vec_list = cuda.jit(nb.void(
            NUMBA_FLOAT[:, ::1], # positions
            NUMBA_FLOAT[:, ::1], # pbc_matrix
            NUMBA_INT[:, ::1], # neighbor_list
            NUMBA_FLOAT[:, :, ::1], # neighbor_vec_list
        ))(self._update_neighbor_vec_list_kernel)

    def set_pbc_matrix(self, pbc_matrix: np.ndarray):
        pbc_matrix = check_quantity_value(pbc_matrix, default_length_unit)
        self._pbc_matrix = check_pbc_matrix(pbc_matrix)
        self._pbc_diag = self._pbc_matrix.diagonal()
        self._device_pbc_matrix = cp.array(self._pbc_matrix, CUPY_FLOAT)

    def set_cutoff_radius(self, cutoff_radius):
        cutoff_radius = check_quantity_value(cutoff_radius, default_length_unit)
        if cutoff_radius == 0:
            raise NeighborListPoorDefinedError(
                'Cutoff radius is poor defined, current value %.3f'
                %(cutoff_radius)
            )
        cell_matrix = np.ones(SPATIAL_DIM) * cutoff_radius
        num_cells_vec = np.floor(self._pbc_diag / cell_matrix).astype(NUMPY_INT)
        for i in num_cells_vec:
            if i < 2:
                raise NeighborListPoorDefinedError(
                    'The cutoff_radius is too large to create cell list'
                )
        # Attributes
        self._cutoff_radius = cutoff_radius
        self._neighbor_ceil_shift = np.ceil((self._cutoff_radius+self._skin_width) / self._cell_width).astype(NUMPY_INT)
        self._num_neighbor_cells = (self._neighbor_ceil_shift * 2 + 1)**3 # +1 for center cell
        self._cutoff_volume = 4 / 3 * np.pi * (self._cutoff_radius+self._skin_width)**3
        # Device attributes
        self._device_cutoff_radius = cp.array([cutoff_radius], CUPY_FLOAT)
        self._device_neighbor_ceil_shift = cp.array([self._neighbor_ceil_shift], CUPY_INT)

    @staticmethod
    def _update_cell_list_kernel(positions, pbc_diag, num_cells_vec, cell_width):
        num_particles = positions.shape[0]
        positions = positions + pbc_diag / 2
        particle_cell_index = np.floor(positions / cell_width).astype(NUMPY_INT)
        # Set variable
        num_x, num_y, num_z = num_cells_vec
        num_particles_each_cell = np.zeros((num_x, num_y, num_z), dtype=NUMPY_INT) # Number of particles in each cells
        for particle in range(num_particles):
            x, y, z = particle_cell_index[particle, :]
            num_particles_each_cell[x, y, z] += 1
        # Build cell list
        max_num_particles_per_cell = NUMPY_INT(num_particles_each_cell.max())
        cell_list = np.zeros((
            num_x, num_y, num_z, max_num_particles_per_cell
        ), dtype=NUMPY_INT) - 1 # -1 for padding value
        cur_cell_flag = np.zeros((num_x, num_y, num_z), dtype=NUMPY_INT)
        for particle in range(num_particles):
            x, y, z = particle_cell_index[particle]
            cell_list[x, y, z, cur_cell_flag[x, y, z]] = particle
            cur_cell_flag[x, y, z] += 1
        return particle_cell_index, cell_list

    @staticmethod
    def _update_neighbor_list_kernel(
        positions,
        pbc_matrix,
        particle_cell_index,
        cell_list,
        num_cell_vec,
        cutoff_radius,
        skin_width,
        neighbor_ceil_shift,
        neighbor_list,
        neighbor_vec_list
    ):
        particle_id1 = cuda.grid(1)
        if particle_id1 >= positions.shape[0]:
            return None
        # Shared array
        shared_pbc_matrix = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        shared_half_pbc_matrix = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        shared_num_cell_vec = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_INT)
        thread_x = cuda.threadIdx.x
        if thread_x == 0:
            for i in range(SPATIAL_DIM):
                shared_pbc_matrix[i] = pbc_matrix[i, i]
                shared_half_pbc_matrix[i] = shared_pbc_matrix[i] / 2
        if thread_x == 1:
            for i in range(SPATIAL_DIM):
                shared_num_cell_vec[i] = num_cell_vec[i]
        cuda.syncthreads()
        # Read shared data
        cutoff_radius = cutoff_radius[0] + skin_width[0]
        neighbor_ceil_shift = neighbor_ceil_shift[0]
        neighbor_ceil_lower = - neighbor_ceil_shift - 1
        neighbor_ceil_upper = neighbor_ceil_shift + 2
        central_cell = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_INT)
        position_id1 = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        for i in range(SPATIAL_DIM):
            central_cell[i] = particle_cell_index[particle_id1, i]
            position_id1[i] = positions[particle_id1, i]
        neighbor_index = 0
        r = 0
        vec = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        for i in range(neighbor_ceil_lower, neighbor_ceil_upper):
            cell_x = central_cell[0] + i
            if cell_x < 0:
                cell_x += shared_num_cell_vec[0]
            elif cell_x >= shared_num_cell_vec[0]:
                cell_x -= shared_num_cell_vec[0]
            for j in range(neighbor_ceil_lower, neighbor_ceil_upper):
                cell_y = central_cell[1] + j
                if cell_y < 0:
                    cell_y += shared_num_cell_vec[1]
                elif cell_y >= shared_num_cell_vec[1]:
                    cell_y -= shared_num_cell_vec[1]
                for k in range(neighbor_ceil_lower, neighbor_ceil_upper):
                    cell_z = central_cell[2] + k
                    if cell_z < 0:
                        cell_z += shared_num_cell_vec[2]
                    elif cell_z >= shared_num_cell_vec[2]:
                        cell_z -= shared_num_cell_vec[2]

                    for particle_id2 in cell_list[cell_x, cell_y, cell_z, :]:
                        if particle_id2 == -1:
                            break
                        if particle_id2 == particle_id1:
                            continue
                        r = 0
                        for i in range(SPATIAL_DIM):
                            vec[i] = (positions[particle_id2, i] - position_id1[i])
                            if vec[i] >= shared_half_pbc_matrix[i]:
                                vec[i] -= shared_pbc_matrix[i]
                            elif vec[i] <= -shared_half_pbc_matrix[i]:
                                vec[i] += shared_pbc_matrix[i]
                            r += vec[i]**2
                        r = math.sqrt(r)
                        if r <= (cutoff_radius):
                            cuda.atomic.add(neighbor_list, (particle_id1, neighbor_index), 1 + particle_id2)
                            cuda.atomic.add(neighbor_vec_list, (particle_id1, neighbor_index, 0), r)
                            cuda.atomic.add(neighbor_vec_list, (particle_id1, neighbor_index, 1), vec[0] / r)
                            cuda.atomic.add(neighbor_vec_list, (particle_id1, neighbor_index, 2), vec[1] / r)
                            cuda.atomic.add(neighbor_vec_list, (particle_id1, neighbor_index, 3), vec[2] / r)
                            neighbor_index += 1

    @staticmethod
    def _update_neighbor_vec_list_kernel(
        positions,
        pbc_matrix,
        neighbor_list,
        neighbor_vec_list
    ):
        particle_id1 = cuda.grid(1)
        if particle_id1 >= positions.shape[0]:
            return None
        # Shared array
        shared_pbc_matrix = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        shared_half_pbc_matrix = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        thread_x = cuda.threadIdx.x
        if thread_x == 0:
            for i in range(SPATIAL_DIM):
                shared_pbc_matrix[i] = pbc_matrix[i, i]
                shared_half_pbc_matrix[i] = shared_pbc_matrix[i] / 2
        cuda.syncthreads()
        position_id1 = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        for i in range(SPATIAL_DIM):
            position_id1[i] = positions[particle_id1, i]
        neighbor_index = 0
        vec = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        for particle_id2 in neighbor_list[particle_id1, :]:
            if particle_id2 == -1:
                break
            if particle_id1 == particle_id2:
                continue
            r = 0
            for i in range(SPATIAL_DIM):
                vec[i] = (positions[particle_id2, i] - position_id1[i])
                if vec[i] >= shared_half_pbc_matrix[i]:
                    vec[i] -= shared_pbc_matrix[i]
                elif vec[i] <= -shared_half_pbc_matrix[i]:
                    vec[i] += shared_pbc_matrix[i]
                r += vec[i]**2
            r = math.sqrt(r)
            cuda.atomic.add(
                neighbor_vec_list, (particle_id1, neighbor_index, 0),
                r - neighbor_vec_list[particle_id1, neighbor_index, 0]
            )
            cuda.atomic.add(
                neighbor_vec_list, (particle_id1, neighbor_index, 1),
                vec[0] / r - neighbor_vec_list[particle_id1, neighbor_index, 1]
            )
            cuda.atomic.add(
                neighbor_vec_list, (particle_id1, neighbor_index, 2),
                vec[1] / r - neighbor_vec_list[particle_id1, neighbor_index, 2]
            )
            cuda.atomic.add(
                neighbor_vec_list, (particle_id1, neighbor_index, 3),
                vec[2] / r - neighbor_vec_list[particle_id1, neighbor_index, 3]
            )
            neighbor_index += 1

    def _judge_max_num_neighbors_per_particle(self, num_particles):
        max_num_neighbors_per_particle = int(np.ceil(
            num_particles / self._pbc_volume * self._cutoff_volume * 1.3
        ))
        if num_particles < NUM_MIN_NEIGHBORS:
            return num_particles
        elif max_num_neighbors_per_particle < NUM_MIN_NEIGHBORS:
            return NUM_MIN_NEIGHBORS
        return max_num_neighbors_per_particle

    def update(self, positions: cp.ndarray, is_update_neighbor_list=True):
        num_particles = positions.shape[0]
        block_per_grid = int(np.ceil(
            num_particles / THREAD_PER_BLOCK
        ))
        if is_update_neighbor_list:
            max_num_neighbors_per_particle = self._judge_max_num_neighbors_per_particle(num_particles)
            self._particle_cell_index, self._cell_list = self._update_cell_list(
                positions.get(), self._pbc_diag,
                self._num_cells_vec, self._cell_width
            )
            self._neighbor_list = cp.zeros((positions.shape[0], max_num_neighbors_per_particle), CUPY_INT) - 1
            self._neighbor_vec_list = cp.zeros((positions.shape[0], max_num_neighbors_per_particle, SPATIAL_DIM+1), CUPY_FLOAT)
            device_particle_cell_index = cp.array(self._particle_cell_index, CUPY_INT)
            device_cell_list = cp.array(self._cell_list, CUPY_INT)
            self._update_neighbor_list[block_per_grid, THREAD_PER_BLOCK](
                positions,
                self._device_pbc_matrix,
                device_particle_cell_index,
                device_cell_list,
                self._device_num_cells_vec,
                self._device_cutoff_radius,
                self._device_skin_width,
                self._device_neighbor_ceil_shift,
                self._neighbor_list,
                self._neighbor_vec_list
            )
        else:
            self._update_neighbor_vec_list[block_per_grid, THREAD_PER_BLOCK](
                positions,
                self._device_pbc_matrix,
                self._neighbor_list,
                self._neighbor_vec_list
            )

    @property
    def cutoff_radius(self):
        return self._cutoff_radius

    @property
    def pbc_matrix(self):
        return self._pbc_matrix

    @property
    def neighbor_list(self):
        return self._neighbor_list

    @property
    def neighbor_vec_list(self):
        return self._neighbor_vec_list
