#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : tile_list.py
created time : 2022/05/04
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import numpy as np
import cupy as cp
import numba.cuda as cuda
from mdpy import SPATIAL_DIM
from mdpy.core import NUM_NEIGHBOR_CELLS, NUM_PARTICLES_PER_TILE
from mdpy.environment import *
from mdpy.utils import *
from mdpy.unit import *
from mdpy.error import *

SKIN_WIDTH = Quantity(1, angstrom)
THREAD_PER_BLOCK = 32

class TileList:
    def __init__(self, pbc_matrix: np.ndarray, skin_width=SKIN_WIDTH,) -> None:
        pbc_matrix = check_quantity_value(pbc_matrix, default_length_unit)
        self._pbc_matrix = check_pbc_matrix(pbc_matrix)
        self._pbc_diag = np.ascontiguousarray(self._pbc_matrix.diagonal(), dtype=NUMPY_FLOAT)
        self._half_pbc_diag = self._pbc_diag / 2
        self._skin_width = check_quantity_value(skin_width, default_length_unit)
        self.set_cutoff_radius(Quantity(1, angstrom))
        # Attribute
        self._num_cells_vec = np.ceil(self._pbc_diag / self._cell_width).astype(NUMPY_INT)
        self._particle_cell_index = None
        self._tile_list = None
        self._tile_cell_index = None
        self._cell_tile_information = None
        self._num_tiles = 0
        # Deivice attribute
        self._device_pbc_matrix = cp.array(self._pbc_matrix, CUPY_FLOAT)
        self._device_pbc_diag = cp.array(self._pbc_diag, CUPY_FLOAT)
        self._device_half_pbc_diag = cp.array(self._half_pbc_diag, CUPY_FLOAT)
        # Kernel
        self._update_tile_list = cuda.jit(nb.void(
            NUMBA_INT[::1], # sorted_particle_index
            NUMBA_INT[:, ::1], # cell_particle_information
            NUMBA_INT[:, ::1], # cell_tile_information
            NUMBA_INT[::1], # num_cells_vec
            NUMBA_INT[:, ::1], # tile_list
            NUMBA_INT[:, ::1] # tile_cell_index
        ))(self._update_tile_list_kernel)
        self._update_tile_neighbor = cuda.jit(nb.void(
            NUMBA_INT[:, ::1], # tile_list
            NUMBA_INT[:, ::1], # tile_cell_index
            NUMBA_INT[:, ::1], # cell_tile_information
            NUMBA_INT[::1], # num_cells_vec
            NUMBA_INT[:, ::1], # tile_neighbor
        ))(self._update_tile_neighbor_kernel)

    def set_pbc_matrix(self, pbc_matrix: np.ndarray):
        pbc_matrix = check_quantity_value(pbc_matrix, default_length_unit)
        self._pbc_matrix = check_pbc_matrix(pbc_matrix)
        self._pbc_diag = self._pbc_matrix.diagonal()
        self._half_pbc_diag = self._pbc_diag / 2
        self._device_pbc_matrix = cp.array(self._pbc_matrix, CUPY_FLOAT)
        self._device_pbc_diag = cp.array(self._pbc_diag, CUPY_FLOAT)
        self._device_half_pbc_diag = cp.array(self._half_pbc_diag, CUPY_FLOAT)

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
        self._cell_width = self._cutoff_radius + self._skin_width
        self._num_cells_vec = np.ceil(self._pbc_diag / self._cell_width).astype(NUMPY_INT)
        # Device attributes
        self._device_cutoff_radius = cp.array([self._cutoff_radius], CUPY_FLOAT)
        self._device_cell_width = cp.array([self._cell_width], CUPY_FLOAT)
        self._device_num_cells_vec = cp.array(self._num_cells_vec, CUPY_INT)

    @staticmethod
    def _update_cell_information_kernel(
        positions: cp.ndarray,
        num_cells_vec: np.ndarray,
        cell_width: cp.ndarray
    ):
        num_cells = np.prod(num_cells_vec)
        particle_cell_index = cp.floor(positions / cell_width).astype(CUPY_INT)
        particle_cell_index_single = (
            particle_cell_index[:, 2] + particle_cell_index[:, 1] * num_cells_vec[2] +
            particle_cell_index[:, 0] * num_cells_vec[2] * num_cells_vec[1]
        ) # z varies fastest to match the c order of reshape
        sorted_particle_index = cp.argsort(particle_cell_index_single).astype(CUPY_INT)
        sorted_particle_cell_index = particle_cell_index_single[sorted_particle_index]
        # Particle information
        num_particles_each_cell = cp.bincount(sorted_particle_cell_index, minlength=num_cells)
        particle_start_index = cp.zeros_like(num_particles_each_cell, dtype=CUPY_INT)
        particle_start_index[1:] = cp.cumsum(num_particles_each_cell)[:-1]
        cell_particle_information = cp.stack([
            particle_start_index,
            num_particles_each_cell
        ], axis=1).astype(CUPY_INT)
        # Tile information
        num_tiles_each_ceil = cp.ceil(num_particles_each_cell / NUM_PARTICLES_PER_TILE).astype(CUPY_INT)
        num_tiles = int(num_tiles_each_ceil.sum())
        max_num_tiles_per_cell = int(num_tiles_each_ceil.max())
        tile_start_index = cp.zeros_like(num_tiles_each_ceil, dtype=CUPY_INT)
        tile_start_index[1:] = cp.cumsum(num_tiles_each_ceil)[:-1]
        cell_tile_information = cp.stack([
            tile_start_index,
            num_tiles_each_ceil
        ], axis=1).astype(CUPY_INT)
        return (
            sorted_particle_index,
            cell_particle_information,
            cell_tile_information,
            num_tiles,
            max_num_tiles_per_cell
        )

    @staticmethod
    def _update_tile_list_kernel(
        sorted_particle_index,
        cell_particle_information,
        cell_tile_information,
        num_cells_vec,
        tile_list,
        tile_cell_index
    ):
        tile_index = cuda.grid(1)
        num_tiles = tile_list.shape[0]
        num_cells = cell_tile_information.shape[0]
        if tile_index >= num_tiles:
            return
        # Binary search of tile's cell index
        low, high = 0, num_cells - 1
        is_found = False
        while low <= high:
            mid = (low + high) // 2
            tile_start_index = cell_tile_information[mid, 0]
            if tile_index < tile_start_index:
                high = mid - 1
            elif tile_index > tile_start_index:
                low = mid + 1
            else:
                cell_index = mid
                is_found = True
                break
        if not is_found:
            cell_index = low - 1
        # Skip cell with 0 tile. The start index will be the same for those cells
        # E.g cell 10: [305, 0], cell 11: [305, 0], cell 12: [305, 2]
        while cell_particle_information[cell_index, 1] == 0:
            cell_index += 1
        # Get cell information
        cell_tile_start_index = cell_tile_information[cell_index, 0]
        cell_particle_start_index = cell_particle_information[cell_index, 0]
        cell_particle_end_index = cell_particle_start_index + cell_particle_information[cell_index, 1]
        # Get tile information
        tile_index_cur_cell = tile_index - cell_tile_start_index
        tile_particle_start_index = cell_particle_start_index + tile_index_cur_cell * NUM_PARTICLES_PER_TILE
        tile_particle_end_index = tile_particle_start_index + NUM_PARTICLES_PER_TILE
        if tile_particle_end_index >= cell_particle_end_index:
            tile_particle_end_index = cell_particle_end_index
        # Assign particles to tile
        tile_particle_index = 0

        dif = tile_particle_end_index - tile_particle_start_index
        if dif <= 0:
            print(cell_tile_start_index, cell_tile_information[cell_index, 1], tile_index)
        for particle_index in range(tile_particle_start_index, tile_particle_end_index):
            tile_list[tile_index, tile_particle_index] = sorted_particle_index[particle_index]
            tile_particle_index += 1
        # Get 3d cell index
        decomposition = cell_index
        num_cells_z = num_cells_vec[2]
        num_cells_yz = num_cells_vec[1] * num_cells_z
        cell_x = decomposition // num_cells_yz
        decomposition -= cell_x * num_cells_yz
        cell_y = decomposition // num_cells_z
        cell_z = decomposition - cell_y * num_cells_z
        tile_cell_index[tile_index, 0] = cell_x
        tile_cell_index[tile_index, 1] = cell_y
        tile_cell_index[tile_index, 2] = cell_z

    @staticmethod
    def _update_tile_neighbor_kernel(
        tile_list,
        tile_cell_index,
        cell_tile_information,
        num_cells_vec,
        tile_neighbors
    ):
        tile_id = cuda.grid(1)
        if tile_id >= tile_list.shape[0]:
            return
        num_cells_x = num_cells_vec[0]
        num_cells_y = num_cells_vec[1]
        num_cells_z = num_cells_vec[2]
        num_cells_yz = num_cells_y * num_cells_z
        central_cell_index = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_INT)
        neighbor_cell_index = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_INT)
        for i in range(SPATIAL_DIM):
            central_cell_index[i] = tile_cell_index[tile_id, i]
        cur_neighbor_tile_index = 0
        for i in range(-1, 2):
            neighbor_cell_index[0] = central_cell_index[0] + i
            if neighbor_cell_index[0] >= num_cells_x:
                neighbor_cell_index[0] -= num_cells_x
            elif neighbor_cell_index[0] < 0:
                neighbor_cell_index[0] += num_cells_x
            for j in range(-1, 2):
                neighbor_cell_index[1] = central_cell_index[1] + j
                if neighbor_cell_index[1] >= num_cells_y:
                    neighbor_cell_index[1] -= num_cells_y
                elif neighbor_cell_index[1] < 0:
                    neighbor_cell_index[1] += num_cells_y
                for k in range(-1, 2):
                    neighbor_cell_index[2] = central_cell_index[2] + k
                    if neighbor_cell_index[2] >= num_cells_z:
                        neighbor_cell_index[2] -= num_cells_z
                    elif neighbor_cell_index[2] < 0:
                        neighbor_cell_index[2] += num_cells_z
                    cell_index = (
                        neighbor_cell_index[2] + neighbor_cell_index[1] * num_cells_z +
                        neighbor_cell_index[0] * num_cells_yz
                    )
                    cur_tile_index = cell_tile_information[cell_index, 0]
                    for _ in range(cell_tile_information[cell_index, 1]):
                        tile_neighbors[tile_id, cur_neighbor_tile_index] = cur_tile_index
                        cur_neighbor_tile_index += 1
                        cur_tile_index += 1

    def update(self, positions: cp.ndarray):
        positive_postions = positions + self._device_half_pbc_diag
        result = self._update_cell_information_kernel(
            positive_postions, self._num_cells_vec, self._device_cell_width
        )
        sorted_particle_index = result[0]
        cell_particle_information = result[1]
        cell_tile_information = result[2]
        self._num_tiles = result[3]
        max_num_tiles_per_cell = result[4]
        # Create tile list
        block_per_grid = int(np.ceil(self._num_tiles / THREAD_PER_BLOCK))
        self._tile_list = cp.zeros((self._num_tiles, NUM_PARTICLES_PER_TILE), dtype=CUPY_INT) - 1
        self._tile_cell_index = cp.zeros((self._num_tiles, SPATIAL_DIM), dtype=CUPY_INT)
        self._update_tile_list[block_per_grid, THREAD_PER_BLOCK](
            sorted_particle_index,
            cell_particle_information,
            cell_tile_information,
            self._device_num_cells_vec,
            self._tile_list,
            self._tile_cell_index
        )
        self._tile_neighbors = cp.zeros((self._num_tiles, max_num_tiles_per_cell * NUM_NEIGHBOR_CELLS), dtype=CUPY_INT) - 1
        self._update_tile_neighbor[block_per_grid, THREAD_PER_BLOCK](
            self._tile_list,
            self._tile_cell_index,
            cell_tile_information,
            self._device_num_cells_vec,
            self._tile_neighbors
        )

    @property
    def cell_width(self):
        return self._cell_width

    @property
    def cutoff_radius(self):
        return self._cutoff_radius

    @property
    def pbc_matrix(self):
        return self._pbc_matrix

    @property
    def num_tiles(self) -> int:
        return self._num_tiles

    @property
    def tile_list(self) -> cp.ndarray:
        return self._tile_list

    @property
    def tile_neighbors(self) -> cp.ndarray:
        return self._tile_neighbors

    @property
    def tile_cell_index(self) -> cp.ndarray:
        return self._tile_cell_index

    @property
    def num_cells_vec(self):
        return self._num_cells_vec

    @property
    def device_num_cells_vec(self):
        return self._device_num_cells_vec

if __name__ == '__main__':
    import mdpy as md
    pdb = md.io.PDBParser('/home/zhenyuwei/nutstore/ZhenyuWei/Note_Research/mdpy/mdpy/benchmark/str/medium.pdb')
    positions = cp.array(pdb.positions, CUPY_FLOAT)

    tile_list = TileList(pdb.pbc_matrix)
    tile_list.set_cutoff_radius(8)

    tile_list.update(positions)

    tile_list_array = tile_list.tile_list.get()
    print(np.count_nonzero(tile_list_array+1))
    tile_index = 10
    print(tile_list.num_cells_vec)
    print(tile_list.tile_cell_index[tile_index, :])
    neighbors = tile_list.tile_neighbor[tile_index, :].get()
    for i in neighbors:
        print(tile_list.tile_cell_index[i, :])
    # for i in range(tile_list._num_cells_vec[0]):
    #     for j in range(tile_list._num_cells_vec[1]):
    #         for k in range(tile_list._num_cells_vec[2]):
    #             index = k+j*tile_list._num_cells_vec[2]+i*tile_list._num_cells_vec[1]*tile_list._num_cells_vec[2]
    #             print((i, j, k), tile_list.cell_tile_information[index, :])

    # tile_index = 494
    # low, high = 0, tile_list.cell_tile_information.shape[0] - 1
    # is_found = False
    # while low <= high:
    #     mid = (low + high) // 2
    #     print(low, high, mid, tile_list.cell_tile_information[mid, 0])
    #     if tile_index < tile_list.cell_tile_information[mid, 0]:
    #         high = mid - 1
    #     elif tile_index > tile_list.cell_tile_information[mid, 0]:
    #         low = mid + 1
    #     else:
    #         cell_index = mid
    #         is_found = True
    #         break
    # if not is_found:
    #     cell_index = low - 1
    # print(cell_index, tile_list.cell_tile_information[cell_index, :])
    # decomposition = cell_index
    # cell_x = decomposition // (tile_list._num_cells_vec[1]*tile_list._num_cells_vec[2])
    # decomposition -= cell_x * tile_list._num_cells_vec[1]*tile_list._num_cells_vec[2]
    # cell_y = decomposition // (tile_list._num_cells_vec[2])
    # cell_z = decomposition - cell_y * tile_list._num_cells_vec[2]
    # print(cell_x, cell_y, cell_z)
    # decomposition -= cell_x *
    # print(tile_list.tile_list[tile_id, :])
    # print(tile_list.tile_cell_index[tile_id, :])
    # for i in tile_list.tile_list[tile_id, :]:
    #     if i == -1:
    #         break
    #     print(positions[i, :] + tile_list._device_half_pbc_diag)

    import time
    ts = time.time()
    epoch = 30
    for i in range(epoch):
        s = time.time()
        tile_list.update(positions)
        e = time.time()
        print('Run xxx for %s s' %(e-s))
    te = time.time()
    print('Run xxx for %s s' %((te-ts)/epoch))