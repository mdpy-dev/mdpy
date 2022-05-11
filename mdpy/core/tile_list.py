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
    def __init__(self, pbc_matrix: np.ndarray, skin_width=SKIN_WIDTH) -> None:
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
            NUMBA_INT[::1], # sorted_matrix_index
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
        self._sort_int_matrix = cuda.jit(nb.void(
            NUMBA_INT[:, ::1], # unsorted_matrix
            NUMBA_INT[::1], # sorted_matrix_mapping_index
            NUMBA_INT[:, ::1] # sorted_matrix
        ))(self._sort_matrix_kernel)
        self._sort_float_matrix = cuda.jit(nb.void(
            NUMBA_FLOAT[:, ::1], # unsorted_matrix
            NUMBA_INT[::1], # sorted_matrix_mapping_index
            NUMBA_FLOAT[:, ::1] # sorted_matrix
        ))(self._sort_matrix_kernel)
        self._unsort_int_matrix = cuda.jit(nb.void(
            NUMBA_INT[:, ::1], # sorted_matrix
            NUMBA_INT[::1], # sorted_matrix_mapping_index
            NUMBA_INT[:, ::1] # unsorted_matrix
        ))(self._unsort_matrix_kernel)
        self._unsort_float_matrix = cuda.jit(nb.void(
            NUMBA_FLOAT[:, ::1], # sorted_matrix
            NUMBA_INT[::1], # sorted_matrix_mapping_index
            NUMBA_FLOAT[:, ::1] # unsorted_matrix
        ))(self._unsort_matrix_kernel)
        self._generate_mask_map = cuda.jit(nb.void(
            NUMBA_INT[:, ::1], # particle_infomation
            NUMBA_INT[:, ::1], # tile_neighbor
            NUMBA_INT[::1], # sorted_matrix_mapping_index
            NUMBA_BIT[:, ::1] # mask_map
        ))(self._generate_mask_map_kernel)

    def set_pbc_matrix(self, pbc_matrix: np.ndarray) -> None:
        pbc_matrix = check_quantity_value(pbc_matrix, default_length_unit)
        self._pbc_matrix = check_pbc_matrix(pbc_matrix)
        self._pbc_diag = self._pbc_matrix.diagonal()
        self._half_pbc_diag = self._pbc_diag / 2
        self._device_pbc_matrix = cp.array(self._pbc_matrix, CUPY_FLOAT)
        self._device_pbc_diag = cp.array(self._pbc_diag, CUPY_FLOAT)
        self._device_half_pbc_diag = cp.array(self._half_pbc_diag, CUPY_FLOAT)

    def set_cutoff_radius(self, cutoff_radius) -> None:
        cutoff_radius = check_quantity_value(cutoff_radius, default_length_unit)
        if cutoff_radius == 0:
            raise TileListPoorDefinedError(
                'Cutoff radius is poor defined, current value %.3f'
                %(cutoff_radius)
            )
        cell_matrix = np.ones(SPATIAL_DIM) * cutoff_radius
        num_cells_vec = np.floor(self._pbc_diag / cell_matrix).astype(NUMPY_INT)
        for i in num_cells_vec:
            if i < 2:
                raise TileListPoorDefinedError(
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
        sorted_matrix_mapping_index,
        tile_list,
        tile_cell_index,
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
        sorted_matrix_index = tile_index * NUM_PARTICLES_PER_TILE # The index of sorted_particle
        for index in range(tile_particle_start_index, tile_particle_end_index):
            particle_index = sorted_particle_index[index]
            tile_list[tile_index, tile_particle_index] = particle_index
            sorted_matrix_mapping_index[sorted_matrix_index] = particle_index
            tile_particle_index += 1
            sorted_matrix_index += 1
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

    def update(self, positions: cp.ndarray) -> None:
        self._num_particles = positions.shape[0]
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
        self._sorted_matrix_mapping_index = cp.zeros(self._num_tiles*NUM_PARTICLES_PER_TILE, dtype=CUPY_INT) - 1
        self._update_tile_list[block_per_grid, THREAD_PER_BLOCK](
            sorted_particle_index,
            cell_particle_information,
            cell_tile_information,
            self._device_num_cells_vec,
            self._sorted_matrix_mapping_index,
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

    def sort_matrix(self, unsorted_matrix: cp.ndarray) -> cp.ndarray:
        matrix_type = unsorted_matrix.dtype
        sorted_matrix = cp.zeros(
            (unsorted_matrix.shape[1], self._num_tiles*NUM_PARTICLES_PER_TILE),
            dtype=matrix_type
        )
        thread_per_block = 32
        block_per_grid = self._num_tiles
        if matrix_type == CUPY_INT:
            self._sort_int_matrix[block_per_grid, thread_per_block](
                unsorted_matrix,
                self._sorted_matrix_mapping_index,
                sorted_matrix
            )
        elif matrix_type == CUPY_FLOAT:
            self._sort_float_matrix[block_per_grid, thread_per_block](
                unsorted_matrix,
                self._sorted_matrix_mapping_index,
                sorted_matrix
            )
        return sorted_matrix

    @staticmethod
    def _sort_matrix_kernel(unsorted_matrix, sorted_matrix_mapping_index, sorted_matrix):
        idx = cuda.grid(1)
        unsorted_index = sorted_matrix_mapping_index[idx]
        if unsorted_index == -1:
            return
        for i in range(unsorted_matrix.shape[1]):
            sorted_matrix[i, idx] = unsorted_matrix[unsorted_index, i]

    def unsort_matrix(self, sorted_matrix: cp.ndarray) -> cp.ndarray:
        matrix_type = sorted_matrix.dtype
        unsorted_matrix = cp.zeros(
            (self._num_particles, sorted_matrix.shape[0]),
            dtype=matrix_type
        )
        thread_per_block = 32
        block_per_grid = self._num_tiles
        if matrix_type == CUPY_INT:
            self._unsort_int_matrix[block_per_grid, thread_per_block](
                sorted_matrix,
                self._sorted_matrix_mapping_index,
                unsorted_matrix
            )
        elif matrix_type == CUPY_FLOAT:
            self._unsort_float_matrix[block_per_grid, thread_per_block](
                sorted_matrix,
                self._sorted_matrix_mapping_index,
                unsorted_matrix
            )
        return unsorted_matrix

    @staticmethod
    def _unsort_matrix_kernel(sorted_matrix, sorted_matrix_mapping_index, unsorted_matrix):
        idx = cuda.grid(1)
        unsorted_index = sorted_matrix_mapping_index[idx]
        if unsorted_index == -1:
            return
        for i in range(unsorted_matrix.shape[1]):
            unsorted_matrix[unsorted_index, i] = sorted_matrix[i, idx]

    def generate_mask_map(self, particle_infomation: cp.ndarray) -> cp.ndarray:
        mask_map = cp.zeros((
            self._num_tiles * NUM_PARTICLES_PER_TILE,
            self._tile_neighbors.shape[1] * NUM_PARTICLES_PER_TILE
        ), CUPY_BIT)
        thread_per_block = (32, 1)
        block_per_grid_x = self._num_tiles
        block_per_grid_y = self._tile_neighbors.shape[1]
        block_per_grid = (block_per_grid_x, block_per_grid_y)
        self._generate_mask_map[block_per_grid, thread_per_block](
            particle_infomation,
            self._tile_neighbors,
            self._sorted_matrix_mapping_index,
            mask_map
        )
        return mask_map

    @staticmethod
    def _generate_mask_map_kernel(particle_infomation, tile_neighbors, sorted_matrix_mapping_index, mask_map):
        tile_id1 = cuda.blockIdx.x
        tile_id2 = tile_neighbors[tile_id1, cuda.blockIdx.y]
        if tile_id2 == -1:
            return
        local_thread_x = cuda.threadIdx.x
        global_thread_x = tile_id1 * cuda.blockDim.x + local_thread_x
        shared_particle_index = cuda.shared.array(shape=(NUM_PARTICLES_PER_TILE), dtype=NUMBA_INT)
        particle_start_index2 = tile_id2 * NUM_PARTICLES_PER_TILE
        shared_particle_index[local_thread_x] = sorted_matrix_mapping_index[particle_start_index2+local_thread_x]
        cuda.syncthreads()

        particle1 = sorted_matrix_mapping_index[global_thread_x]
        start_index = cuda.blockIdx.y * NUM_PARTICLES_PER_TILE
        for particle_index in range(NUM_PARTICLES_PER_TILE):
            for information_index in range(particle_infomation.shape[1]):
                information = particle_infomation[particle1, information_index]
                if information == -1:
                    break
                if information == shared_particle_index[particle_index]:
                    mask_map[global_thread_x, start_index + particle_index] = 1
                    break

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
    def sorted_matrix_mapping_index(self) -> cp.ndarray:
        return self._sorted_matrix_mapping_index

    @property
    def num_cells_vec(self):
        return self._num_cells_vec

    @property
    def device_num_cells_vec(self):
        return self._device_num_cells_vec

if __name__ == '__main__':
    import time
    import mdpy as md
    from cupy.cuda.nvtx import RangePush, RangePop
    pdb = md.io.PDBParser('/home/zhenyuwei/nutstore/ZhenyuWei/Note_Research/mdpy/mdpy/benchmark/str/medium.pdb')
    psf = md.io.PSFParser('/home/zhenyuwei/nutstore/ZhenyuWei/Note_Research/mdpy/mdpy/benchmark/str/medium.psf')
    positions = cp.array(pdb.positions, CUPY_FLOAT)
    positive_positions = positions + cp.array(np.diagonal(pdb.pbc_matrix)) / 2

    tile_list = TileList(pdb.pbc_matrix)
    tile_list.set_cutoff_radius(8)

    tile_list.update(positions)

    print(cp.count_nonzero(tile_list.tile_list+1))
    if not True: # sort_matrix
        for tile in range(32):
            print('# Tile %d, cell_index: %s' %(tile, tile_list.tile_cell_index[tile, :]))
            print(tile_list.sorted_matrix_mapping_index[tile*NUM_PARTICLES_PER_TILE:(tile+1)*NUM_PARTICLES_PER_TILE])
            print(tile_list.sort_matrix(positive_positions)[tile*NUM_PARTICLES_PER_TILE:(tile+1)*NUM_PARTICLES_PER_TILE, :])
    if not True:
        sorted_positions = tile_list.sort_matrix(positions)
        unsorted_positions = tile_list.unsort_matrix(sorted_positions)
        print(cp.hstack([positions, unsorted_positions])[:100, :])
    if True:
        excluded_mask_map = tile_list.generate_mask_map(cp.array(psf.topology.excluded_particles, CUPY_INT))
        print(cp.count_nonzero(excluded_mask_map))
        num_excluded_particles = 0
        for particle in psf.topology.particles:
            num_excluded_particles += particle.num_excluded_particles
        print(num_excluded_particles)

    epoch = 30
    RangePush('Update tile')
    ts = time.time()
    for i in range(epoch):
        tile_list.update(positions)
    te = time.time()
    print('Run update for %s s' %((te-ts)/epoch))
    RangePop()

    RangePush('Sort float matrix')
    ts = time.time()
    for i in range(epoch):
        tile_list.sort_matrix(positions)
    te = time.time()
    print('Run sort float for %s s' %((te-ts)/epoch))
    RangePop()

    int_data = cp.random.randint(0, 100, size=(positions.shape[0], 15), dtype=CUPY_INT)
    RangePush('Sort int matrix')
    ts = time.time()
    for i in range(epoch):
        tile_list.sort_matrix(int_data)
    te = time.time()
    print('Run sort int for %s s' %((te-ts)/epoch))
    RangePop()


    sorted_positions = tile_list.sort_matrix(positions)
    RangePush('Unsort matrix')
    ts = time.time()
    for i in range(epoch):
        tile_list.unsort_matrix(sorted_positions)
    te = time.time()
    print('Run unsort for %s s' %((te-ts)/epoch))
    RangePop()

    excluded_particles = cp.array(psf.topology.excluded_particles, CUPY_INT)
    RangePush('Unsort matrix')
    ts = time.time()
    for i in range(epoch):
        tile_list.generate_mask_map(excluded_particles)
    te = time.time()
    print('Run generate mask for %s s' %((te-ts)/epoch))
    RangePop()
