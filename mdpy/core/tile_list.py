#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : tile_list.py
created time : 2022/05/04
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import os
import math
import numpy as np
import cupy as cp
import numba.cuda as cuda
from mdpy import SPATIAL_DIM
from mdpy.core import MAX_NUM_EXCLUDED_PARTICLES, NUM_NEIGHBOR_CELLS, NUM_PARTICLES_PER_TILE
from mdpy.environment import *
from mdpy.utils import *
from mdpy.unit import *
from mdpy.error import *

SKIN_WIDTH = Quantity(1.5, angstrom)
THREAD_PER_BLOCK = 32
NUM_TILES_PER_THREAD = 16

cur_dir = os.path.dirname(os.path.abspath(__file__))
lookup_table_dir = os.path.join(cur_dir, '../../data/space_filling_curve_lookup_table/')
class TileList:
    def __init__(self, pbc_matrix: np.ndarray, skin_width=SKIN_WIDTH, num_bits=7) -> None:
        pbc_matrix = check_quantity_value(pbc_matrix, default_length_unit)
        self._pbc_matrix = check_pbc_matrix(pbc_matrix)
        self._pbc_diag = np.ascontiguousarray(self._pbc_matrix.diagonal(), dtype=NUMPY_FLOAT)
        self._half_pbc_diag = self._pbc_diag / 2
        self._skin_width = check_quantity_value(skin_width, default_length_unit)
        self._num_bits = num_bits
        # Attribute
        self.set_cutoff_radius(Quantity(1, angstrom))
        self._num_cells_vec = np.ceil(self._pbc_diag / self._cell_width).astype(NUMPY_INT)
        self._tile_list = None
        self._tile_cell_index = None
        self._particle_tile_information = None
        self._cell_tile_information = None
        self._num_tiles = 0
        # lookup table
        lookup_file = os.path.join(lookup_table_dir, 'hilbert_%d_bits.npy' %num_bits)
        self._code_range = 2**self._num_bits
        self._code_normalize_factor = self._code_range**3
        self._lookup_table = np.load(lookup_file).astype(NUMPY_INT)
        self._device_lookup_table = cp.array(self._lookup_table, CUPY_INT)
        # Deivice attribute
        self._device_pbc_matrix = cp.array(self._pbc_matrix, CUPY_FLOAT)
        self._device_pbc_diag = cp.array(self._pbc_diag, CUPY_FLOAT)
        self._device_half_pbc_diag = cp.array(self._half_pbc_diag, CUPY_FLOAT)
        # Kernel
        self._construct_tile_list = cuda.jit(nb.void(
            NUMBA_FLOAT[:, ::1], # positions
            NUMBA_INT[::1], # sorted_particle_index
            NUMBA_INT[:, ::1], # cell_particle_information
            NUMBA_INT[:, ::1], # cell_tile_information
            NUMBA_INT[::1], # num_cells_vec
            NUMBA_INT[::1], # sorted_matrix_index
            NUMBA_FLOAT[:, ::1], # tile_box
            NUMBA_INT[:, ::1], # tile_list
            NUMBA_INT[:, ::1] # tile_cell_index
        ))(self._construct_tile_list_kernel)
        self._find_tile_neighbors = cuda.jit(nb.void(
            NUMBA_INT[::1], # num_cells_vec
            NUMBA_FLOAT[::1], # cutoff_radius
            NUMBA_FLOAT[:, ::1], # pbc_matrix
            NUMBA_INT[:, ::1], # tile_cell_index
            NUMBA_INT[:, ::1], # cell_tile_information
            NUMBA_FLOAT[:, ::1], # tile_box
            NUMBA_INT[::1], # tile_num_neighbors
            NUMBA_INT[:, ::1] # tile_neighbors
        ))(self._find_tile_neighbors_kernel)
        # sort
        self._sort_int_matrix = cuda.jit(nb.void(
            NUMBA_INT[:, ::1], # unsorted_matrix
            NUMBA_INT[::1], # matrix_id_mapping_index
            NUMBA_INT[:, ::1] # sorted_matrix
        ))(self._sort_matrix_kernel)
        self._sort_float_matrix = cuda.jit(nb.void(
            NUMBA_FLOAT[:, ::1], # unsorted_matrix
            NUMBA_INT[::1], # matrix_id_mapping_index
            NUMBA_FLOAT[:, ::1] # sorted_matrix
        ))(self._sort_matrix_kernel)
        # unsort
        self._unsort_int_matrix = cuda.jit(nb.void(
            NUMBA_INT[:, ::1], # sorted_matrix
            NUMBA_INT[::1], # matrix_id_mapping_index
            NUMBA_INT[:, ::1] # unsorted_matrix
        ))(self._unsort_matrix_kernel)
        self._unsort_float_matrix = cuda.jit(nb.void(
            NUMBA_FLOAT[:, ::1], # sorted_matrix
            NUMBA_INT[::1], # matrix_id_mapping_index
            NUMBA_FLOAT[:, ::1] # unsorted_matrix
        ))(self._unsort_matrix_kernel)
        # exclusion mask
        self._generate_exclusion_mask_map = cuda.jit(nb.void(
            NUMBA_FLOAT[::1], # cutoff_radius
            NUMBA_FLOAT[:, ::1], # pbc_matrix
            NUMBA_FLOAT[:, ::1], # sorted_positions
            NUMBA_INT[:, ::1], # excluded_particles
            NUMBA_INT[:, ::1], # tile_neighbors
            NUMBA_INT[::1], # padded_sorted_particle_index
            NUMBA_BIT[:, ::1] # mask_map
        ))(self._generate_exclusion_mask_map_kernel)

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
        self._num_cells_vec = np.floor(self._pbc_diag / self._cell_width).astype(NUMPY_INT)
        self._num_cells_vec[self._num_cells_vec < 3] = 3
        # Device attributes
        self._device_cutoff_radius = cp.array([self._cutoff_radius], CUPY_FLOAT)
        self._device_cell_width = cp.array([self._cell_width], CUPY_FLOAT)
        self._device_num_cells_vec = cp.array(self._num_cells_vec, CUPY_INT)

    def _encode_particles(self, positive_positions: cp.ndarray):
        # Prevent particle at boundary
        scaled_positions = positive_positions / (self._device_pbc_diag + 0.01) * self._device_num_cells_vec
        scaled_int_part = cp.floor(scaled_positions)
        scaled_fraction_part = scaled_positions - scaled_int_part
        scaled_int_part = scaled_int_part.astype(CUPY_INT)
        # cell index
        cell_index = (
            scaled_int_part[:, 2] + scaled_int_part[:, 1] * self._num_cells_vec[2] +
            scaled_int_part[:, 0] * self._num_cells_vec[2] * self._num_cells_vec[1]
        )
        # cell particle information
        num_particles_each_cell = cp.bincount(cell_index, minlength=np.prod(self._num_cells_vec))
        particle_start_index = cp.zeros_like(num_particles_each_cell, dtype=CUPY_INT)
        particle_start_index[1:] = cp.cumsum(num_particles_each_cell)[:-1]
        self._cell_particle_information = cp.stack([
            particle_start_index,
            num_particles_each_cell
        ], axis=1).astype(CUPY_INT)
        # cell tile information
        num_tiles_each_ceil = cp.ceil(num_particles_each_cell / NUM_PARTICLES_PER_TILE).astype(CUPY_INT)
        self._num_tiles = int(num_tiles_each_ceil.sum())
        self._max_num_tiles_per_cell = int(num_tiles_each_ceil.max())
        tile_start_index = cp.zeros_like(num_tiles_each_ceil, dtype=CUPY_INT)
        tile_start_index[1:] = cp.cumsum(num_tiles_each_ceil)[:-1]
        self._cell_tile_information = cp.stack([
            tile_start_index,
            num_tiles_each_ceil
        ], axis=1).astype(CUPY_INT)
        # Subcell space filling index
        subcell_index = cp.round(scaled_fraction_part * self._code_range).astype(CUPY_INT)
        subcell_index = self._device_lookup_table[
            subcell_index[:, 0],
            subcell_index[:, 1],
            subcell_index[:, 2],
        ] / self._code_normalize_factor
        # particle index
        particle_index = cell_index + subcell_index
        self._sorted_particle_index = cp.argsort(particle_index).astype(CUPY_INT)

    @staticmethod
    def _construct_tile_list_kernel(
        positions,
        sorted_particle_index,
        cell_particle_information,
        cell_tile_information,
        num_cells_vec,
        matrix_id_mapping_index,
        tile_box,
        tile_list,
        tile_cell_index,
    ):
        tile_index = cuda.grid(1)
        if tile_index >= tile_list.shape[0]:
            return
        # Binary search of tile's cell index
        low, high = 0, cell_tile_information.shape[0] - 1
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
        local_box = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        box_max = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        box_min = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        tile_particle_index = 0
        sorted_matrix_index = tile_index * NUM_PARTICLES_PER_TILE # The index of sorted_particle
        # first particle
        particle_index = sorted_particle_index[tile_particle_start_index]
        tile_list[tile_index, tile_particle_index] = particle_index
        matrix_id_mapping_index[sorted_matrix_index] = particle_index
        for i in range(3):
            data = positions[particle_index, i]
            local_box[i] = data
            box_max[i] = data
            box_min[i] = data
        tile_particle_index += 1
        sorted_matrix_index += 1
        for index in range(tile_particle_start_index+1, tile_particle_end_index):
            particle_index = sorted_particle_index[index]
            tile_list[tile_index, tile_particle_index] = particle_index
            matrix_id_mapping_index[sorted_matrix_index] = particle_index
            for i in range(SPATIAL_DIM):
                data = positions[particle_index, i]
                local_box[i] += data
                if data > box_max[i]:
                    box_max[i] = data
                if data < box_min[i]:
                    box_min[i] = data
            tile_particle_index += 1
            sorted_matrix_index += 1
        diag = NUMBA_FLOAT(0)
        for i in range(SPATIAL_DIM):
            tile_box[tile_index, i] = local_box[i] / tile_particle_index
            tile_box[tile_index, i+SPATIAL_DIM] = (box_max[i] - box_min[i]) * NUMBA_FLOAT(0.5)
            diag += tile_box[tile_index, i+SPATIAL_DIM]**2
        tile_box[tile_index, 6] = math.sqrt(diag)
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
    def _find_tile_neighbors_kernel(
        num_cells_vec,
        cutoff_radius,
        pbc_matrix,
        tile_cell_index,
        cell_tile_information,
        tile_box,
        tile_num_neighbors,
        tile_neighbors
    ):
        tile_id = cuda.grid(1)
        if tile_id >= tile_neighbors.shape[0]:
            return
        central_cell_index = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_INT)
        num_cells_x = num_cells_vec[0]
        num_cells_y = num_cells_vec[1]
        num_cells_z = num_cells_vec[2]
        num_cells_yz = NUMBA_INT(num_cells_y * num_cells_z)
        central_cell_index[0] = tile_cell_index[tile_id, 0]
        central_cell_index[1] = tile_cell_index[tile_id, 1]
        central_cell_index[2] = tile_cell_index[tile_id, 2]

        # shared data
        local_thread_x = cuda.threadIdx.x
        shared_pbc_matrix = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        shared_half_pbc_matrix = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        if local_thread_x <= 2:
            shared_pbc_matrix[local_thread_x] = pbc_matrix[local_thread_x, local_thread_x]
            shared_half_pbc_matrix[local_thread_x] = shared_pbc_matrix[local_thread_x] * NUMBA_FLOAT(0.5)

        neighbor_cell_index = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_INT)
        local_box = cuda.local.array(shape=(7), dtype=NUMBA_FLOAT)
        vec = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        cutoff_radius = cutoff_radius[0]
        for i in range(7):
            local_box[i] = tile_box[tile_id, i]
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
                        neighbor_tile_id = cur_tile_index
                        is_neighbor = True
                        r = NUMBA_FLOAT(0)
                        for i in range(SPATIAL_DIM):
                            vec[i] = abs(tile_box[neighbor_tile_id, i] - local_box[i])
                            if vec[i] > shared_half_pbc_matrix[i]:
                                vec[i] = shared_pbc_matrix[i] - vec[i]
                            if vec[i] - local_box[i+3] - tile_box[neighbor_tile_id, i+3] >= cutoff_radius:
                                is_neighbor = False
                                break
                            r += vec[i]**2
                        # is_neighbor = True
                        if is_neighbor:
                            if math.sqrt(r) - local_box[6] - tile_box[neighbor_tile_id, 6] < cutoff_radius:
                                tile_neighbors[tile_id, cur_neighbor_tile_index] = neighbor_tile_id
                                cur_neighbor_tile_index += 1
                        cur_tile_index += 1
        tile_num_neighbors[tile_id] = cur_neighbor_tile_index

    def update(self, positions: cp.ndarray) -> None:
        self._num_particles = positions.shape[0]
        self._positions = positions
        # Encode particles
        positive_positions = self._positions + self._device_half_pbc_diag
        self._encode_particles(positive_positions)
        # Create tile list
        block_per_grid = int(np.ceil(self._num_tiles / THREAD_PER_BLOCK))
        self._tile_box = cp.zeros((self._num_tiles, 7), CUPY_FLOAT)
        self._tile_list = cp.zeros((self._num_tiles, NUM_PARTICLES_PER_TILE), dtype=CUPY_INT) - 1
        self._tile_cell_index = cp.zeros((self._num_tiles, SPATIAL_DIM), dtype=CUPY_INT)
        self._matrix_id_mapping_index = cp.zeros(self._num_tiles*NUM_PARTICLES_PER_TILE, dtype=CUPY_INT) - 1
        self._construct_tile_list[block_per_grid, THREAD_PER_BLOCK](
            positions,
            self._sorted_particle_index,
            self._cell_particle_information,
            self._cell_tile_information,
            self._device_num_cells_vec,
            self._matrix_id_mapping_index,
            self._tile_box,
            self._tile_list,
            self._tile_cell_index
        )
        # Find tile neighbors
        thread_per_block = 32
        block_per_grid = int(np.ceil(self._num_tiles / thread_per_block))
        self._tile_neighbors = cp.zeros((self._num_tiles, int(self._max_num_tiles_per_cell)*NUM_NEIGHBOR_CELLS), CUPY_INT) - 1
        self._tile_num_neighbors = cp.zeros((self._num_tiles), CUPY_INT)
        self._find_tile_neighbors[block_per_grid, thread_per_block](
            self._device_num_cells_vec,
            self._device_cutoff_radius,
            self._device_pbc_matrix,
            self._tile_cell_index,
            self._cell_tile_information,
            self._tile_box,
            self._tile_num_neighbors,
            self._tile_neighbors
        )
        self._tile_neighbors = cp.array(self._tile_neighbors[:, :int(cp.max(self._tile_num_neighbors))], CUPY_INT)

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
                self._matrix_id_mapping_index,
                sorted_matrix
            )
        elif matrix_type == CUPY_FLOAT:
            self._sort_float_matrix[block_per_grid, thread_per_block](
                unsorted_matrix,
                self._matrix_id_mapping_index,
                sorted_matrix
            )
        return sorted_matrix

    @staticmethod
    def _sort_matrix_kernel(unsorted_matrix, matrix_id_mapping_index, sorted_matrix):
        idx = cuda.grid(1)
        unsorted_index = matrix_id_mapping_index[idx]
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
                self._matrix_id_mapping_index,
                unsorted_matrix
            )
        elif matrix_type == CUPY_FLOAT:
            self._unsort_float_matrix[block_per_grid, thread_per_block](
                sorted_matrix,
                self._matrix_id_mapping_index,
                unsorted_matrix
            )
        return unsorted_matrix

    @staticmethod
    def _unsort_matrix_kernel(sorted_matrix, matrix_id_mapping_index, unsorted_matrix):
        idx = cuda.grid(1)
        unsorted_index = matrix_id_mapping_index[idx]
        if unsorted_index == -1:
            return
        for i in range(unsorted_matrix.shape[1]):
            unsorted_matrix[unsorted_index, i] = sorted_matrix[i, idx]

    def generate_exclusion_mask_map(self, device_excluded_particles: cp.ndarray) -> cp.ndarray:
        sorted_positions = self.sort_matrix(self._positions)
        mask_map = cp.zeros((
            self._tile_neighbors.shape[1], self._num_tiles * NUM_PARTICLES_PER_TILE
        ), CUPY_BIT)
        thread_per_block = (NUM_PARTICLES_PER_TILE, 1)
        block_per_grid_x = self._num_tiles
        block_per_grid_y = int(np.ceil(self._tile_neighbors.shape[1] / NUM_TILES_PER_THREAD))
        block_per_grid = (block_per_grid_x, block_per_grid_y)
        self._generate_exclusion_mask_map[block_per_grid, thread_per_block](
            self._device_cutoff_radius,
            self._device_pbc_matrix,
            sorted_positions,
            device_excluded_particles,
            self._tile_neighbors,
            self._matrix_id_mapping_index,
            mask_map
        )
        return mask_map

    @staticmethod
    def _generate_exclusion_mask_map_kernel(
        cutoff_radius,
        pbc_matrix,
        sorted_positions,
        excluded_particles,
        tile_neighbors,
        padded_sorted_particle_index,
        mask_map
    ):
        tile_id1 = cuda.blockIdx.x
        tile_id2_start = cuda.blockIdx.y * NUM_TILES_PER_THREAD
        tile_id2_end = tile_id2_start + NUM_TILES_PER_THREAD
        if tile_id2_end >= tile_neighbors.shape[1]:
            tile_id2_end = tile_neighbors.shape[1]
        # Particle index information
        local_thread_x = cuda.threadIdx.x
        global_thread_x = local_thread_x + cuda.blockIdx.x * cuda.blockDim.x
        # shared data
        shared_pbc_matrix = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        shared_half_pbc_matrix = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        if local_thread_x <= 2:
            shared_pbc_matrix[local_thread_x] = pbc_matrix[local_thread_x, local_thread_x]
            shared_half_pbc_matrix[local_thread_x] = shared_pbc_matrix[local_thread_x] * NUMBA_FLOAT(0.5)
        tile1_positions = cuda.shared.array(shape=(SPATIAL_DIM, NUM_PARTICLES_PER_TILE), dtype=NUMBA_FLOAT)
        tile2_positions = cuda.shared.array(shape=(SPATIAL_DIM, NUM_PARTICLES_PER_TILE), dtype=NUMBA_FLOAT)
        tile2_particle_index = cuda.shared.array(shape=(NUM_PARTICLES_PER_TILE), dtype=NUMBA_INT)
        tile1_index = tile_id1 * NUM_PARTICLES_PER_TILE + local_thread_x
        cuda.syncthreads()
        # Read data
        for i in range(SPATIAL_DIM):
            tile1_positions[i, local_thread_x] = sorted_positions[i, tile1_index]
        # Local data
        local_forces = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        vec = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        cutoff_radius = cutoff_radius[0]
        for i in range(SPATIAL_DIM):
            local_forces[i] = 0
        for tile_index in range(tile_id2_start, tile_id2_end):
            tile_id2 = tile_neighbors[tile_id1, tile_index]
            if tile_id2 == -1:
                mask_map[tile_index, global_thread_x] = NUMBA_BIT(4294967295)
                continue
            tile2_index = tile_id2 * NUM_PARTICLES_PER_TILE + local_thread_x
            for i in range(SPATIAL_DIM):
                tile2_positions[i, local_thread_x] = sorted_positions[i, tile2_index]
            tile2_particle_index[local_thread_x] = padded_sorted_particle_index[tile2_index]
            cuda.syncthreads()
            exclusion_flag = NUMBA_BIT(0)
            particle1 = padded_sorted_particle_index[global_thread_x]
            if particle1 == -1:
                mask_map[tile_index, global_thread_x] = 4294967295 # 2**32 - 1 all 1
                continue
            for particle_index in range(NUM_PARTICLES_PER_TILE):
                particle2 = tile2_particle_index[particle_index]
                if particle2 == -1:
                    exclusion_flag = exclusion_flag ^ (1 << particle_index)
                else:
                    for information_index in range(MAX_NUM_EXCLUDED_PARTICLES):
                        particle2 = excluded_particles[particle1, information_index]
                        if particle2 == -1:
                            break
                        elif particle2 == tile2_particle_index[particle_index]:
                            exclusion_flag = exclusion_flag ^ (1 << particle_index)
                            break
                    r = NUMBA_FLOAT(0)
                    for i in range(SPATIAL_DIM):
                        vec[i] = tile2_positions[i, particle_index] - tile1_positions[i, local_thread_x]
                        if vec[i] < - shared_half_pbc_matrix[i]:
                            vec[i] += shared_pbc_matrix[i]
                        elif vec[i] > shared_half_pbc_matrix[i]:
                            vec[i] -= shared_pbc_matrix[i]
                        r += vec[i]**2
                    r = math.sqrt(r)
                    if r > cutoff_radius:
                        exclusion_flag = exclusion_flag ^ (1 << particle_index)
            mask_map[tile_index, global_thread_x] = exclusion_flag

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
    def sorted_particle_index(self) -> cp.ndarray:
        return self._sorted_particle_index

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

    # epoch = 30
    # RangePush('Update tile')
    # ts = time.time()
    # for i in range(epoch):
    #     tile_list.update(positions)
    # te = time.time()
    # print('Run update for %s s' %((te-ts)/epoch))
    # RangePop()

    # RangePush('Sort float matrix')
    # ts = time.time()
    # for i in range(epoch):
    #     tile_list.sort_matrix(positions)
    # te = time.time()
    # print('Run sort float for %s s' %((te-ts)/epoch))
    # RangePop()

    # int_data = cp.random.randint(0, 100, size=(positions.shape[0], 15), dtype=CUPY_INT)
    # RangePush('Sort int matrix')
    # ts = time.time()
    # for i in range(epoch):
    #     tile_list.sort_matrix(int_data)
    # te = time.time()
    # print('Run sort int for %s s' %((te-ts)/epoch))
    # RangePop()


    # sorted_positions = tile_list.sort_matrix(positions)
    # RangePush('Unsort matrix')
    # ts = time.time()
    # for i in range(epoch):
    #     tile_list.unsort_matrix(sorted_positions)
    # te = time.time()
    # print('Run unsort for %s s' %((te-ts)/epoch))
    # RangePop()

    # excluded_particles = cp.array(psf.topology.excluded_particles, CUPY_INT)
    # RangePush('Unsort matrix')
    # ts = time.time()
    # for i in range(epoch):
    #     tile_list.generate_exclusion_mask_map(excluded_particles)
    # te = time.time()
    # print('Run generate mask for %s s' %((te-ts)/epoch))
    # RangePop()
