#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_tile_list.py
created time : 2022/05/05
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import pytest, os
import numpy as np
import cupy as cp
import mdpy as md
from mdpy import SPATIAL_DIM
from mdpy.core import TileList
from mdpy.core.tile_list import NUM_PARTICLES_PER_TILE
from mdpy.environment import *
from mdpy.unit import *
from mdpy.error import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data')

class TestTileList:
    def setup(self):
        self.tile_list = TileList(np.diag([80, 80, 100]))
        self.tile_list.set_cutoff_radius(Quantity(8, angstrom))

    def teardown(self):
        self.tile_list = None

    def test_attributes(self):
        assert self.tile_list.cutoff_radius == 8
        assert self.tile_list.pbc_matrix[0, 0] == 80

    def test_exceptions(self):
        with pytest.raises(UnitDimensionDismatchedError):
            self.tile_list.set_cutoff_radius(Quantity(1, second))

        with pytest.raises(TileListPoorDefinedError):
            self.tile_list.set_cutoff_radius(0)

        with pytest.raises(TileListPoorDefinedError):
            self.tile_list.set_cutoff_radius(56)

    def test_update(self):
        num_particles = 300000
        positions = ((np.random.rand(num_particles, SPATIAL_DIM) - 0.5) * self.tile_list._pbc_diag * 0.5).astype(np.float32)
        positive_positions = positions + self.tile_list._half_pbc_diag
        self.tile_list.update(cp.array(positions, CUPY_FLOAT))
        tile_list = self.tile_list.tile_list.get()

        # All particles has been assigned to a tile once
        assert np.count_nonzero(tile_list != -1) == num_particles

        # All particles has been assigned to the right tile
        for tile_index in range(0, self.tile_list.num_tiles, 100):
            cell_index = self.tile_list._tile_cell_index[tile_index, :]
            position_lower = cell_index * self.tile_list._cell_width
            position_upper = (cell_index + 1) * self.tile_list._cell_width
            for particle_tile_index in range(NUM_PARTICLES_PER_TILE):
                particle_index = tile_list[tile_index, particle_tile_index]
                if particle_index == -1:
                    break
                particle_positions = positive_positions[particle_index, :]
                for i in range(SPATIAL_DIM):
                    assert particle_positions[i] <= position_upper[i]
                    assert particle_positions[i] >= position_lower[i]

        # All neighbors has been assigned
        for tile_index in range(0, self.tile_list.num_tiles, 500):
            neighbors = self.tile_list.tile_neighbors[tile_index, :]
            central_cell_index = self.tile_list._tile_cell_index[tile_index, :]
            for neighhor in neighbors[::10]:
                if neighhor == -1:
                    break
                neighbor_cell_index = self.tile_list._tile_cell_index[neighhor, :]
                for i in range(SPATIAL_DIM):
                    diff = neighbor_cell_index[i] - central_cell_index[i]
                    if diff > self.tile_list.num_cells_vec[i] / 2:
                        diff -= self.tile_list.num_cells_vec[i]
                    elif diff < - self.tile_list.num_cells_vec[i] / 2:
                        diff += self.tile_list.num_cells_vec[i]
                    assert np.abs(diff) <= 1

    def test_exclusion_map(self):
        pdb = md.io.PDBParser(os.path.join(data_dir, 'solvated_6PO6.pdb'))
        topology = md.io.PSFParser(os.path.join(data_dir, 'solvated_6PO6.psf')).topology
        positions = cp.array(pdb.positions, CUPY_FLOAT)
        device_excluded_particles = cp.array(topology.excluded_particles, CUPY_INT)

        tile_list = md.core.TileList(pdb.pbc_matrix)
        tile_list.set_cutoff_radius(Quantity(8, angstrom))
        tile_list.update(positions)
        exclusion_mask = tile_list.generate_exclusion_mask_map(device_excluded_particles)

        for tile_index in np.random.randint(0, tile_list.num_tiles, 5):
            print(tile_index, tile_list.tile_list[tile_index, :])
            tile_neighbors = tile_list.tile_neighbors[tile_index]
            particle_neighbors = np.zeros([tile_neighbors.shape[0]*NUM_PARTICLES_PER_TILE], NUMPY_INT) - 1
            for i, j in enumerate(tile_neighbors):
                if j != -1:
                    particle_neighbors[i*NUM_PARTICLES_PER_TILE:(i+1)*NUM_PARTICLES_PER_TILE] = tile_list.tile_list[j, :].get()
            particle_start_index = tile_index * NUM_PARTICLES_PER_TILE
            for i, j in enumerate(tile_list.tile_list[tile_index]):
                sorted_index = particle_start_index + i
                if j == -1:
                    is_excluded = np.ones_like(particle_neighbors, NUMPY_BIT)
                else:
                    is_excluded = np.zeros_like(particle_neighbors, NUMPY_BIT)
                    excluded_particles = list(device_excluded_particles[j, :].get())
                    excluded_particles = [x for x in excluded_particles if x != -1] + [j]
                    for k, l in enumerate(particle_neighbors):
                        if l == -1:
                            is_excluded[k] = 1
                        elif l in excluded_particles:
                            is_excluded[k] = 1
                exclusion_mask_res = exclusion_mask[:, sorted_index].get()
                target_is_excluded = np.zeros_like(is_excluded)
                index = 0
                for i in exclusion_mask_res:
                    for j in range(NUM_PARTICLES_PER_TILE):
                        target_is_excluded[index] = i >> j &0b1
                        index += 1
                assert np.all(target_is_excluded == is_excluded)

test = TestTileList()
test.test_exclusion_map()