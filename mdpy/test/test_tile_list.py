#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_tile_list.py
created time : 2022/05/05
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

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
data_dir = os.path.join(cur_dir, "data/simulation")
out_dir = os.path.join(cur_dir, "out/tile_list")
pdb_file = os.path.join(data_dir, "solvated_6PO6.pdb")
psf_file = os.path.join(data_dir, "solvated_6PO6.psf")


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

    def view_encode_particles(self):
        pdb = md.io.PDBParser(pdb_file)
        positions = cp.array(pdb.positions, CUPY_FLOAT)
        tile_list = md.core.TileList(pdb.pbc_matrix)
        tile_list.set_cutoff_radius(Quantity(8, angstrom))
        tile_list.update(positions)

        print(tile_list.num_tiles)
        with open(os.path.join(out_dir, "sorted_particle_positions.xyz"), "w") as f:
            print("%d" % tile_list._num_particles, file=f)
            print("test morton code", file=f)
            for i in range(tile_list.num_tiles):
                for j in tile_list.tile_list[i, :]:
                    if j == -1:
                        break
                    print(
                        "H%d %.2f %.2f %.2f"
                        % (i, positions[j, 0], positions[j, 1], positions[j, 2]),
                        file=f,
                    )

    def view_neighbors(self):
        pdb = md.io.PDBParser(pdb_file)
        positions = cp.array(pdb.positions, CUPY_FLOAT)
        tile_list = md.core.TileList(pdb.pbc_matrix)
        tile_list.set_cutoff_radius(Quantity(8, angstrom))
        tile_list.update(positions)

        tile_index = 1200
        tile_neighbors = tile_list.tile_neighbors.get()[tile_index, :]
        tcl_template = 'mol selection "type H%d"\n'
        tcl_template += "mol representation vdw\n"
        tcl_template += "mol color colorid %d\n"
        tcl_template += "mol addrep top"
        with open(os.path.join(out_dir, "view_neighbors.tcl"), "w") as f:
            print("mol delete all\nmol load xyz sorted_particle_positions.xyz", file=f)
            print(tcl_template % (tile_index, 1), file=f)
            tcl_template = tcl_template.replace("vdw", "surf")
            for index, neighbor in enumerate(tile_neighbors):
                if neighbor != -1:
                    print(tcl_template % (neighbor, index % 32), file=f)

    def test_find_neighbors(self):
        pdb = md.io.PDBParser(pdb_file)
        positions = cp.array(pdb.positions, CUPY_FLOAT)
        tile_list = md.core.TileList(pdb.pbc_matrix)
        tile_list.set_cutoff_radius(Quantity(8, angstrom))
        tile_list.update(positions)
        # print(cp.count_nonzero(tile_list.tile_neighbors != -1))
        # print(tile_list.tile_neighbors.shape)
        for particle_id in np.random.randint(0, pdb.num_particles, 5):
            for tile_id in range(tile_list.num_tiles):
                if cp.any(tile_list.tile_list[tile_id, :] == particle_id):
                    break
            neighbor_particles = []
            for tile in tile_list.tile_neighbors[tile_id]:
                if tile != -1:
                    neighbor_particles.append(tile_list.tile_list[tile])
            neighbor_particles = cp.hstack(neighbor_particles)
            neighbor_particles = list(
                neighbor_particles[neighbor_particles != -1].flatten().get()
            )
            # print(neighbor_particles)
            diff = (positions - positions[particle_id, :]) / tile_list._device_pbc_diag
            diff = (cp.round(diff) - diff) * tile_list._device_pbc_diag
            r = cp.sqrt((diff**2).sum(1))
            # print(len(neighbor_particles))
            # print(cp.count_nonzero(r < 8))
            for neighbor in cp.argwhere(r < 8).flatten()[::5]:
                assert neighbor in neighbor_particles

    def test_exclusion_map(self):
        pdb = md.io.PDBParser(pdb_file)
        topology = md.io.PSFParser(psf_file).topology
        positions = cp.array(pdb.positions, CUPY_FLOAT)
        device_excluded_particles = cp.array(topology.excluded_particles, CUPY_INT)

        tile_list = md.core.TileList(pdb.pbc_matrix)
        tile_list.set_cutoff_radius(Quantity(8, angstrom))
        tile_list.update(positions)
        exclusion_mask = tile_list.generate_exclusion_mask_map(
            device_excluded_particles
        )

        for tile_index in np.random.randint(0, tile_list.num_tiles, 5):
            print(tile_index, tile_list.tile_list[tile_index, :])
            tile_neighbors = tile_list.tile_neighbors[tile_index]
            particle_neighbors = (
                np.zeros([tile_neighbors.shape[0] * NUM_PARTICLES_PER_TILE], NUMPY_INT)
                - 1
            )
            for i, j in enumerate(tile_neighbors):
                if j != -1:
                    particle_neighbors[
                        i * NUM_PARTICLES_PER_TILE : (i + 1) * NUM_PARTICLES_PER_TILE
                    ] = tile_list.tile_list[j, :].get()
            particle_start_index = tile_index * NUM_PARTICLES_PER_TILE
            for i, j in enumerate(tile_list.tile_list[tile_index]):
                sorted_index = particle_start_index + i
                if j == -1:
                    is_excluded = np.ones_like(particle_neighbors, NUMPY_BIT)
                else:
                    is_excluded = np.zeros_like(particle_neighbors, NUMPY_BIT)
                    excluded_particles = list(device_excluded_particles[j, :].get())
                    excluded_particles = [x for x in excluded_particles if x != -1] + [
                        j
                    ]
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
                        target_is_excluded[index] = i >> j & 0b1
                        index += 1
                target_is_excluded = list(target_is_excluded)
                for i in is_excluded[::5]:
                    assert i in target_is_excluded


if __name__ == "__main__":
    test = TestTileList()
    test.view_encode_particles()
    test.view_neighbors()
