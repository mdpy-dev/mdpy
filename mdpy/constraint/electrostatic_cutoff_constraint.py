#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : electrostatic_cutoff_constraint.py
created time : 2021/10/13
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import math
import numpy as np
import numba as nb
import cupy as cp
from numba import cuda
from mdpy import SPATIAL_DIM
from mdpy.environment import *
from mdpy.core import Ensemble
from mdpy.core import NUM_PARTICLES_PER_TILE
from mdpy.constraint import Constraint
from mdpy.utils import *
from mdpy.unit import *

TILES_PER_THREAD = 4

class ElectrostaticCutoffConstraint(Constraint):
    def __init__(self, cutoff_radius=Quantity(12, angstrom)) -> None:
        super().__init__()
        # Attributes
        self._cutoff_radius = check_quantity_value(cutoff_radius, default_length_unit)
        self._device_cutoff_radius = cp.array([self._cutoff_radius], CUPY_FLOAT)
        self._k = 4 * np.pi * EPSILON0.value
        self._device_inverse_k = cp.array([1 / self._k], CUPY_FLOAT)
        # Kernel
        self._update_electrostatic_cutoff = cuda.jit(nb.void(
            NUMBA_FLOAT[::1], # inverse_k
            NUMBA_FLOAT[::1], # cutoff_radius
            NUMBA_FLOAT[:, ::1], # pbc_matrix
            NUMBA_FLOAT[:, ::1], # sorted_positions
            NUMBA_FLOAT[:, ::1], # sorted_charges
            NUMBA_BIT[:, ::1], # exclusion_map
            NUMBA_INT[:, ::1], # tile_neighbors
            NUMBA_FLOAT[:, ::1], # sorted_forces
            NUMBA_FLOAT[::1] # potential_energy
        ))(self._update_electrostatic_cutoff_kernel)

    def __repr__(self) -> str:
        return '<mdpy.constraint.ElectrostaticCutoffConstraint object>'

    def __str__(self) -> str:
        return 'Cutoff electrostatic constraint'

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._constraint_id = ensemble.constraints.index(self)

    @staticmethod
    def _update_electrostatic_cutoff_kernel(
        inverse_k,
        cutoff_radius,
        pbc_matrix,
        sorted_positions,
        sorted_charges,
        exclusion_map,
        tile_neighbors,
        sorted_forces, potential_energy
    ):
        # Particle index information
        local_thread_x = cuda.threadIdx.x
        local_thread_y = cuda.threadIdx.y
        tile_id1 = cuda.blockIdx.x * TILES_PER_THREAD + local_thread_y
        if tile_id1 >= tile_neighbors.shape[0]:
            return
        tile1_particle_index = tile_id1 * NUM_PARTICLES_PER_TILE + local_thread_x
        # shared data
        local_pbc_matrix = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        local_half_pbc_matrix = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        for i in range(SPATIAL_DIM):
            local_pbc_matrix[i] = pbc_matrix[i, i]
            local_half_pbc_matrix[i] = local_pbc_matrix[i] * NUMBA_FLOAT(0.5)
        tile1_positions = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        tile2_positions = cuda.shared.array(shape=(SPATIAL_DIM, TILES_PER_THREAD, NUM_PARTICLES_PER_TILE), dtype=NUMBA_FLOAT)
        tile2_charges = cuda.shared.array(shape=(TILES_PER_THREAD, NUM_PARTICLES_PER_TILE), dtype=NUMBA_FLOAT)
        cuda.syncthreads()
        # Read data
        for i in range(SPATIAL_DIM):
            tile1_positions[i] = sorted_positions[i, tile1_particle_index]
        tile1_charges = sorted_charges[0, tile1_particle_index]
        # Local data
        local_forces = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        vec = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        energy = NUMBA_FLOAT(0)
        inverse_k = inverse_k[0]
        inverse_sqrt_pi = NUMBA_FLOAT(1) / math.sqrt(NUMBA_FLOAT(math.pi))
        cutoff_radius = cutoff_radius[0]
        for i in range(SPATIAL_DIM):
            local_forces[i] = 0
        for neighbor_index in range(tile_neighbors.shape[1]):
            tile_id2 = tile_neighbors[tile_id1, neighbor_index]
            tile2_particle_index = tile_id2 * NUM_PARTICLES_PER_TILE + local_thread_x
            cuda.syncthreads()
            for i in range(SPATIAL_DIM):
                tile2_positions[i, local_thread_y, local_thread_x] = sorted_positions[i, tile2_particle_index]
            tile2_charges[local_thread_y, local_thread_x] = sorted_charges[0, tile2_particle_index]
            exclusion_flag = exclusion_map[neighbor_index, tile1_particle_index]
            cuda.syncthreads()
            if tile_id2 == -1:
                break
            # Computation
            for particle_index in range(NUM_PARTICLES_PER_TILE):
                if exclusion_flag >> particle_index & 0b1:
                    continue
                r = NUMBA_FLOAT(0)
                for i in range(SPATIAL_DIM):
                    vec[i] = tile2_positions[i, local_thread_y, particle_index] - tile1_positions[i]
                    if vec[i] < - local_half_pbc_matrix[i]:
                        vec[i] += local_pbc_matrix[i]
                    elif vec[i] > local_half_pbc_matrix[i]:
                        vec[i] -= local_pbc_matrix[i]
                    r += vec[i]**2
                r = math.sqrt(r)
                if r < cutoff_radius:
                    e1e2_over_k = tile1_charges * tile2_charges[local_thread_y, particle_index] * inverse_k
                    inverse_r = NUMBA_FLOAT(1) / r
                    energy += e1e2_over_k * inverse_r * NUMBA_FLOAT(0.5)
                    force_val = - e1e2_over_k * inverse_r**3
                    for i in range(SPATIAL_DIM):
                        local_forces[i] += force_val * vec[i]
        for i in range(SPATIAL_DIM):
            cuda.atomic.add(sorted_forces, (i, tile1_particle_index), local_forces[i])
        cuda.atomic.add(potential_energy, 0, energy)

    def update(self):
        self._check_bound_state()
        self._forces = cp.zeros(self._parent_ensemble.state.matrix_shape, CUPY_FLOAT)
        self._potential_energy = cp.zeros([1], CUPY_FLOAT)
        # Update
        self._potential_energy = cp.zeros([1], CUPY_FLOAT)
        sorted_forces = cp.zeros((SPATIAL_DIM, self._parent_ensemble.tile_list.num_tiles * NUM_PARTICLES_PER_TILE), CUPY_FLOAT)
        thread_per_block = (NUM_PARTICLES_PER_TILE, TILES_PER_THREAD)
        block_per_grid = (int(np.ceil(self._parent_ensemble.tile_list.num_tiles / TILES_PER_THREAD)))
        self._update_electrostatic_cutoff[block_per_grid, thread_per_block](
            self._device_inverse_k,
            self._device_cutoff_radius,
            self._parent_ensemble.state.device_pbc_matrix,
            self._parent_ensemble.state.sorted_positions,
            self._parent_ensemble.topology.device_sorted_charges,
            self._parent_ensemble.topology.device_exclusion_map,
            self._parent_ensemble.tile_list.tile_neighbors,
            sorted_forces, self._potential_energy
        )
        self._forces = self._parent_ensemble.tile_list.unsort_matrix(sorted_forces)