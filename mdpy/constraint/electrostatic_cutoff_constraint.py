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
from mdpy.core import MAX_NUM_BONDED_PARTICLES
from mdpy.core import NUM_NEIGHBOR_CELLS, DEVICE_NEIGHBOR_CELL_TEMPLATE
from mdpy.constraint import Constraint
from mdpy.utils import *
from mdpy.unit import *

THREAD_PER_BLOCK = (8, 8)

class ElectrostaticCutoffConstraint(Constraint):
    def __init__(self, cutoff_radius=Quantity(12, angstrom)) -> None:
        super().__init__()
        # Attributes
        self._cutoff_radius = check_quantity_value(cutoff_radius, default_length_unit)
        self._device_cutoff_radius = cuda.to_device(np.array([self._cutoff_radius]))
        self._int_parameters = []
        self._float_parameters = []
        self._device_k = cuda.to_device(np.array([4 * np.pi * EPSILON0.value], dtype=NUMPY_FLOAT))
        # Kernel
        self._update = cuda.jit(nb.void(
            NUMBA_FLOAT[:, ::1], # positions
            NUMBA_FLOAT[:, ::1], # charges
            NUMBA_FLOAT[::1], # k
            NUMBA_FLOAT[:, ::1], # pbc_matrix
            NUMBA_FLOAT[::1], # cutoff_radius
            NUMBA_INT[:, ::1], # bonded_particles
            NUMBA_INT[:, ::1], # particle_cell_index
            NUMBA_INT[:, :, :, ::1], # cell_list
            NUMBA_INT[::1], # num_cell_vec
            NUMBA_INT[:, ::1], # neighbor_cell_termplate
            NUMBA_FLOAT[:, ::1], # forces
            NUMBA_FLOAT[::1] # potential_energy
        ))(self._update_kernel)

    def __repr__(self) -> str:
        return '<mdpy.constraint.ElectrostaticCutoffConstraint object>'

    def __str__(self) -> str:
        return 'Cutoff electrostatic constraint'

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._constraint_id = ensemble.constraints.index(self)
        self._device_charges = cuda.to_device(self._parent_ensemble.topology.charges)
        self._device_pbc_matrix = cuda.to_device(self._parent_ensemble.state.pbc_matrix)
        self._device_cutoff_radius = cuda.to_device(np.array([self._cutoff_radius], dtype=NUMPY_FLOAT))
        self._device_bonded_particles = cuda.to_device(self._parent_ensemble.topology.bonded_particles)
        self._device_scaling_particles = cuda.to_device(self._parent_ensemble.topology.scaling_particles)

    @staticmethod
    def _update_kernel(
        positions, charges, k, pbc_matrix, cutoff_radius,
        bonded_particles, particle_cell_index,
        cell_list, num_cell_vec, neighbor_cell_template,
        forces, potential_energy
    ):
        thread_x, thread_y = cuda.grid(2)
        num_particles_per_cell = cell_list.shape[3]
        num_particles = positions.shape[0]

        id1 = thread_x
        if id1 >= num_particles:
            return None
        cell_id = thread_y
        if cell_id >= NUM_NEIGHBOR_CELLS:
            return None
        # Shared array
        thread_x = cuda.threadIdx.x
        thread_y = cuda.threadIdx.y
        # PBC matrix
        shared_pbc_matrix = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        shared_half_pbc_matrix = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        # Bonded particle
        shared_bonded_particles = cuda.shared.array(
            shape=(THREAD_PER_BLOCK[0], MAX_NUM_BONDED_PARTICLES), dtype=NUMBA_INT
        )
        # Parameters
        shared_parameters = cuda.shared.array(shape=(THREAD_PER_BLOCK[0], 4), dtype=NUMBA_FLOAT)
        # num_cell_vec
        shared_num_cell_vec = cuda.shared.array(shape=(3), dtype=NUMBA_INT)
        if thread_y == 0:
            if thread_x == 0:
                shared_pbc_matrix[0] = pbc_matrix[0, 0]
                shared_pbc_matrix[1] = pbc_matrix[1, 1]
                shared_pbc_matrix[2] = pbc_matrix[2, 2]
                shared_half_pbc_matrix[0] = shared_pbc_matrix[0] / 2
                shared_half_pbc_matrix[1] = shared_pbc_matrix[1] / 2
                shared_half_pbc_matrix[2] = shared_pbc_matrix[2] / 2
                shared_num_cell_vec[0] = num_cell_vec[0]
                shared_num_cell_vec[1] = num_cell_vec[1]
                shared_num_cell_vec[2] = num_cell_vec[2]
            for i in range(MAX_NUM_BONDED_PARTICLES):
                shared_bonded_particles[thread_x, i] = bonded_particles[id1, i]
        cuda.syncthreads()

        cell_id_x = particle_cell_index[id1, 0] + neighbor_cell_template[cell_id, 0]
        cell_id_x = cell_id_x - shared_num_cell_vec[0] if cell_id_x >= shared_num_cell_vec[0] else cell_id_x
        cell_id_y = particle_cell_index[id1, 1] + neighbor_cell_template[cell_id, 1]
        cell_id_y = cell_id_y - shared_num_cell_vec[1] if cell_id_y >= shared_num_cell_vec[1] else cell_id_y
        cell_id_z = particle_cell_index[id1, 2] + neighbor_cell_template[cell_id, 2]
        cell_id_z = cell_id_z - shared_num_cell_vec[2] if cell_id_z >= shared_num_cell_vec[2] else cell_id_z
        # id1 attribute
        positions_id1_x = positions[id1, 0]
        positions_id1_y = positions[id1, 1]
        positions_id1_z = positions[id1, 2]
        # cutoff
        cutoff_radius = cutoff_radius[0]
        force_x = 0
        force_y = 0
        force_z = 0
        energy = 0
        e1 = charges[id1, 0]
        k = k[0]
        for index in range(num_particles_per_cell):
            id2 = cell_list[cell_id_x, cell_id_y, cell_id_z, index]
            if id1 == id2: # self-self term
                continue
            if id2 == -1: # self-padding term
                break
            is_continue = False
            for i in shared_bonded_particles[thread_x, :]:
                if i == -1: # padding of bonded particle
                    break
                elif id2 == i: # self-bonded particle term
                    is_continue = True
                    break
            if is_continue:
                continue
            x = (positions[id2, 0] - positions_id1_x)
            if x >= shared_half_pbc_matrix[0]:
                x -= shared_pbc_matrix[0]
            elif x <= -shared_half_pbc_matrix[0]:
                x += shared_pbc_matrix[0]
            y = (positions[id2, 1] - positions_id1_y)
            if y >= shared_half_pbc_matrix[1]:
                y -= shared_pbc_matrix[1]
            elif y <= -shared_half_pbc_matrix[1]:
                y += shared_pbc_matrix[1]
            z = (positions[id2, 2] - positions_id1_z)
            if z >= shared_half_pbc_matrix[2]:
                z -= shared_pbc_matrix[2]
            elif z <= -shared_half_pbc_matrix[2]:
                z += shared_pbc_matrix[2]
            r = math.sqrt(x**2 + y**2 + z**2)
            if r <= cutoff_radius:
                scaled_x, scaled_y, scaled_z = x / r, y / r, z / r
                e1e2 = e1 * charges[id2, 0]
                force_val = - e1e2 / k / r**2
                force_x += scaled_x * force_val
                force_y += scaled_y * force_val
                force_z += scaled_z * force_val
                # Energy still need divided by 2
                energy += e1e2 / k / r / 2
        cuda.atomic.add(forces, (id1, 0), force_x)
        cuda.atomic.add(forces, (id1, 1), force_y)
        cuda.atomic.add(forces, (id1, 2), force_z)
        cuda.atomic.add(potential_energy, 0, energy)

    def update(self):
        self._check_bound_state()
        self._forces = cp.zeros_like(self._parent_ensemble.state.positions, CUPY_FLOAT)
        self._potential_energy = cp.zeros([1], CUPY_FLOAT)

        block_per_grid_x = int(np.ceil(
            self._parent_ensemble.topology.num_particles / THREAD_PER_BLOCK[0]
        ))
        block_per_grid_y = int(np.ceil(
            NUM_NEIGHBOR_CELLS / THREAD_PER_BLOCK[1]
        ))
        block_per_grid = (block_per_grid_x, block_per_grid_y)
        self._update[block_per_grid, THREAD_PER_BLOCK](
            self._parent_ensemble.state.device_positions,
            self._device_charges, self._device_k,
            self._device_pbc_matrix, self._device_cutoff_radius,
            self._device_bonded_particles,
            self._parent_ensemble.state.cell_list.device_particle_cell_index,
            self._parent_ensemble.state.cell_list.device_cell_list,
            self._parent_ensemble.state.cell_list.device_num_cell_vec,
            DEVICE_NEIGHBOR_CELL_TEMPLATE,
            self._forces, self._potential_energy
        )