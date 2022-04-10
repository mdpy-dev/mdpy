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
from numba import cuda
from operator import floordiv
from mdpy import env
from mdpy.core import Ensemble, NUM_NEIGHBOR_CELLS, NEIGHBOR_CELL_TEMPLATE
from mdpy.constraint import Constraint
from mdpy.utils import *
from mdpy.unit import *

class ElectrostaticCutoffConstraint(Constraint):
    def __init__(self, parameters=None, force_id: int = 0, force_group: int = 0) -> None:
        super().__init__(parameters, force_id=force_id, force_group=force_group)
        # Attributes
        self._int_parameters = []
        self._float_parameters = []
        self._device_k = cuda.to_device(np.array([4 * np.pi * EPSILON0.value], dtype=env.NUMPY_FLOAT))
        # Kernel
        self._update = cuda.jit(nb.void(
            env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[::1], env.NUMBA_INT[:, ::1], env.NUMBA_FLOAT[:, ::1],
            env.NUMBA_INT[:, ::1], env.NUMBA_INT[:, :, :, ::1], env.NUMBA_INT[::1], env.NUMBA_INT[:, ::1],
            env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[::1]
        ))(self._update_kernel)

    def __repr__(self) -> str:
        return '<mdpy.constraint.ElectrostaticCutoffConstraint object>'

    def __str__(self) -> str:
        return 'Cutoff electrostatic constraint'

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._force_id = ensemble.constraints.index(self)
        self._device_charges = cuda.to_device(self._parent_ensemble.topology.charges)
        self._device_pbc_matrix = cuda.to_device(self._parent_ensemble.state.pbc_matrix)
        self._device_cutoff_radius = cuda.to_device(np.array([self._cutoff_radius], dtype=env.NUMPY_FLOAT))
        self._device_bonded_particles = cuda.to_device(self._parent_ensemble.topology.bonded_particles)
        self._device_scaling_particles = cuda.to_device(self._parent_ensemble.topology.scaling_particles)

    @staticmethod
    def _update_kernel(
        positions, charges, k, bonded_particles, pbc_matrix,
        particle_cell_index, cell_list, num_cell_vec, neighbor_cell_template,
        forces, potential_energy
    ):
        thread_x, thread_y = cuda.grid(2)
        num_particles_per_cell = cell_list.shape[3]
        num_particles = positions.shape[0]
        id1 = thread_x
        if id1 >= num_particles:
            return None
        cell_id = floordiv(thread_y, num_particles_per_cell)
        cell_particle_id = thread_y % num_particles_per_cell
        if cell_id >= NUM_NEIGHBOR_CELLS:
            return None
        x = particle_cell_index[id1, 0] + neighbor_cell_template[cell_id, 0]
        x = x - num_cell_vec[0] if x >= num_cell_vec[0] else x
        y = particle_cell_index[id1, 1] + neighbor_cell_template[cell_id, 1]
        y = y - num_cell_vec[1] if y >= num_cell_vec[1] else y
        z = particle_cell_index[id1, 2] + neighbor_cell_template[cell_id, 2]
        z = z - num_cell_vec[2] if z >= num_cell_vec[2] else z
        id2 = cell_list[x, y, z, cell_particle_id]
        if id1 == id2:
            return None
        if id2 == -1:
            return None
        for i in bonded_particles[id1, :]:
            if i == -1:
                break
            if id2 == i:
                return None
        e1 = charges[id1, 0]
        e2 = charges[id2, 0]
        x = (positions[id2, 0] - positions[id1, 0]) / pbc_matrix[0, 0]
        x = (x - round(x)) * pbc_matrix[0, 0]
        y = (positions[id2, 1] - positions[id1, 1]) / pbc_matrix[1, 1]
        y = (y - round(y)) * pbc_matrix[1, 1]
        z = (positions[id2, 2] - positions[id1, 2]) / pbc_matrix[2, 2]
        z = (z - round(z)) * pbc_matrix[2, 2]
        r = math.sqrt(x**2 + y**2 + z**2)
        scaled_x, scaled_y, scaled_z = x / r, y / r, z / r
        force_val = - e1 * e2 / k[0] / r**2
        force_x = scaled_x * force_val / 2
        force_y = scaled_y * force_val / 2
        force_z = scaled_z * force_val / 2
        cuda.atomic.add(forces, (id1, 0), force_x)
        cuda.atomic.add(forces, (id1, 1), force_y)
        cuda.atomic.add(forces, (id1, 2), force_z)
        cuda.atomic.add(forces, (id2, 0), -force_x)
        cuda.atomic.add(forces, (id2, 1), -force_y)
        cuda.atomic.add(forces, (id2, 2), -force_z)
        energy = e1 * e2 / k[0] / r / 2
        cuda.atomic.add(potential_energy, 0, energy)

    def update(self):
        self._forces = np.zeros_like(self._parent_ensemble.state.positions)
        self._potential_energy = np.zeros([1], dtype=env.NUMPY_FLOAT)
        device_positions = cuda.to_device(self._parent_ensemble.state.positions)
        device_particle_cell_index = cuda.to_device(self._parent_ensemble.state.cell_list.particle_cell_index)
        device_cell_list = cuda.to_device(self._parent_ensemble.state.cell_list.cell_list)
        device_num_cell_vec = cuda.to_device(self._parent_ensemble.state.cell_list.num_cell_vec)
        device_neighbor_cell_template = cuda.to_device(NEIGHBOR_CELL_TEMPLATE.astype(env.NUMPY_INT))
        device_forces = cuda.to_device(self._forces)
        device_potential_energy = cuda.to_device(self._potential_energy)
        thread_per_block = (8, 8)
        block_per_grid_x = int(np.ceil(
            self._parent_ensemble.topology.num_particles / thread_per_block[0]
        ))
        block_per_grid_y = int(np.ceil(
            self._parent_ensemble.state.cell_list.num_particles_per_cell * NUM_NEIGHBOR_CELLS / thread_per_block[1]
        ))
        block_per_grid = (block_per_grid_x, block_per_grid_y)
        self._update[block_per_grid, thread_per_block](
            device_positions, self._device_charges, self._device_k,
            self._device_bonded_particles, self._device_pbc_matrix,
            device_particle_cell_index, device_cell_list,
            device_num_cell_vec, device_neighbor_cell_template,
            device_forces, device_potential_energy
        )
        self._forces = device_forces.copy_to_host()
        self._potential_energy = device_potential_energy.copy_to_host()[0]