#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : charmm_vdw_constraint.py
created time : 2021/10/12
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

class CharmmVDWConstraint(Constraint):
    def __init__(self, parameters, cutoff_radius=12, force_id: int = 0, force_group: int = 0) -> None:
        super().__init__(parameters, force_id=force_id, force_group=force_group)
        # Attributes
        self._cutoff_radius = check_quantity_value(cutoff_radius, default_length_unit)
        self._device_cutoff_radius = cuda.to_device(np.array([self._cutoff_radius]))
        self._parameters_list = []
        # Kernel
        self._update = cuda.jit(nb.void(
            env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[:, ::1],
            env.NUMBA_FLOAT[::1], env.NUMBA_INT[:, ::1], env.NUMBA_INT[:, ::1],
            env.NUMBA_INT[:, ::1], env.NUMBA_INT[:, :, :, ::1], env.NUMBA_INT[::1], env.NUMBA_INT[:, ::1],
            env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[::1]
        ))(self._update_kernel)

    def __repr__(self) -> str:
        return '<mdpy.constraint.CharmmVDWConstraint object>'

    def __str__(self) -> str:
        return 'VDW constraint'

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._force_id = ensemble.constraints.index(self)
        self._parameters_list = []
        for particle in self._parent_ensemble.topology.particles:
            param = self._parameters[particle.particle_type]
            if len(param) == 2:
                epsilon, sigma = param
                self._parameters_list.append([epsilon, sigma, epsilon, sigma])
            elif len(param) == 4:
                epsilon, sigma, epsilon14, sigma14 = param
                self._parameters_list.append([epsilon, sigma, epsilon14, sigma14])
        self._parameters_list = np.vstack(self._parameters_list).astype(env.NUMPY_FLOAT)
        self._device_parameters_list = cuda.to_device(self._parameters_list)
        self._device_pbc_matrix = cuda.to_device(self._parent_ensemble.state.pbc_matrix)
        self._device_bonded_particles = cuda.to_device(self._parent_ensemble.topology.bonded_particles)
        self._device_scaling_particles = cuda.to_device(self._parent_ensemble.topology.scaling_particles)

    @staticmethod
    def _update_kernel(
        positions, parameters, pbc_matrix, cutoff_radius,
        bonded_particles, scaling_particles,
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
        if id1 == id2: # self-self term
            return None
        if id2 == -1: # self-padding term
            return None
        for i in bonded_particles[id1, :]:
            if i == -1: # padding of bonded particle
                break
            elif id2 == i: # self-bonded particle term
                return None
        x = (positions[id2, 0] - positions[id1, 0]) / pbc_matrix[0, 0]
        x = (x - round(x)) * pbc_matrix[0, 0]
        y = (positions[id2, 1] - positions[id1, 1]) / pbc_matrix[1, 1]
        y = (y - round(y)) * pbc_matrix[1, 1]
        z = (positions[id2, 2] - positions[id1, 2]) / pbc_matrix[2, 2]
        z = (z - round(z)) * pbc_matrix[2, 2]
        r = math.sqrt(x**2 + y**2 + z**2)
        if r <= cutoff_radius[0]:
            scaled_x, scaled_y, scaled_z = x / r, y / r, z / r
            is_scaled = False
            for i in scaling_particles[id1, :]:
                if id2 == i:
                    is_scaled = True
                    break
            if not is_scaled:
                epsilon1, sigma1 = parameters[id1, :2]
                epsilon2, sigma2 = parameters[id2, :2]
            else:
                epsilon1, sigma1 = parameters[id1, 2:]
                epsilon2, sigma2 = parameters[id2, 2:]
            epsilon, sigma = (
                math.sqrt(epsilon1 * epsilon2),
                (sigma1 + sigma2) / 2
            )
            scaled_r = sigma / r
            force_val = - (2 * scaled_r**12 - scaled_r**6) / r * epsilon * 24 # Sequence for small number divide small number
            force_x = scaled_x * force_val / 2
            force_y = scaled_y * force_val / 2
            force_z = scaled_z * force_val / 2
            cuda.atomic.add(forces, (id1, 0), force_x)
            cuda.atomic.add(forces, (id1, 1), force_y)
            cuda.atomic.add(forces, (id1, 2), force_z)
            cuda.atomic.add(forces, (id2, 0), -force_x)
            cuda.atomic.add(forces, (id2, 1), -force_y)
            cuda.atomic.add(forces, (id2, 2), -force_z)
            energy = 2 * epsilon * (scaled_r**12 - scaled_r**6)
            cuda.atomic.add(potential_energy, 0, energy)

    def update(self):
        self._check_bound_state()
        self._forces = np.zeros_like(self._parent_ensemble.state.positions)
        self._potential_energy = np.zeros([1], dtype=env.NUMPY_FLOAT)
        # Device
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
            device_positions, self._device_parameters_list,
            self._device_pbc_matrix, self._device_cutoff_radius,
            self._device_bonded_particles, self._device_scaling_particles,
            device_particle_cell_index, device_cell_list, device_num_cell_vec,
            device_neighbor_cell_template,
            device_forces, device_potential_energy
        )
        self._forces = device_forces.copy_to_host()
        self._potential_energy = device_potential_energy.copy_to_host()[0]
