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
from mdpy import SPATIAL_DIM, env
from mdpy.core import Ensemble
from mdpy.core import NUM_NEIGHBOR_CELLS, NEIGHBOR_CELL_TEMPLATE
from mdpy.core import MAX_NUM_BONDED_PARTICLES, MAX_NUM_SCALING_PARTICLES
from mdpy.constraint import Constraint
from mdpy.utils import *
from mdpy.unit import *

THREAD_PER_BLOCK = (32, 4)
class CharmmVDWConstraint(Constraint):
    def __init__(self, parameter_dict: dict, cutoff_radius=Quantity(12, angstrom)) -> None:
        super().__init__()
        # Attributes
        self._parameter_dict = parameter_dict
        self._cutoff_radius = check_quantity_value(cutoff_radius, default_length_unit)
        self._device_cutoff_radius = cuda.to_device(np.array([self._cutoff_radius]))
        self._parameters_list = []
        # Kernel
        self._update = cuda.jit(nb.void(
            env.NUMBA_FLOAT[:, ::1], # positions
            env.NUMBA_FLOAT[:, ::1], # parameters
            env.NUMBA_FLOAT[:, ::1], # pbc_matrix
            env.NUMBA_FLOAT[::1], # cutoff_radius
            env.NUMBA_INT[:, ::1], # bonded_particle
            env.NUMBA_INT[:, ::1], # scaling_particles
            env.NUMBA_INT[:, ::1], # particle_cell_index
            env.NUMBA_INT[:, :, :, ::1], # cell_list
            env.NUMBA_INT[::1], # num_cell_vec
            env.NUMBA_INT[:, ::1], # neighbor_cell_template
            env.NUMBA_FLOAT[:, ::1], # forces
            env.NUMBA_FLOAT[::1] # potential_energy
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
            param = self._parameter_dict[particle.particle_type]
            if len(param) == 2:
                epsilon, sigma = param
                self._parameters_list.append([epsilon, sigma, epsilon, sigma])
            elif len(param) == 4:
                epsilon, sigma, epsilon14, sigma14 = param
                self._parameters_list.append([epsilon, sigma, epsilon14, sigma14])
        self._parameters_list = np.vstack(self._parameters_list).astype(env.NUMPY_FLOAT)
        self._device_parameters_list = cuda.to_device(self._parameters_list)
        self._device_bonded_particles = cuda.to_device(self._parent_ensemble.topology.bonded_particles)
        self._device_scaling_particles = cuda.to_device(self._parent_ensemble.topology.scaling_particles)

    @staticmethod
    def _update_kernel(
        positions, parameters, pbc_matrix, cutoff_radius,
        bonded_particles, scaling_particles,
        particle_cell_index, cell_list, num_cell_vec,
        neighbor_cell_template,
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
        shared_pbc_matrix = cuda.shared.array(
            shape=(SPATIAL_DIM), dtype=nb.float32
        )
        shared_half_pbc_matrix = cuda.shared.array(
            shape=(SPATIAL_DIM), dtype=nb.float32
        )
        # Bonded particle
        shared_bonded_particles = cuda.shared.array(
            shape=(THREAD_PER_BLOCK[0], MAX_NUM_BONDED_PARTICLES), dtype=nb.int32
        )
        # Scaling particles
        shared_scaling_particles = cuda.shared.array(
            shape=(THREAD_PER_BLOCK[0], MAX_NUM_SCALING_PARTICLES), dtype=nb.int32
        )
        # Parameters
        shared_parameters = cuda.shared.array(
            shape=(THREAD_PER_BLOCK[0], 4), dtype=nb.float32
        )
        # num_cell_vec
        shared_num_cell_vec = cuda.shared.array(
            shape=(3), dtype=nb.int32
        )
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
        elif thread_y == 1:
            for i in range(MAX_NUM_SCALING_PARTICLES):
                shared_scaling_particles[thread_x, i] = scaling_particles[id1, i]
            shared_parameters[thread_x, 0] = parameters[id1, 0]
            shared_parameters[thread_x, 1] = parameters[id1, 1]
            shared_parameters[thread_x, 2] = parameters[id1, 2]
            shared_parameters[thread_x, 3] = parameters[id1, 3]
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
                is_scaled = False
                for i in shared_scaling_particles[thread_x, :]:
                    if i == -1:
                        break
                    if id2 == i:
                        is_scaled = True
                        break
                if not is_scaled:
                    epsilon1, sigma1 = shared_parameters[thread_x, :2]
                    epsilon2, sigma2 = parameters[id2, :2]
                else:
                    epsilon1, sigma1 = shared_parameters[thread_x, 2:]
                    epsilon2, sigma2 = parameters[id2, 2:]
                epsilon, sigma = (
                    math.sqrt(epsilon1 * epsilon2),
                    (sigma1 + sigma2) / 2
                )
                scaled_r = sigma / r
                scaled_r6 = scaled_r**6
                scaled_r12 = scaled_r6**2
                force_val = - (2 * scaled_r12 - scaled_r6) / r * epsilon * 24
                force_x += scaled_x * force_val
                force_y += scaled_y * force_val
                force_z += scaled_z * force_val
                # Energy still need divided by 2
                energy += 2 * epsilon * (scaled_r12 - scaled_r6)
        cuda.atomic.add(forces, (id1, 0), force_x)
        cuda.atomic.add(forces, (id1, 1), force_y)
        cuda.atomic.add(forces, (id1, 2), force_z)
        cuda.atomic.add(potential_energy, 0, energy)

    def update(self):
        self._check_bound_state()
        self._forces = np.zeros_like(self._parent_ensemble.state.positions)
        self._potential_energy = np.zeros([1], dtype=env.NUMPY_FLOAT)
        # Device
        device_neighbor_cell_template = cuda.to_device(NEIGHBOR_CELL_TEMPLATE.astype(env.NUMPY_INT))
        device_forces = cuda.to_device(self._forces)
        device_potential_energy = cuda.to_device(self._potential_energy)
        block_per_grid_x = int(np.ceil(
            self._parent_ensemble.topology.num_particles / THREAD_PER_BLOCK[0]
        ))
        block_per_grid_y = int(np.ceil(
            NUM_NEIGHBOR_CELLS / THREAD_PER_BLOCK[1]
        ))
        block_per_grid = (block_per_grid_x, block_per_grid_y)
        self._update[block_per_grid, THREAD_PER_BLOCK](
            self._parent_ensemble.state.device_positions,
            self._device_parameters_list,
            self._parent_ensemble.state.device_pbc_matrix,
            self._device_cutoff_radius,
            self._device_bonded_particles,
            self._device_scaling_particles,
            self._parent_ensemble.state.cell_list.device_particle_cell_index,
            self._parent_ensemble.state.cell_list.device_cell_list,
            self._parent_ensemble.state.cell_list.device_num_cell_vec,
            device_neighbor_cell_template,
            device_forces, device_potential_energy
        )
        self._forces = device_forces.copy_to_host()
        self._potential_energy = device_potential_energy.copy_to_host()[0]
