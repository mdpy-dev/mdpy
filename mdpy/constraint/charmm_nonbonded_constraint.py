#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : charmm_nonbonded_constraint.py
created time : 2021/10/12
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np
import numba as nb
import math
from numba import cuda
from operator import floordiv
from . import Constraint, NUM_NEIGHBOR_CELLS, NEIGHBOR_CELL_TEMPLATE
from .. import env
from ..ensemble import Ensemble
from ..utils import *
from ..unit import *

class CharmmNonbondedConstraint(Constraint):
    def __init__(self, params, cutoff_radius=12, force_id: int = 0, force_group: int = 0) -> None:
        super().__init__(params, force_id=force_id, force_group=force_group)
        self._cutoff_radius = check_quantity_value(cutoff_radius, default_length_unit)
        self._params_list = []
        self._neighbor_list = []
        self._neighbor_distance = []
        self._num_nonbonded_pairs = 0
        if env.platform == 'CPU':
            self._kernel = nb.njit((
                env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[:, ::1],
                env.NUMBA_FLOAT, env.NUMBA_INT[:, ::1], env.NUMBA_INT[:, ::1],
                env.NUMBA_INT[:, ::1], env.NUMBA_INT[:, :, :, ::1], env.NUMBA_INT[::1], env.NUMBA_INT[:, ::1]
            ))(self.cpu_kernel)
        elif env.platform == 'CUDA':
            self._kernel = cuda.jit(nb.void(
                env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[:, ::1],
                env.NUMBA_FLOAT[::1], env.NUMBA_INT[:, ::1], env.NUMBA_INT[:, ::1],
                env.NUMBA_INT[:, ::1], env.NUMBA_INT[:, :, :, ::1], env.NUMBA_INT[::1], env.NUMBA_INT[:, ::1],
                env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[::1]
            ))(self.cuda_kernel)
        
    def __repr__(self) -> str:
        return '<mdpy.constraint.CharmmNonbondedConstraint object>'

    __str__ = __repr__

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._force_id = ensemble.constraints.index(self)
        self._params_list = []
        for particle in self._parent_ensemble.topology.particles:
            param = self._params[particle.particle_name]
            if len(param) == 2:
                epsilon, sigma = param
                self._params_list.append([epsilon, sigma, epsilon, sigma])
            elif len(param) == 4:
                epsilon, sigma, epsilon14, sigma14 = param
                self._params_list.append([epsilon, sigma, epsilon14, sigma14])
        self._params_list = np.vstack(self._params_list).astype(env.NUMPY_FLOAT)
        self._device_params_list = cuda.to_device(self._params_list)

    @staticmethod
    def cpu_kernel(
        positions, params, pbc_matrix, pbc_inv, cutoff_radius,
        bonded_particles, scaling_particles, 
        particle_cell_index, cell_list, num_cell_vec, neighbor_cell_template,
    ):
        forces = np.zeros_like(positions)
        potential_energy = forces[0, 0]
        int_type = particle_cell_index.dtype
        num_particles = positions.shape[0]
        num_neighbor_cells = neighbor_cell_template.shape[0]
        for id1 in range(num_particles):
            cur_bonded_particles = bonded_particles[id1][bonded_particles[id1] != -1]
            cur_scaling_particles = scaling_particles[id1][scaling_particles[id1] != -1]
            neighbor_cell = neighbor_cell_template + particle_cell_index[id1]
            wrap_flag = (neighbor_cell >= num_cell_vec).astype(int_type)
            neighbor_cell -= wrap_flag * num_cell_vec
            for cell_index in range(num_neighbor_cells):
                i, j, k = neighbor_cell[cell_index, :]
                neighbors = [i for i in cell_list[i, j, k, :] if not i in cur_bonded_particles and i != -1 and i != id1]
                for id2 in neighbors:
                    force_vec = unwrap_vec(
                        positions[id2] - positions[id1],
                        pbc_matrix, pbc_inv
                    )
                    r = np.linalg.norm(force_vec)
                    if r <= cutoff_radius:
                        force_vec /= r
                        if id2 in cur_scaling_particles:
                            epsilon1, sigma1 = params[id1, 2:]
                            epsilon2, sigma2 = params[id2, 2:]
                        else:
                            epsilon1, sigma1 = params[id1, :2]
                            epsilon2, sigma2 = params[id2, :2]
                        epsilon, sigma = (
                            np.sqrt(epsilon1 * epsilon2),
                            (sigma1 + sigma2) / 2
                        )
                        scaled_r = sigma / r
                        force_val = - (2 * scaled_r**12 - scaled_r**6) / r * epsilon * 24 # Sequence for small number divide small number
                        force = force_vec * force_val / 2
                        forces[id1, :] += force
                        forces[id2, :] -= force
                        potential_energy += 4 * epsilon * (scaled_r**12 - scaled_r**6) / 2
        return forces, potential_energy

    @staticmethod
    def cuda_kernel(
        positions, params, pbc_matrix, cutoff_radius,
        bonded_particles, scaling_particles, 
        particle_cell_index, cell_list, num_cell_vec, neighbor_cell_template,
        forces, potential_energy
    ):
        thread_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        thread_y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
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
            elif id2 == i:
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
                epsilon1, sigma1 = params[id1, :2]
                epsilon2, sigma2 = params[id2, :2]
            else:
                epsilon1, sigma1 = params[id1, 2:]
                epsilon2, sigma2 = params[id2, 2:]
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
        if env.platform == 'CPU':
                self._forces, self._potential_energy = self._kernel(
                self._parent_ensemble.state.positions, self._params_list, 
                *self._parent_ensemble.state.pbc_info, self._cutoff_radius, 
                self._parent_ensemble.topology.bonded_particles, 
                self._parent_ensemble.topology.scaling_particles,
                self._parent_ensemble.state.cell_list.particle_cell_index,
                self._parent_ensemble.state.cell_list.cell_list,
                self._parent_ensemble.state.cell_list.num_cell_vec,
                NEIGHBOR_CELL_TEMPLATE.astype(env.NUMPY_INT)
            )
        elif env.platform == 'CUDA':
            self._forces = np.zeros_like(self._parent_ensemble.state.positions)
            self._potential_energy = np.zeros([1], dtype=env.NUMPY_FLOAT)
            d_positions = cuda.to_device(self._parent_ensemble.state.positions)
            d_pbc_martix = cuda.to_device(self._parent_ensemble.state.pbc_matrix)
            d_cutoff_radius = cuda.to_device(np.array([self._cutoff_radius], dtype=env.NUMPY_FLOAT))
            d_bonded_particles = cuda.to_device(self._parent_ensemble.topology.bonded_particles)
            d_scaling_particles = cuda.to_device(self._parent_ensemble.topology.scaling_particles)
            d_particle_cell_index = cuda.to_device(self._parent_ensemble.state.cell_list.particle_cell_index)
            d_cell_list = cuda.to_device(self._parent_ensemble.state.cell_list.cell_list)
            d_num_cell_vec = cuda.to_device(self._parent_ensemble.state.cell_list.num_cell_vec)
            d_neighbor_cell_template = cuda.to_device(NEIGHBOR_CELL_TEMPLATE.astype(env.NUMPY_INT))
            d_forces = cuda.to_device(self._forces)
            d_potential_energy = cuda.to_device(self._potential_energy)
            thread_per_block = (8, 8)
            block_per_grid_x = int(np.ceil(
                self._parent_ensemble.topology.num_particles / thread_per_block[0]
            ))
            block_per_grid_y = int(np.ceil(
                self._parent_ensemble.state.cell_list.num_particles_per_cell * NUM_NEIGHBOR_CELLS / thread_per_block[1]
            ))
            block_per_grid = (block_per_grid_x, block_per_grid_y)
            self._kernel[block_per_grid, thread_per_block](
                d_positions, self._device_params_list, d_pbc_martix, d_cutoff_radius, 
                d_bonded_particles, d_scaling_particles, 
                d_particle_cell_index, d_cell_list, d_num_cell_vec,
                d_neighbor_cell_template,
                d_forces, d_potential_energy
            )
            self._forces = d_forces.copy_to_host()
            self._potential_energy = d_potential_energy.copy_to_host()[0]
    
    @property
    def num_nonbonded_pairs(self):
        return self._num_nonbonded_pairs