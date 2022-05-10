#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : charmm_vdw_constraint.py
created time : 2021/10/12
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import math
from turtle import position
import numpy as np
import numba as nb
from numba import cuda
from mdpy import SPATIAL_DIM
from mdpy.environment import *
from mdpy.core import NUM_PARTICLES_PER_TILE, Ensemble
from mdpy.core import MAX_NUM_EXCLUDED_PARTICLES, MAX_NUM_SCALED_PARTICLES
from mdpy.constraint import Constraint
from mdpy.utils import *
from mdpy.unit import *

THREAD_PER_BLOCK = (NUM_PARTICLES_PER_TILE, 1)
class CharmmVDWConstraint(Constraint):
    def __init__(self, parameter_dict: dict, cutoff_radius=Quantity(12, angstrom)) -> None:
        super().__init__()
        # Attributes
        self._parameter_dict = parameter_dict
        self._cutoff_radius = check_quantity_value(cutoff_radius, default_length_unit)
        self._device_cutoff_radius = cp.array([self._cutoff_radius], CUPY_FLOAT)
        self._parameters = []
        # Kernel
        self._update = cuda.jit(nb.void(
            NUMBA_FLOAT[::1], # cutoff_radius
            NUMBA_FLOAT[:, ::1], # pbc_matrix
            NUMBA_FLOAT[:, ::1], # sorted_positions
            NUMBA_FLOAT[:, ::1], # sorted_parameters
            NUMBA_INT[:, ::1], # sorted_excluded_particles
            NUMBA_INT[:, ::1], # sorted_scaled_particles
            NUMBA_INT[::1], # sorted_matrix_mapping_index
            NUMBA_INT[:, ::1], # tile_neighbors
            NUMBA_FLOAT[:, ::1], # sorted_forces
            NUMBA_FLOAT[::1] # potential_energy
        ))(self._update_charmm_vdw_kernel)

    def __repr__(self) -> str:
        return '<mdpy.constraint.CharmmVDWConstraint object>'

    def __str__(self) -> str:
        return 'VDW constraint'

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._constraint_id = ensemble.constraints.index(self)
        self._parameters = []
        for particle in self._parent_ensemble.topology.particles:
            param = self._parameter_dict[particle.particle_type]
            if len(param) == 2:
                epsilon, sigma = param
                self._parameters.append([epsilon, sigma, epsilon, sigma])
            elif len(param) == 4:
                epsilon, sigma, epsilon14, sigma14 = param
                self._parameters.append([epsilon, sigma, epsilon14, sigma14])
        self._device_parameters = cp.array(np.vstack(self._parameters), CUPY_FLOAT)

    @staticmethod
    def _update_charmm_vdw_kernel(
        cutoff_radius,
        pbc_matrix,
        sorted_positions,
        sorted_parameters,
        sorted_excluded_particles,
        sorted_scaled_particles,
        sorted_matrix_mapping_index,
        tile_neighbors,
        sorted_forces, potential_energy
    ):
        # tile information
        tile_id1 = cuda.blockIdx.x
        tile_id2 = tile_neighbors[tile_id1, cuda.blockIdx.y]
        if tile_id2 == -1:
            return
        # Particle index information
        local_thread_x = cuda.threadIdx.x
        global_thread_x = tile_id1 * cuda.blockDim.x + local_thread_x
        tile2_start_index = tile_id2 * NUM_PARTICLES_PER_TILE
        # shared data
        shared_pbc_matrix = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        shared_half_pbc_matrix = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        if local_thread_x <= 2:
            shared_pbc_matrix[local_thread_x] = pbc_matrix[local_thread_x, local_thread_x]
            shared_half_pbc_matrix[local_thread_x] = shared_pbc_matrix[local_thread_x] * NUMBA_FLOAT(0.5)
        shared_positions = cuda.shared.array(shape=(SPATIAL_DIM, NUM_PARTICLES_PER_TILE), dtype=NUMBA_FLOAT)
        shared_parameters = cuda.shared.array(shape=(4, NUM_PARTICLES_PER_TILE), dtype=NUMBA_FLOAT)
        shared_particle_index_2 = cuda.shared.array(shape=(NUM_PARTICLES_PER_TILE), dtype=NUMBA_INT)
        index = tile2_start_index+local_thread_x
        shared_particle_index_2[local_thread_x] = sorted_matrix_mapping_index[index]
        if shared_particle_index_2[local_thread_x] != -1:
            for i in range(SPATIAL_DIM):
                shared_positions[i, local_thread_x] = sorted_positions[i, index]
            for i in range(4):
                shared_parameters[i, local_thread_x] = sorted_parameters[i, index]
        cuda.syncthreads()

        # Local data
        particle_1 = sorted_matrix_mapping_index[global_thread_x]
        if particle_1 == -1:
            return
        local_excluded_particles = cuda.local.array(shape=(MAX_NUM_EXCLUDED_PARTICLES), dtype=NUMBA_INT)
        num_excluded_particles = NUMBA_INT(0)
        for i in range(MAX_NUM_EXCLUDED_PARTICLES):
            local_excluded_particles[i] = sorted_excluded_particles[i, global_thread_x]
            if local_excluded_particles[i] == -1:
                break
            num_excluded_particles += 1
        local_scaled_particles = cuda.local.array(shape=(MAX_NUM_SCALED_PARTICLES), dtype=NUMBA_INT)
        num_scaled_particles = NUMBA_INT(0)
        for i in range(MAX_NUM_SCALED_PARTICLES):
            local_scaled_particles[i] = sorted_scaled_particles[i, global_thread_x]
            if local_scaled_particles[i] == -1:
                break
            num_scaled_particles += 1
        local_parameters = cuda.local.array(shape=(4), dtype=NUMBA_FLOAT)
        for i in range(4):
            local_parameters[i] = sorted_parameters[i, global_thread_x]
        local_positions = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        local_forces = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        vec = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        energy = NUMBA_FLOAT(0)
        cutoff_radius = cutoff_radius[0]
        for i in range(SPATIAL_DIM):
            local_positions[i] = sorted_positions[i, global_thread_x]
            local_forces[i] = 0

        # Computation
        for index in range(NUM_PARTICLES_PER_TILE):
            particle_2 = shared_particle_index_2[index]
            if particle_2 == -1:
                break
            if particle_1 == particle_2:
                continue
            is_excluded = False
            for i in range(num_excluded_particles):
                if particle_2 == local_excluded_particles[i]:
                    is_excluded = True
                    break
            if is_excluded:
                continue

            r = NUMBA_FLOAT(0)
            for i in range(SPATIAL_DIM):
                vec[i] = shared_positions[i, index] - local_positions[i]
                if vec[i] < - shared_half_pbc_matrix[i]:
                    vec[i] += shared_pbc_matrix[i]
                elif vec[i] > shared_half_pbc_matrix[i]:
                    vec[i] -= shared_pbc_matrix[i]
                r += vec[i]**2
            r = math.sqrt(r)
            if r <= cutoff_radius:
                for i in range(SPATIAL_DIM):
                    vec[i] /= r
                is_scaled = False
                for i in range(num_scaled_particles):
                    if particle_2 == local_scaled_particles[i]:
                        is_scaled = True
                        break
                if not is_scaled:
                    epsilon = math.sqrt(local_parameters[0] * shared_parameters[0, index])
                    sigma = (local_parameters[1] + shared_parameters[1, index]) * NUMBA_FLOAT(0.5)
                else:
                    epsilon = math.sqrt(local_parameters[2] * shared_parameters[2, index])
                    sigma = (local_parameters[3] + shared_parameters[3, index]) * NUMBA_FLOAT(0.5)
                scaled_r = sigma / r
                scaled_r6 = scaled_r**6
                scaled_r12 = scaled_r6**2
                energy += NUMBA_FLOAT(2) * epsilon * (scaled_r12 - scaled_r6)
                force_val = - (NUMBA_FLOAT(2) * scaled_r12 - scaled_r6) / r * epsilon * NUMBA_FLOAT(24)
                for i in range(SPATIAL_DIM):
                    local_forces[i] += force_val * vec[i]
        for i in range(SPATIAL_DIM):
            cuda.atomic.add(sorted_forces, (i, global_thread_x), local_forces[i])
        cuda.atomic.add(potential_energy, 0, energy)

    def update(self):
        self._check_bound_state()
        sorted_forces = cp.zeros((SPATIAL_DIM, self._parent_ensemble.tile_list.num_tiles * NUM_PARTICLES_PER_TILE), CUPY_FLOAT)
        self._potential_energy = cp.zeros([1], CUPY_FLOAT)
        sorted_positions = self._parent_ensemble.tile_list.sort_matrix(self._parent_ensemble.state.positions)
        device_sorted_parameter_list = self._parent_ensemble.tile_list.sort_matrix(self._device_parameters)
        block_per_grid_x = self._parent_ensemble.tile_list.num_tiles
        block_per_grid_y = self._parent_ensemble.tile_list.tile_neighbors.shape[1]
        block_per_grid = (block_per_grid_x, block_per_grid_y)
        # Device
        self._update[block_per_grid, THREAD_PER_BLOCK](
            self._device_cutoff_radius,
            self._parent_ensemble.state.device_pbc_matrix,
            sorted_positions,
            device_sorted_parameter_list,
            self._parent_ensemble.topology.device_sorted_excluded_particles,
            self._parent_ensemble.topology.device_sorted_scaled_particles,
            self._parent_ensemble.tile_list.sorted_matrix_mapping_index,
            self._parent_ensemble.tile_list.tile_neighbors,
            sorted_forces, self._potential_energy
        )
        self._forces = self._parent_ensemble.tile_list.unsort_matrix(sorted_forces)

if __name__ == '__main__':
    import os
    import mdpy as md
    from mdpy.unit import *
    data_dir = '/home/zhenyuwei/nutstore/ZhenyuWei/Note_Research/mdpy/mdpy/benchmark/data'
    psf_file = os.path.join(data_dir, 'str.psf')
    pdb_file = os.path.join(data_dir, 'str.pdb')
    cutoff_radius = Quantity(8, angstrom)
    # IO
    psf = md.io.PSFParser(psf_file)
    pdb = md.io.PDBParser(pdb_file)
    charmm_prm = md.io.CharmmTopparParser(
        os.path.join(data_dir, 'par_all36_prot.prm'),
        os.path.join(data_dir, 'toppar_water_ions_namd.str'),
    )
    # constraint
    ensemble = md.core.Ensemble(psf.topology, pdb.pbc_matrix)
    constraint = CharmmVDWConstraint(charmm_prm.parameters['nonbonded'], cutoff_radius=cutoff_radius)
    ensemble.add_constraints(constraint)
    id1, id2 = 6570, 6569
    ensemble.state.set_positions(pdb.positions)
    ensemble.update_tile_list()
    print(ensemble.tile_list.tile_neighbors.shape)
    constraint.update()
    print(Quantity(constraint.forces, default_force_unit).convert_to(kilojoule_permol_over_nanometer).value)