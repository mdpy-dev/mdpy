#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : charmm_vdw_constraint.py
created time : 2021/10/12
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import math
import operator
import numpy as np
import numba as nb
from numba import cuda
from mdpy import SPATIAL_DIM
from mdpy.environment import *
from mdpy.core import DEVICE_NEIGHBOR_CELL_TEMPLATE, NUM_NEIGHBOR_CELLS, NUM_PARTICLES_PER_TILE, Ensemble
from mdpy.core import MAX_NUM_EXCLUDED_PARTICLES, MAX_NUM_SCALED_PARTICLES
from mdpy.constraint import Constraint
from mdpy.utils import *
from mdpy.unit import *

THREAD_PER_BLOCK = (32, 32)
class CharmmVDWConstraint(Constraint):
    def __init__(self, parameter_dict: dict, cutoff_radius=Quantity(12, angstrom)) -> None:
        super().__init__()
        # Attributes
        self._parameter_dict = parameter_dict
        self._cutoff_radius = check_quantity_value(cutoff_radius, default_length_unit)
        self._device_cutoff_radius = cp.array([self._cutoff_radius], CUPY_FLOAT)
        self._parameters_list = []
        # Kernel
        self._update = cuda.jit(nb.void(
            NUMBA_FLOAT[:, ::1], # positions
            NUMBA_FLOAT[:, ::1], # parameters
            NUMBA_INT[:, ::1], # excluded_particle
            NUMBA_INT[:, ::1], # scaled_particles
            NUMBA_FLOAT[::1], # cutoff_radius
            NUMBA_FLOAT[:, ::1], # pbc_matrix
            NUMBA_INT[:, ::1], # tile_list
            NUMBA_INT[:, ::1], # tile_neighbors
            NUMBA_FLOAT[:, ::1], # forces
            NUMBA_FLOAT[::1] # potential_energy
        ))(self._update_charmm_vdw_kernel)

    def __repr__(self) -> str:
        return '<mdpy.constraint.CharmmVDWConstraint object>'

    def __str__(self) -> str:
        return 'VDW constraint'

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._constraint_id = ensemble.constraints.index(self)
        self._parameters_list = []
        for particle in self._parent_ensemble.topology.particles:
            param = self._parameter_dict[particle.particle_type]
            if len(param) == 2:
                epsilon, sigma = param
                self._parameters_list.append([epsilon, sigma, epsilon, sigma])
            elif len(param) == 4:
                epsilon, sigma, epsilon14, sigma14 = param
                self._parameters_list.append([epsilon, sigma, epsilon14, sigma14])
        self._device_parameters_list = cp.array(np.vstack(self._parameters_list), CUPY_FLOAT)

    @staticmethod
    def _update_charmm_vdw_kernel(
        positions,
        parameters,
        excluded_particles,
        scaled_particles,
        cutoff_radius,
        pbc_matrix,
        tile_list,
        tile_neighbors,
        forces, potential_energy
    ):
        # tile information
        thread_x = cuda.threadIdx.x
        thread_y = cuda.threadIdx.y
        tile_id1 = cuda.blockIdx.x
        tile_id2 = tile_neighbors[tile_id1, cuda.blockIdx.y]

        if tile_id1 >= tile_list.shape[0]:
            return
        if tile_id2 == -1:
            return
        # pbc matrix
        shared_pbc_matrix = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        shared_half_pbc_matrix = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        shared_forces = cuda.shared.array(shape=(NUM_PARTICLES_PER_TILE, SPATIAL_DIM), dtype=NUMBA_FLOAT)
        shared_potential_energy = cuda.shared.array(shape=(1), dtype=NUMBA_FLOAT)
        if thread_x == 0:
            for i in range(SPATIAL_DIM):
                if thread_y == 0:
                    shared_pbc_matrix[i] = pbc_matrix[i, i]
                    shared_half_pbc_matrix[i] = shared_pbc_matrix[i] / 2
                shared_forces[thread_y, i] = 0
        cuda.syncthreads()
        particle_1 = tile_list[tile_id1, thread_x]
        particle_2 = tile_list[tile_id2, thread_y]
        if particle_1 == -1:
            return
        if particle_2 == -1:
            return
        if particle_1 == particle_2:
            return
        for i in range(MAX_NUM_EXCLUDED_PARTICLES):
            particle = excluded_particles[particle_1, i]
            if particle == -1:
                break
            if particle == particle_2:
                return
        vec = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        r = 0
        for i in range(SPATIAL_DIM):
            vec[i] = positions[particle_2, i] - positions[particle_1, i]
            if vec[i] < - shared_half_pbc_matrix[i]:
                vec[i] += shared_pbc_matrix[i]
            elif vec[i] > shared_half_pbc_matrix[i]:
                vec[i] -= shared_pbc_matrix[i]
            r += vec[i]**2
        r = math.sqrt(r)
        if r <= cutoff_radius[0]:
            for i in range(SPATIAL_DIM):
                vec[i] /= r
            is_scaled = False
            for i in range(MAX_NUM_SCALED_PARTICLES):
                particle = scaled_particles[particle_1, i]
                if particle == -1:
                    break
                if particle == particle_2:
                    is_scaled = True
                    break
            if is_scaled:
                epsilon1, sigma1 = parameters[particle_1, 2:]
                epsilon2, sigma2 = parameters[particle_2, 2:]
            else:
                epsilon1, sigma1 = parameters[particle_1, :2]
                epsilon2, sigma2 = parameters[particle_2, :2]
            epsilon, sigma = (
                math.sqrt(epsilon1 * epsilon2),
                (sigma1 + sigma2) / 2
            )
            scaled_r = sigma / r
            scaled_r6 = scaled_r**6
            scaled_r12 = scaled_r6**2
            energy = 2 * epsilon * (scaled_r12 - scaled_r6)
            force_val = - (2 * scaled_r12 - scaled_r6) / r * epsilon * 24
            for i in range(SPATIAL_DIM):
                cuda.atomic.add(forces, (particle_1, i), force_val * vec[i])
            cuda.atomic.add(potential_energy, 0, energy)
            
        # cuda.syncthreads()
        # if thread_y == 0:
        #     for i in range(SPATIAL_DIM):
        #         cuda.atomic.add(forces, (particle_1, i), shared_forces[thread_x, i])
        #     if thread_x == 0:
        #         cuda.atomic.add(potential_energy, 0, shared_potential_energy[0])

        # # pbc matrix
        # shared_pbc_matrix = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        # shared_half_pbc_matrix = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        # shared_cutoff_radius = cuda.shared.array(shape=(1), dtype=NUMBA_FLOAT)
        # shared_potential_energy = cuda.shared.array(shape=(1), dtype=NUMBA_FLOAT)
        # # particle information
        # shared_particle_index_tile1 = cuda.shared.array(shape=(NUM_PARTICLES_PER_TILE), dtype=NUMBA_INT)
        # shared_positions_tile1 = cuda.shared.array(shape=(NUM_PARTICLES_PER_TILE, SPATIAL_DIM), dtype=NUMBA_FLOAT)
        # shared_parameters_tile1 = cuda.shared.array(shape=(NUM_PARTICLES_PER_TILE, SPATIAL_DIM), dtype=NUMBA_FLOAT)
        # shared_forces = cuda.shared.array(shape=(NUM_PARTICLES_PER_TILE, SPATIAL_DIM), dtype=NUMBA_FLOAT)
        # shared_excluded_particles_tile1 = cuda.shared.array(shape=(NUM_PARTICLES_PER_TILE, MAX_NUM_EXCLUDED_PARTICLES), dtype=NUMBA_INT)
        # shared_scaled_particles_tile1 = cuda.shared.array(shape=(NUM_PARTICLES_PER_TILE, MAX_NUM_SCALED_PARTICLES), dtype=NUMBA_INT)
        # shared_particle_index_tile2 = cuda.shared.array(shape=(NUM_PARTICLES_PER_TILE), dtype=NUMBA_INT)
        # shared_positions_tile2 = cuda.shared.array(shape=(NUM_PARTICLES_PER_TILE, SPATIAL_DIM), dtype=NUMBA_FLOAT)
        # shared_parameters_tile2 = cuda.shared.array(shape=(NUM_PARTICLES_PER_TILE, SPATIAL_DIM), dtype=NUMBA_FLOAT)

        # if thread_x == 0 and thread_y == 1:
        #     for i in range(SPATIAL_DIM):
        #         shared_pbc_matrix[i] = pbc_matrix[i, i]
        #         shared_half_pbc_matrix[i] = shared_pbc_matrix[i] / 2
        #     shared_cutoff_radius[0] = cutoff_radius[0]
        #     shared_potential_energy[0] = 0
        # if thread_x == thread_y:
        #     shared_particle_index_tile1[thread_x] = tile_list[tile_id1, thread_x]
        #     shared_particle_index_tile2[thread_x] = tile_list[tile_id2, thread_x]
        #     if shared_particle_index_tile1[thread_x] != -1:
        #         for i in range(SPATIAL_DIM):
        #             shared_positions_tile1[thread_x, i] = positions[shared_particle_index_tile1[thread_x], i]
        #             shared_forces[thread_x, i] = 0
        #         for i in range(4):
        #             shared_parameters_tile1[thread_x, i] = parameters[shared_particle_index_tile1[thread_x], i]
        #         is_terminated = False
        #         for i in range(MAX_NUM_EXCLUDED_PARTICLES):
        #             if not is_terminated:
        #                 particle_id = excluded_particles[shared_particle_index_tile1[thread_x], i]
        #                 shared_excluded_particles_tile1[thread_y, i] = particle_id
        #                 if particle_id == -1:
        #                     is_terminated = True
        #             else:
        #                 shared_excluded_particles_tile1[thread_y, i] = -1
        #         is_terminated = False
        #         for i in range(MAX_NUM_SCALED_PARTICLES):
        #             if not is_terminated:
        #                 particle_id = scaled_particles[shared_particle_index_tile1[thread_x], i]
        #                 shared_scaled_particles_tile1[thread_y, i] = particle_id
        #                 if particle_id == -1:
        #                     is_terminated = True
        #             else:
        #                 shared_scaled_particles_tile1[thread_y, i] = -1
        #     if shared_particle_index_tile2[thread_x] != -1:
        #         for i in range(SPATIAL_DIM):
        #             shared_positions_tile2[thread_y, i] = positions[shared_particle_index_tile2[thread_x], i]
        #         for i in range(4):
        #             shared_parameters_tile2[thread_y, i] = parameters[shared_particle_index_tile2[thread_x], i]
        # cuda.syncthreads()
        # # Calculate
        # is_excluded = False
        # if shared_particle_index_tile1[thread_x] == -1:
        #     is_excluded = True
        # if shared_particle_index_tile2[thread_y] == -1:
        #     is_excluded = True
        # if shared_particle_index_tile2[thread_y] == shared_particle_index_tile1[thread_x]:
        #     is_excluded = True
        # if not is_excluded:
        #     for i in shared_scaled_particles_tile1[thread_x, :]:
        #         if i == -1:
        #             break
        #         if i == shared_particle_index_tile2[thread_x]:
        #             is_excluded = True
        #             break
        # if not is_excluded:
        #     vec = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        #     r = 0
        #     for i in range(SPATIAL_DIM):
        #         vec[i] = shared_positions_tile2[thread_y, i] - shared_positions_tile1[thread_x, i]
        #         if vec[i] < - shared_half_pbc_matrix[i]:
        #             vec[i] += shared_pbc_matrix[i]
        #         elif vec[i] > shared_half_pbc_matrix[i]:
        #             vec[i] -= shared_pbc_matrix[i]
        #         r += vec[i]**2
        #     r = math.sqrt(r)
        #     if r <= shared_cutoff_radius[0]:
        #         is_scaled = False
        #         for i in shared_excluded_particles_tile1[thread_x, :]:
        #             if i == -1:
        #                 break
        #             if i == shared_particle_index_tile2[thread_y]:
        #                 is_scaled = True
        #                 break
        #         if is_scaled:
        #             epsilon1, sigma1 = shared_parameters_tile1[thread_x, 2:]
        #             epsilon2, sigma2 = shared_parameters_tile2[thread_y, 2:]
        #         else:
        #             epsilon1, sigma1 = shared_parameters_tile1[thread_x, :2]
        #             epsilon2, sigma2 = shared_parameters_tile2[thread_y, :2]
        #         epsilon, sigma = (
        #             math.sqrt(epsilon1 * epsilon2),
        #             (sigma1 + sigma2) / 2
        #         )
        #         scaled_r = sigma / r
        #         scaled_r6 = scaled_r**6
        #         scaled_r12 = scaled_r6**2
        #         energy = 2 * epsilon * (scaled_r12 - scaled_r6)
        #         force_val = - (2 * scaled_r12 - scaled_r6) / r * epsilon * 24
        #         for i in range(SPATIAL_DIM):
        #             cuda.atomic.add(shared_forces, (thread_x, i), force_val * vec[i] / r)
        #         cuda.atomic.add(shared_potential_energy, 0, energy)
        # cuda.syncthreads()
        # if thread_x == 0:
        #     for i in range(SPATIAL_DIM):
        #         cuda.atomic.add(forces, (shared_particle_index_tile1[thread_y], i), shared_forces[thread_y, i])
        #     if thread_y == 0:
        #         cuda.atomic.add(potential_energy, 0, shared_potential_energy[0])

    def update(self):
        self._check_bound_state()
        self._forces = cp.zeros(self._parent_ensemble.state.matrix_shape, CUPY_FLOAT)
        self._potential_energy = cp.zeros([1], CUPY_FLOAT)
        block_per_grid_x = int(np.ceil(
            self._parent_ensemble.state.tile_list.num_tiles * NUM_PARTICLES_PER_TILE / THREAD_PER_BLOCK[0]
        ))
        block_per_grid_y = int(np.ceil(
            self._parent_ensemble.state.tile_list.tile_neighbors.shape[1] * NUM_PARTICLES_PER_TILE / THREAD_PER_BLOCK[1]
        ))
        self._block_per_grid = (block_per_grid_x, block_per_grid_y)
        # Device
        self._update[self._block_per_grid, THREAD_PER_BLOCK](
            self._parent_ensemble.state.positions,
            self._device_parameters_list,
            self._parent_ensemble.topology.device_excluded_particles,
            self._parent_ensemble.topology.device_scaled_particles,
            self._device_cutoff_radius,
            self._parent_ensemble.state.device_pbc_matrix,
            self._parent_ensemble.state.tile_list.tile_list,
            self._parent_ensemble.state.tile_list.tile_neighbors,
            self._forces, self._potential_energy
        )

if __name__ == '__main__':
    import os
    import mdpy as md
    data_dir = '/home/zhenyuwei/nutstore/ZhenyuWei/Note_Research/mdpy/mdpy/benchmark/data'
    psf_file = os.path.join(data_dir, 'str.psf')
    pdb_file = os.path.join(data_dir, 'str.pdb')
    cutoff_radius = Quantity(9, angstrom)
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

    ensemble.state.set_positions(pdb.positions)
    constraint.update()
    print(constraint.forces)