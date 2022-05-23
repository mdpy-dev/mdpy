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
from mdpy import SPATIAL_DIM
from mdpy.environment import *
from mdpy.core import NUM_PARTICLES_PER_TILE, Ensemble
from mdpy.constraint import Constraint
from mdpy.utils import *
from mdpy.unit import *

THREAD_PER_BLOCK = (NUM_PARTICLES_PER_TILE, 1)
NUM_TILES_PER_THREAD = 8
class CharmmVDWConstraint(Constraint):
    def __init__(self, parameter_dict: dict, cutoff_radius=Quantity(12, angstrom)) -> None:
        super().__init__()
        # Attributes
        self._parameter_dict = parameter_dict
        self._cutoff_radius = check_quantity_value(cutoff_radius, default_length_unit)
        self._device_cutoff_radius = cp.array([self._cutoff_radius], CUPY_FLOAT)
        self._parameters = []
        # Kernel
        self._update_charmm_vdw = cuda.jit(nb.void(
            NUMBA_FLOAT[::1], # cutoff_radius
            NUMBA_FLOAT[:, ::1], # pbc_matrix
            NUMBA_FLOAT[:, ::1], # sorted_positions
            NUMBA_FLOAT[:, ::1], # sorted_parameters
            NUMBA_BIT[:, ::1], # exclusion_map
            NUMBA_INT[:, ::1], # tile_neighbors
            NUMBA_FLOAT[:, ::1], # sorted_forces
            NUMBA_FLOAT[::1] # potential_energy
        ))(self._update_charmm_vdw_kernel)
        self._update_scaled_charmm_vdw = cuda.jit(nb.void(
            NUMBA_FLOAT[:, ::1], # positions
            NUMBA_FLOAT[:, ::1], # parameters
            NUMBA_FLOAT[:, ::1], # pbc_matrix
            NUMBA_FLOAT[::1], # cutoff_radius
            NUMBA_INT[:, ::1], # scaled_particles
            NUMBA_FLOAT[:, ::1], # positions
            NUMBA_FLOAT[::1] # potential_energy
        ))(self._update_scaled_charmm_vdw_kernel)

    def __repr__(self) -> str:
        return '<mdpy.constraint.CharmmVDWConstraint object>'

    def __str__(self) -> str:
        return 'VDW constraint'

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._constraint_id = ensemble.constraints.index(self)
        self._parameters = []
        for particle in self._parent_ensemble.topology.particles:
            parameter = self._parameter_dict[particle.particle_type]
            if len(parameter) == 2:
                epsilon, sigma = parameter
                self._parameters.append([epsilon, sigma, epsilon, sigma])
            elif len(parameter) == 4:
                epsilon, sigma, epsilon14, sigma14 = parameter
                self._parameters.append([epsilon, sigma, epsilon14, sigma14])
        self._device_parameters = cp.array(np.vstack(self._parameters), CUPY_FLOAT)

    @staticmethod
    def _update_charmm_vdw_kernel(
        cutoff_radius,
        pbc_matrix,
        sorted_positions,
        sorted_parameters,
        exclusion_map,
        tile_neighbors,
        sorted_forces, potential_energy
    ):
        # tile information
        tile_id1 = cuda.blockIdx.x
        tile_id2_start = cuda.blockIdx.y * NUM_TILES_PER_THREAD
        tile_id2_end = tile_id2_start + NUM_TILES_PER_THREAD
        if tile_id2_end >= tile_neighbors.shape[1]:
            tile_id2_end = tile_neighbors.shape[1]
        # Particle index information
        local_thread_x = cuda.threadIdx.x
        global_thread_x = local_thread_x + cuda.blockIdx.x * cuda.blockDim.x
        # shared data
        local_pbc_matrix = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        local_half_pbc_matrix = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        for i in range(SPATIAL_DIM):
            local_pbc_matrix[i] = pbc_matrix[i, i]
            local_half_pbc_matrix[i] = local_pbc_matrix[i] * NUMBA_FLOAT(0.5)
        tile1_positions = cuda.shared.array(shape=(SPATIAL_DIM, NUM_PARTICLES_PER_TILE), dtype=NUMBA_FLOAT)
        tile1_parameters = cuda.shared.array(shape=(2, NUM_PARTICLES_PER_TILE), dtype=NUMBA_FLOAT)
        tile2_positions = cuda.shared.array(shape=(SPATIAL_DIM, NUM_PARTICLES_PER_TILE), dtype=NUMBA_FLOAT)
        tile2_parameters = cuda.shared.array(shape=(2, NUM_PARTICLES_PER_TILE), dtype=NUMBA_FLOAT)
        tile1_index = tile_id1 * NUM_PARTICLES_PER_TILE + local_thread_x
        cuda.syncthreads()
        # Read data
        for i in range(SPATIAL_DIM):
            tile1_positions[i, local_thread_x] = sorted_positions[i, tile1_index]
        for i in range(2):
            tile1_parameters[i, local_thread_x] = sorted_parameters[i, tile1_index]

        # Local data
        local_forces = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        vec = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        energy = NUMBA_FLOAT(0)
        cutoff_radius = cutoff_radius[0]
        for i in range(SPATIAL_DIM):
            local_forces[i] = 0
        for tile_index in range(tile_id2_start, tile_id2_end):
            tile_id2 = tile_neighbors[tile_id1, tile_index]
            if tile_id2 == -1:
                break
            tile2_index = tile_id2 * NUM_PARTICLES_PER_TILE + local_thread_x
            cuda.syncthreads()
            for i in range(SPATIAL_DIM):
                tile2_positions[i, local_thread_x] = sorted_positions[i, tile2_index]
            for i in range(2):
                tile2_parameters[i, local_thread_x] = sorted_parameters[i, tile2_index]
            exclusion_flag = exclusion_map[tile_index, tile1_index]
            cuda.syncthreads()

            # Computation
            for index in range(NUM_PARTICLES_PER_TILE):
                if exclusion_flag >> index & 0b1:
                    continue
                r = NUMBA_FLOAT(0)
                for i in range(SPATIAL_DIM):
                    vec[i] = tile2_positions[i, index] - tile1_positions[i, local_thread_x]
                    if vec[i] < - local_half_pbc_matrix[i]:
                        vec[i] += local_pbc_matrix[i]
                    elif vec[i] > local_half_pbc_matrix[i]:
                        vec[i] -= local_pbc_matrix[i]
                    r += vec[i]**2
                r = math.sqrt(r)
                if r < cutoff_radius:
                    inverse_r = NUMBA_FLOAT(1) / r
                    inverse_r_square = inverse_r**2
                    epsilon = math.sqrt(tile2_parameters[0, index] * tile1_parameters[0, local_thread_x])
                    sigma = (tile2_parameters[1, index] + tile1_parameters[1, local_thread_x]) * NUMBA_FLOAT(0.5)
                    scaled_r = sigma * inverse_r
                    scaled_r6 = scaled_r**6
                    scaled_r12 = scaled_r6**2
                    energy += NUMBA_FLOAT(2) * epsilon * (scaled_r12 - scaled_r6)
                    force_val = - (NUMBA_FLOAT(2) * scaled_r12 - scaled_r6) * inverse_r_square * epsilon * NUMBA_FLOAT(24)
                    for i in range(SPATIAL_DIM):
                        local_forces[i] += force_val * vec[i]

        for i in range(SPATIAL_DIM):
            cuda.atomic.add(sorted_forces, (i, global_thread_x), local_forces[i])
        cuda.atomic.add(potential_energy, 0, energy)

    @staticmethod
    def _update_scaled_charmm_vdw_kernel(
        positions,
        parameters,
        pbc_matrix,
        cutoff_radius,
        scaled_particles,
        forces, potential_energy
    ):
        particle1 = cuda.grid(1)
        local_thread_x = cuda.threadIdx.x
        if particle1 >= positions.shape[0]:
            return

        shared_pbc_matrix = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        shared_half_pbc_matrix = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        if local_thread_x <= 2:
            shared_pbc_matrix[local_thread_x] = pbc_matrix[local_thread_x, local_thread_x]
            shared_half_pbc_matrix[local_thread_x] = shared_pbc_matrix[local_thread_x] * NUMBA_FLOAT(0.5)

        local_parameters = cuda.local.array(shape=(4), dtype=NUMBA_FLOAT)
        local_positions = cuda.local.array(shape=(3), dtype=NUMBA_FLOAT)
        local_forces = cuda.local.array(shape=(3), dtype=NUMBA_FLOAT)
        vec = cuda.local.array(shape=(3), dtype=NUMBA_FLOAT)
        for i in range(SPATIAL_DIM):
            local_positions[i] = positions[particle1, i]
            local_forces[i] = 0
            local_parameters[i] = parameters[particle1, i]
        local_parameters[3] = parameters[particle1, 3]
        cutoff_radius = cutoff_radius[0]
        energy = NUMBA_FLOAT(0)
        is_scaled = False
        for i in range(scaled_particles.shape[1]):
            particle2 = scaled_particles[particle1, i]
            if particle2 == -1:
                break
            is_scaled = True
            r = NUMBA_FLOAT(0)
            for i in range(SPATIAL_DIM):
                vec[i] = positions[particle2, i] - local_positions[i]
                if vec[i] < - shared_half_pbc_matrix[i]:
                    vec[i] += shared_pbc_matrix[i]
                elif vec[i] > shared_half_pbc_matrix[i]:
                    vec[i] -= shared_pbc_matrix[i]
                r += vec[i]**2
            r = math.sqrt(r)
            if r < cutoff_radius:
                inverse_r = NUMBA_FLOAT(1) / r
                epsilon1 = math.sqrt(local_parameters[0] * parameters[particle2, 0])
                sigma1 = (local_parameters[1] + parameters[particle2, 1]) * NUMBA_FLOAT(0.5)
                scaled_r = sigma1 * inverse_r
                scaled_r6 = scaled_r**6
                scaled_r12 = scaled_r6**2
                energy -= NUMBA_FLOAT(2) * epsilon1 * (scaled_r12 - scaled_r6)
                force_val = (NUMBA_FLOAT(2) * scaled_r12 - scaled_r6) * inverse_r * epsilon1 * NUMBA_FLOAT(24)
                epsilon2 = math.sqrt(local_parameters[2] * parameters[particle2, 2])
                sigma2 = (local_parameters[3] + parameters[particle2, 3]) * NUMBA_FLOAT(0.5)
                scaled_r = sigma2 * inverse_r
                scaled_r6 = scaled_r**6
                scaled_r12 = scaled_r6**2
                energy += NUMBA_FLOAT(2) * epsilon2 * (scaled_r12 - scaled_r6)
                force_val -= (NUMBA_FLOAT(2) * scaled_r12 - scaled_r6) * inverse_r * epsilon2 * NUMBA_FLOAT(24)
                for i in range(SPATIAL_DIM):
                    local_forces[i] += force_val * vec[i] * inverse_r
        if is_scaled:
            for i in range(SPATIAL_DIM):
                cuda.atomic.add(forces, (particle1, i), local_forces[i])
            cuda.atomic.add(potential_energy, 0, energy)

    def update(self):
        self._check_bound_state()
        sorted_forces = cp.zeros((SPATIAL_DIM, self._parent_ensemble.tile_list.num_tiles * NUM_PARTICLES_PER_TILE), CUPY_FLOAT)
        self._potential_energy = cp.zeros([1], CUPY_FLOAT)
        sorted_positions = self._parent_ensemble.tile_list.sort_matrix(self._parent_ensemble.state.positions)
        device_sorted_parameter_list = self._parent_ensemble.tile_list.sort_matrix(self._device_parameters)
        # update
        block_per_grid_x = self._parent_ensemble.tile_list.num_tiles
        block_per_grid_y = int(np.ceil(
            self._parent_ensemble.tile_list.tile_neighbors.shape[1] / NUM_TILES_PER_THREAD
        ))
        block_per_grid = (block_per_grid_x, block_per_grid_y)
        self._update_charmm_vdw[block_per_grid, THREAD_PER_BLOCK](
            self._device_cutoff_radius,
            self._parent_ensemble.state.device_pbc_matrix,
            sorted_positions,
            device_sorted_parameter_list,
            self._parent_ensemble.topology.device_exclusion_map,
            self._parent_ensemble.tile_list.tile_neighbors,
            sorted_forces, self._potential_energy
        )
        self._forces = self._parent_ensemble.tile_list.unsort_matrix(sorted_forces)

        thread_per_block = 64
        block_per_grid = int(np.ceil(
            self._parent_ensemble.topology.num_particles / thread_per_block
        ))
        self._update_scaled_charmm_vdw[block_per_grid, thread_per_block](
            self._parent_ensemble.state.positions,
            self._device_parameters,
            self._parent_ensemble.tile_list._device_pbc_matrix,
            self._device_cutoff_radius,
            self._parent_ensemble.topology.device_scaled_particles,
            self._forces, self._potential_energy
        )


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
    print(Quantity(constraint.forces, default_force_unit).convert_to(kilojoule_permol_over_nanometer).value[-20:, :])