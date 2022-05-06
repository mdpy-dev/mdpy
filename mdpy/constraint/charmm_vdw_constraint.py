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

THREAD_PER_BLOCK = (32, 1)
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
            NUMBA_FLOAT[::1], # cutoff_radius
            NUMBA_FLOAT[:, ::1], # pbc_matrix
            NUMBA_INT[:, ::1], # excluded_particles
            NUMBA_INT[:, ::1], # scaled_particles
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
        cutoff_radius,
        pbc_matrix,
        excluded_particles,
        scaled_particles,
        tile_list,
        tile_neighbors,
        forces, potential_energy
    ):
        # tile information
        tile_id1 = cuda.blockIdx.x
        tile_id2 = tile_neighbors[tile_id1, cuda.blockIdx.y]
        thread_x = cuda.threadIdx.x
        if tile_id1 >= tile_list.shape[0]:
            return
        if tile_id2 == -1:
            return
        # shared data
        shared_particle_index = cuda.shared.array(shape=(NUM_PARTICLES_PER_TILE), dtype=NUMBA_INT)
        shared_positions = cuda.shared.array(shape=(NUM_PARTICLES_PER_TILE, SPATIAL_DIM), dtype=NUMBA_FLOAT)
        shared_parameters = cuda.shared.array(shape=(NUM_PARTICLES_PER_TILE, 4), dtype=NUMBA_FLOAT)
        shared_pbc_matrix = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        shared_half_pbc_matrix = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        if thread_x == 0:
            for i in range(SPATIAL_DIM):
                shared_pbc_matrix[i] = pbc_matrix[i, i]
                shared_half_pbc_matrix[i] = shared_pbc_matrix[i] / 2
        shared_particle_index[thread_x] = tile_list[tile_id2, thread_x]
        for i in range(SPATIAL_DIM):
            shared_positions[thread_x, i] = positions[shared_particle_index[thread_x], i]
        for i in range(4):
            shared_parameters[thread_x, i] = parameters[shared_particle_index[thread_x], i]
        cuda.syncthreads()

        # Local data
        particle_1 = tile_list[tile_id1, thread_x]
        if particle_1 == -1:
            return
        local_excluded_particles = cuda.local.array(shape=(MAX_NUM_EXCLUDED_PARTICLES), dtype=NUMBA_INT)
        num_excluded_particles = 0
        for i in range(MAX_NUM_EXCLUDED_PARTICLES):
            local_excluded_particles[i] = excluded_particles[particle_1, i]
            if local_excluded_particles[i] == -1:
                break
            num_excluded_particles += 1
        local_scaled_particles = cuda.local.array(shape=(MAX_NUM_SCALED_PARTICLES), dtype=NUMBA_INT)
        num_scaled_particles = 0
        for i in range(MAX_NUM_SCALED_PARTICLES):
            local_scaled_particles[i] = scaled_particles[particle_1, i]
            if local_scaled_particles[i] == -1:
                break
            num_scaled_particles += 1
        local_parameters = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        for i in range(4):
            local_parameters[i] = parameters[particle_1, i]
        local_positions = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        local_forces = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        vec = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        energy = 0
        cutoff_radius = cutoff_radius[0]
        for i in range(SPATIAL_DIM):
            local_positions[i] = positions[particle_1, i]
            local_forces[i] = 0

        # Computation
        for index in range(NUM_PARTICLES_PER_TILE):
            particle_2 = shared_particle_index[index]
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

            r = 0
            for i in range(SPATIAL_DIM):
                vec[i] = shared_positions[index, i] - local_positions[i]
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
                    epsilon, sigma = (
                        math.sqrt(local_parameters[0] * shared_parameters[index, 0]),
                        (local_parameters[1] + shared_parameters[index, 1]) / 2
                    )
                else:
                    epsilon, sigma = (
                        math.sqrt(local_parameters[2] * shared_parameters[index, 2]),
                        (local_parameters[3] + shared_parameters[index, 3]) / 2
                    )
                scaled_r = sigma / r
                scaled_r6 = scaled_r**6
                scaled_r12 = scaled_r6**2
                energy += 2 * epsilon * (scaled_r12 - scaled_r6)
                force_val = - (2 * scaled_r12 - scaled_r6) / r * epsilon * 24
                for i in range(SPATIAL_DIM):
                    local_forces[i] += force_val * vec[i]
        for i in range(SPATIAL_DIM):
            cuda.atomic.add(forces, (particle_1, i), local_forces[i])
        cuda.atomic.add(potential_energy, 0, energy)

    def update(self):
        self._check_bound_state()
        self._forces = cp.zeros(self._parent_ensemble.state.matrix_shape, CUPY_FLOAT)
        self._potential_energy = cp.zeros([1], CUPY_FLOAT)
        block_per_grid_x = int(np.ceil(
            self._parent_ensemble.state.tile_list.num_tiles * NUM_PARTICLES_PER_TILE / THREAD_PER_BLOCK[0]
        ))
        block_per_grid_y = int(np.ceil(
            self._parent_ensemble.state.tile_list.tile_neighbors.shape[1]
        ))
        self._block_per_grid = (block_per_grid_x, block_per_grid_y)
        # Device
        self._update[self._block_per_grid, THREAD_PER_BLOCK](
            self._parent_ensemble.state.positions,
            self._device_parameters_list,
            self._device_cutoff_radius,
            self._parent_ensemble.state.device_pbc_matrix,
            self._parent_ensemble.topology.device_excluded_particles,
            self._parent_ensemble.topology.device_scaled_particles,
            self._parent_ensemble.state.tile_list.tile_list,
            self._parent_ensemble.state.tile_list.tile_neighbors,
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
    print(ensemble.state.tile_list.tile_neighbors.shape)
    constraint.update()
    # print(constraint.forces)
    print(Quantity(constraint.forces, default_force_unit).convert_to(kilojoule_permol_over_nanometer).value)