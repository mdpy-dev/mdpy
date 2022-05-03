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
from mdpy.environment import *
from mdpy.core import Ensemble
from mdpy.core import MAX_NUM_EXCLUDED_PARTICLES, MAX_NUM_SCALED_PARTICLES
from mdpy.constraint import Constraint
from mdpy.utils import *
from mdpy.unit import *

THREAD_PER_BLOCK = (64)
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
            NUMBA_FLOAT[::1], # cutoff_radius
            NUMBA_FLOAT[:, ::1], # parameters
            NUMBA_INT[:, ::1], # excluded_particle
            NUMBA_INT[:, ::1], # scaled_particles
            NUMBA_INT[:, ::1], # neighbor_list
            NUMBA_FLOAT[:, :, ::1], # neighbor_vec_list
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
        self._block_per_grid = int(np.ceil(
            self._parent_ensemble.topology.num_particles / THREAD_PER_BLOCK
        ))

    @staticmethod
    def _update_charmm_vdw_kernel(
        cutoff_radius,
        parameters,
        excluded_particles,
        scaled_particles,
        neighbor_list,
        neighbor_vec_list,
        forces, potential_energy
    ):
        particle_id1 = cuda.grid(1)
        num_particles = neighbor_list.shape[0]
        if particle_id1 >= num_particles:
            return None
        # Bonded particle
        local_excluded_particles = cuda.local.array(
            shape=(MAX_NUM_EXCLUDED_PARTICLES), dtype=NUMBA_INT
        )
        for i in range(MAX_NUM_EXCLUDED_PARTICLES):
            local_excluded_particles[i] = excluded_particles[particle_id1, i]
        # Scaling particles
        local_scaled_particles = cuda.local.array(
            shape=(MAX_NUM_SCALED_PARTICLES), dtype=NUMBA_INT
        )
        for i in range(MAX_NUM_SCALED_PARTICLES):
            local_scaled_particles[i] = scaled_particles[particle_id1, i]
        # Parameters
        local_parameters = cuda.local.array(shape=(4), dtype=NUMBA_FLOAT)
        local_parameters[0] = parameters[particle_id1, 0]
        local_parameters[1] = parameters[particle_id1, 1]
        local_parameters[2] = parameters[particle_id1, 2]
        local_parameters[3] = parameters[particle_id1, 3]
        # cutoff
        cutoff_radius = cutoff_radius[0]
        force_x = 0
        force_y = 0
        force_z = 0
        energy = 0
        for neighbor_index in range(neighbor_list.shape[1]):
            particle_id2 = neighbor_list[particle_id1, neighbor_index]
            if particle_id2 == -1: # self-padding term
                break
            if particle_id1 == particle_id2: # self-self term
                continue
            is_bonded = False
            for i in local_excluded_particles:
                if i == -1: # padding of bonded particle
                    break
                elif particle_id2 == i: # self-bonded particle term
                    is_bonded = True
                    break
            if is_bonded:
                continue
            r = neighbor_vec_list[particle_id1, neighbor_index, 0]
            if r <= cutoff_radius:
                scaled_x = neighbor_vec_list[particle_id1, neighbor_index, 1]
                scaled_y = neighbor_vec_list[particle_id1, neighbor_index, 2]
                scaled_z = neighbor_vec_list[particle_id1, neighbor_index, 3]
                is_scaled = False
                for i in local_scaled_particles:
                    if i == -1:
                        break
                    if particle_id2 == i:
                        is_scaled = True
                        break
                if not is_scaled:
                    epsilon1, sigma1 = local_parameters[:2]
                    epsilon2, sigma2 = parameters[particle_id2, :2]
                else:
                    epsilon1, sigma1 = local_parameters[2:]
                    epsilon2, sigma2 = parameters[particle_id2, 2:]
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
        cuda.atomic.add(forces, (particle_id1, 0), force_x)
        cuda.atomic.add(forces, (particle_id1, 1), force_y)
        cuda.atomic.add(forces, (particle_id1, 2), force_z)
        cuda.atomic.add(potential_energy, 0, energy)

    def update(self):
        self._check_bound_state()
        self._forces = cp.zeros(self._parent_ensemble.state.matrix_shape, CUPY_FLOAT)
        self._potential_energy = cp.zeros([1], CUPY_FLOAT)
        # Device
        self._update[self._block_per_grid, THREAD_PER_BLOCK](
            self._device_cutoff_radius,
            self._device_parameters_list,
            self._parent_ensemble.topology.device_excluded_particles,
            self._parent_ensemble.topology.device_scaled_particles,
            self._parent_ensemble.state.neighbor_list.neighbor_list,
            self._parent_ensemble.state.neighbor_list.neighbor_vec_list,
            self._forces, self._potential_energy
        )
