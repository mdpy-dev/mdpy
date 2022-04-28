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
from mdpy.environment import *
from mdpy.core import Ensemble
from mdpy.core import MAX_NUM_BONDED_PARTICLES
from mdpy.constraint import Constraint
from mdpy.utils import *
from mdpy.unit import *

THREAD_PER_BLOCK = (64)

class ElectrostaticCutoffConstraint(Constraint):
    def __init__(self, cutoff_radius=Quantity(12, angstrom)) -> None:
        super().__init__()
        # Attributes
        self._cutoff_radius = check_quantity_value(cutoff_radius, default_length_unit)
        self._device_cutoff_radius = cp.array([self._cutoff_radius], CUPY_FLOAT)
        self._k = cp.array([4 * np.pi * EPSILON0.value], CUPY_FLOAT)
        # Kernel
        self._update = cuda.jit(nb.void(
            NUMBA_FLOAT[:, ::1], # charges
            NUMBA_FLOAT[::1], # k
            NUMBA_FLOAT[::1], # cutoff_radius
            NUMBA_INT[:, ::1], # bonded_particles
            NUMBA_INT[:, ::1], # neighbor_list
            NUMBA_FLOAT[:, :, ::1], # neighbor_vec_list
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
        self._charges = cp.array(self._parent_ensemble.topology.charges, CUPY_FLOAT)
        self._bonded_particles = cp.array(self._parent_ensemble.topology.bonded_particles, CUPY_INT)
        self._block_per_grid = int(np.ceil(
            self._parent_ensemble.topology.num_particles / THREAD_PER_BLOCK
        ))

    @staticmethod
    def _update_kernel(
        charges,
        k, cutoff_radius,
        bonded_particles,
        neighbor_list,
        neighbor_vec_list,
        forces, potential_energy
    ):
        particle_id1 = cuda.grid(1)
        num_particles = neighbor_list.shape[0]
        if particle_id1 >= num_particles:
            return None
        # Bonded particle
        local_bonded_particles = cuda.local.array(
            shape=(MAX_NUM_BONDED_PARTICLES), dtype=NUMBA_INT
        )
        for i in range(MAX_NUM_BONDED_PARTICLES):
            local_bonded_particles[i] = bonded_particles[particle_id1, i]

        shared_k = cuda.shared.array(shape=(1), dtype=NUMBA_FLOAT)
        shared_cutoff_radius = cuda.shared.array(shape=(1), dtype=NUMBA_FLOAT)
        thread_x = cuda.threadIdx.x
        if thread_x == 0:
            shared_k[0] = k[0]
        elif thread_x == 1:
            shared_cutoff_radius[0] = cutoff_radius[0]
        cuda.syncthreads()

        # cutoff
        force_x = 0
        force_y = 0
        force_z = 0
        energy = 0
        e1 = charges[particle_id1, 0]
        for neighbor_index in range(neighbor_list.shape[1]):
            particle_id2 = neighbor_list[particle_id1, neighbor_index]
            if particle_id2 == -1: # self-padding term
                break
            if particle_id1 == particle_id2:
                continue
            if particle_id1 == particle_id2: # self-self term
                continue
            is_bonded = False
            for i in local_bonded_particles:
                if i == -1: # padding of bonded particle
                    break
                elif particle_id2 == i: # self-bonded particle term
                    is_bonded = True
                    break
            if is_bonded:
                continue
            r = neighbor_vec_list[particle_id1, neighbor_index, 0]
            if r <= shared_cutoff_radius[0]:
                scaled_x = neighbor_vec_list[particle_id1, neighbor_index, 1]
                scaled_y = neighbor_vec_list[particle_id1, neighbor_index, 2]
                scaled_z = neighbor_vec_list[particle_id1, neighbor_index, 3]
                e1e2 = e1 * charges[particle_id2, 0]
                force_val = - e1e2 / shared_k[0] / r**2
                force_x += scaled_x * force_val
                force_y += scaled_y * force_val
                force_z += scaled_z * force_val
                # Energy still need divided by 2
                energy += e1e2 / shared_k[0] / r / 2
        cuda.atomic.add(forces, (particle_id1, 0), force_x)
        cuda.atomic.add(forces, (particle_id1, 1), force_y)
        cuda.atomic.add(forces, (particle_id1, 2), force_z)
        cuda.atomic.add(potential_energy, 0, energy)

    def update(self):
        self._check_bound_state()
        self._forces = cp.zeros_like(self._parent_ensemble.state.positions, CUPY_FLOAT)
        self._potential_energy = cp.zeros([1], CUPY_FLOAT)
        # Update
        self._update[self._block_per_grid, THREAD_PER_BLOCK](
            self._charges, self._k,
            self._device_cutoff_radius,
            self._bonded_particles,
            self._parent_ensemble.state.neighbor_list.device_neighbor_list,
            self._parent_ensemble.state.neighbor_list.device_neighbor_vec_list,
            self._forces, self._potential_energy
        )