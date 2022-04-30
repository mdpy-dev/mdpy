#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : charmm_angle_constraint.py
created time : 2021/10/10
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import math
from re import L
import numpy as np
import numba as nb
import cupy as cp
import numba.cuda as cuda
from mdpy import SPATIAL_DIM
from mdpy.environment import *
from mdpy.core import Ensemble
from mdpy.constraint import Constraint
from mdpy.utils import *

THREAD_PER_BLOCK = (32)

class CharmmAngleConstraint(Constraint):
    def __init__(self, parameter_dict: dict) -> None:
        super().__init__()
        self._parameter_dict = parameter_dict
        self._int_parameters = []
        self._float_parameters = []
        self._num_angles = 0
        self._update = cuda.jit(nb.void(
            NUMBA_INT[:, ::1], # int_parameters
            NUMBA_FLOAT[:, ::1], # float_parameters
            NUMBA_FLOAT[:, ::1], # positions
            NUMBA_FLOAT[:, ::1], # pbc_matrix
            NUMBA_FLOAT[:, ::1], # forces
            NUMBA_FLOAT[::1] # potential_energy
        ))(self._update_kernel)

    def __repr__(self) -> str:
        return '<mdpy.constraint.CharmmAngleConstraint object>'

    def __str__(self) -> str:
        return 'Angle constraint'

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._constraint_id = ensemble.constraints.index(self)
        self._int_parameters = []
        self._float_parameters = []
        self._num_angles = 0
        for angle in self._parent_ensemble.topology.angles:
            angle_type = '%s-%s-%s' %(
                self._parent_ensemble.topology.particles[angle[0]].particle_type,
                self._parent_ensemble.topology.particles[angle[1]].particle_type,
                self._parent_ensemble.topology.particles[angle[2]].particle_type
            )
            # matrix_id of 3 particles which form the angle
            self._int_parameters.append([
                self._parent_ensemble.topology.particles[angle[0]].matrix_id,
                self._parent_ensemble.topology.particles[angle[1]].matrix_id,
                self._parent_ensemble.topology.particles[angle[2]].matrix_id
            ])
            # Angle parameters
            self._float_parameters.append(self._parameter_dict[angle_type])
            self._num_angles += 1
        self._device_int_parameters = cp.array(np.vstack(self._int_parameters), CUPY_INT)
        self._device_float_parameters = cp.array(np.vstack(self._float_parameters), CUPY_FLOAT)
        self._block_per_grid = (int(np.ceil(
            self._parent_ensemble.topology.num_angles / THREAD_PER_BLOCK
        )))

    @staticmethod
    def _update_kernel(
        int_parameters, float_parameters,
        positions, pbc_matrix,
        forces, potential_energy
    ):
        angle_id = cuda.grid(1)
        shared_num_angles = cuda.shared.array(shape=(1), dtype=NUMBA_INT)
        shared_pbc = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        shared_half_pbc = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        if cuda.threadIdx.x == 0:
            shared_num_angles[0] = int_parameters.shape[0]
        if cuda.threadIdx.x == 1:
            shared_pbc[0] = pbc_matrix[0, 0]
            shared_pbc[1] = pbc_matrix[1, 1]
            shared_pbc[2] = pbc_matrix[2, 2]
            shared_half_pbc[0] = shared_pbc[0] / 2
            shared_half_pbc[1] = shared_pbc[1] / 2
            shared_half_pbc[2] = shared_pbc[2] / 2
        cuda.syncthreads()
        if angle_id >= shared_num_angles[0]:
            return
        id1, id2, id3 = int_parameters[angle_id, :]
        k, theta0, ku, u0 = float_parameters[angle_id, :]
        # Initialization
        particle_ids = cuda.local.array(shape=(4), dtype=NUMBA_INT)
        local_positions = cuda.local.array(shape=(3, SPATIAL_DIM), dtype=NUMBA_FLOAT)
        local_forces = cuda.local.array(shape=(3, SPATIAL_DIM), dtype=NUMBA_FLOAT)
        for i in range(3):
            particle_ids[i] = int_parameters[angle_id, i]
            for j in range(SPATIAL_DIM):
                local_positions[i, j] = positions[particle_ids[i], j]
        energy = 0
        # Positions
        # vec
        # r21
        v21 = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        v23 = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        v13 = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        r21, r23, r13 = 0, 0, 0
        for i in range(SPATIAL_DIM):
            v21[i] = local_positions[0, i] - local_positions[1, i]
            if v21[i] >= shared_half_pbc[i]:
                v21[i] -= shared_pbc[i]
            elif v21[i] <= -shared_half_pbc[i]:
                v21[i] += shared_pbc[i]
            r21 += v21[i]**2
            v23[i] = local_positions[2, i] - local_positions[1, i]
            if v23[i] >= shared_half_pbc[i]:
                v23[i] -= shared_pbc[i]
            elif v23[i] <= -shared_half_pbc[i]:
                v23[i] += shared_pbc[i]
            r23 += v23[i]**2
            v13[i] = local_positions[2, i] - local_positions[0, i]
            if v13[i] >= shared_half_pbc[i]:
                v13[i] -= shared_pbc[i]
            elif v13[i] <= -shared_half_pbc[i]:
                v13[i] += shared_pbc[i]
            r13 += v13[i]**2
        r21 = math.sqrt(r21)
        r23 = math.sqrt(r23)
        r13 = math.sqrt(r13)
        scaled_v21 = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        scaled_v23 = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        scaled_v13 = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)

        theta = 0
        for i in range(SPATIAL_DIM):
            scaled_v21[i] = v21[i] / r21
            scaled_v23[i] = v23[i] / r23
            scaled_v13[i] = v13[i] / r13
            theta += scaled_v21[i] * scaled_v23[i]
        # Harmonic angle
        # theta dot(v21, v23) / (r21 * r23)
        theta = math.acos(theta)
        delta_theta = theta - theta0
        force_val = - 2 * k * delta_theta
        energy += force_val * delta_theta *  -0.5
        # vec_norm cross(r21, r23)
        vec_norm = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        vec_norm[0] = scaled_v21[1]*scaled_v23[2] - scaled_v21[2]*scaled_v23[1]
        vec_norm[1] = - scaled_v21[0]*scaled_v23[2] + scaled_v21[2]*scaled_v23[0]
        vec_norm[2] = scaled_v21[0]*scaled_v23[1] - scaled_v21[1]*scaled_v23[0]
        # force_vec1 cross(r21, vec_norm)
        force_vec1 = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        force_vec1[0] = scaled_v21[1]*vec_norm[2] - scaled_v21[2]*vec_norm[1]
        force_vec1[1] = - scaled_v21[0]*vec_norm[2] + scaled_v21[2]*vec_norm[0]
        force_vec1[2] = scaled_v21[0]*vec_norm[1] - scaled_v21[1]*vec_norm[0]
        # force_vec3 cross(-r23, vec_norm)
        force_vec3 = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        force_vec3[0] = - scaled_v23[1]*vec_norm[2] + scaled_v23[2]*vec_norm[1]
        force_vec3[1] = scaled_v23[0]*vec_norm[2] - scaled_v23[2]*vec_norm[0]
        force_vec3[2] = - scaled_v23[0]*vec_norm[1] + scaled_v23[1]*vec_norm[0]
        norm_force_vec1 = 0
        norm_force_vec3 = 0
        for i in range(SPATIAL_DIM):
            norm_force_vec1 += force_vec1[i]**2
            norm_force_vec3 += force_vec3[i]**2
        norm_force_vec1 = math.sqrt(norm_force_vec1)
        norm_force_vec3 = math.sqrt(norm_force_vec3)

        # particle1
        force_val1 = force_val / r21 / norm_force_vec1
        force_val3 = force_val / r23 / norm_force_vec3
        for i in range(SPATIAL_DIM):
            local_forces[0, i] = force_val1 * force_vec1[i]
            local_forces[2, i] = force_val3 * force_vec3[i]
            local_forces[1, i] = - (local_forces[0, i] + local_forces[2, i])

        # Urey-Bradley
        delta_u = r13 - u0
        energy += ku * delta_u**2
        force_val = 2 * ku * delta_u
        for i in range(SPATIAL_DIM):
            force_val_ub = force_val * scaled_v13[i]
            local_forces[0, i] += force_val_ub
            local_forces[2, i] -= force_val_ub

        # Summary
        for i in range(3):
            for j in range(SPATIAL_DIM):
                cuda.atomic.add(forces, (particle_ids[i], j), local_forces[i, j])
        cuda.atomic.add(potential_energy, 0, energy)

    def update(self):
        self._check_bound_state()
        self._forces = cp.zeros(self._parent_ensemble.state.matrix_shape, CUPY_FLOAT)
        self._potential_energy = cp.zeros([1], CUPY_FLOAT)
        # Device
        self._update[self._block_per_grid, THREAD_PER_BLOCK](
            self._device_int_parameters,
            self._device_float_parameters,
            self._parent_ensemble.state.positions,
            self._parent_ensemble.state.device_pbc_matrix,
            self._forces, self._potential_energy
        )

    @property
    def num_angles(self):
        return self._num_angles