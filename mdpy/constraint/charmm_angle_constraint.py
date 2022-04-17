#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : charmm_angle_constraint.py
created time : 2021/10/10
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import math
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
        self._int_parameters = np.vstack(self._int_parameters).astype(NUMPY_INT)
        self._float_parameters = np.vstack(self._float_parameters).astype(NUMPY_FLOAT)
        self._device_int_parameters = cuda.to_device(self._int_parameters)
        self._device_float_parameters =cuda.to_device(self._float_parameters)

    @staticmethod
    def _cpu_kernel(int_parameters, float_parameters, positions, pbc_matrix, pbc_inv):
        forces = np.zeros_like(positions)
        potential_energy = forces[0, 0]
        num_angles = int_parameters.shape[0]
        for angle in range(num_angles):
            id1, id2, id3 = int_parameters[angle]
            k, theta0, ku, u0 = float_parameters[angle, :]
            r21 = unwrap_vec(
                positions[id1, :] - positions[id2, :],
                pbc_matrix, pbc_inv
            )
            l21 = np.linalg.norm(r21)
            r23 = unwrap_vec(
                positions[id3, :] - positions[id2, :],
                pbc_matrix, pbc_inv
            )
            l23 = np.linalg.norm(r23)
            cos_theta = np.dot(r21, r23) / (l21 * l23)
            theta = np.arccos(cos_theta)
            # Force
            force_val = - 2 * k * (theta - theta0)
            vec_norm = np.cross(r21, r23)
            force_vec1 = get_unit_vec(np.cross(r21, vec_norm)) / l21
            force_vec3 = get_unit_vec(np.cross(-r23, vec_norm)) / l23
            forces[id1, :] += force_val * force_vec1
            forces[id2, :] -= force_val * (force_vec1 + force_vec3)
            forces[id3, :] += force_val * force_vec3
            # Potential energy
            potential_energy += k * (theta - theta0)**2
            # Urey-Bradley
            r13 = unwrap_vec(
                positions[id3, :] - positions[id1, :],
                pbc_matrix, pbc_inv
            )
            l13 = np.linalg.norm(r13)
            force_val = 2 * ku * (l13 - u0)
            force_vec = r13 / l13
            forces[id1, :] += force_val * force_vec
            forces[id3, :] -= force_val * force_vec
            potential_energy += ku * (l13 - u0)**2
        return forces, potential_energy

    @staticmethod
    def _update_kernel(
        int_parameters, float_parameters,
        positions, pbc_matrix,
        forces, potential_energy
    ):
        angle_id = cuda.grid(1)
        shared_num_angles = cuda.shared.array(shape=(1), dtype=nb.int32)
        shared_pbc = cuda.shared.array(shape=(SPATIAL_DIM), dtype=nb.float32)
        shared_half_pbc = cuda.shared.array(shape=(SPATIAL_DIM), dtype=nb.float32)
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
        force_1_x = 0
        force_1_y = 0
        force_1_z = 0
        force_2_x = 0
        force_2_y = 0
        force_2_z = 0
        force_3_x = 0
        force_3_y = 0
        force_3_z = 0
        energy = 0
        # Positions
        position_1_x = positions[id1, 0]
        position_1_y = positions[id1, 1]
        position_1_z = positions[id1, 2]
        position_2_x = positions[id2, 0]
        position_2_y = positions[id2, 1]
        position_2_z = positions[id2, 2]
        position_3_x = positions[id3, 0]
        position_3_y = positions[id3, 1]
        position_3_z = positions[id3, 2]
        # vec
        # r21
        x21 = position_1_x - position_2_x
        if x21 >= shared_half_pbc[0]:
            x21 -= shared_pbc[0]
        elif x21 <= -shared_half_pbc[0]:
            x21 += shared_pbc[0]
        y21 = position_1_y - position_2_y
        if y21 >= shared_half_pbc[1]:
            y21 -= shared_pbc[1]
        elif y21 <= -shared_half_pbc[1]:
            y21 += shared_pbc[1]
        z21 = position_1_z - position_2_z
        if z21 >= shared_half_pbc[2]:
            z21 -= shared_pbc[2]
        elif z21 <= -shared_half_pbc[2]:
            z21 += shared_pbc[2]
        l21 = math.sqrt(x21**2 + y21**2 + z21**2)
        scaled_x21 = x21 / l21
        scaled_y21 = y21 / l21
        scaled_z21 = z21 / l21
        # r23
        x23 = position_3_x - position_2_x
        if x23 >= shared_half_pbc[0]:
            x23 -= shared_pbc[0]
        elif x23 <= -shared_half_pbc[0]:
            x23 += shared_pbc[0]
        y23 = position_3_y - position_2_y
        if y23 >= shared_half_pbc[1]:
            y23 -= shared_pbc[1]
        elif y23 <= -shared_half_pbc[1]:
            y23 += shared_pbc[1]
        z23 = position_3_z - position_2_z
        if z23 >= shared_half_pbc[2]:
            z23 -= shared_pbc[2]
        elif z23 <= -shared_half_pbc[2]:
            z23 += shared_pbc[2]
        l23 = math.sqrt(x23**2 + y23**2 + z23**2)
        scaled_x23 = x23 / l23
        scaled_y23 = y23 / l23
        scaled_z23 = z23 / l23
        # r13
        x13 = position_3_x - position_1_x
        if x13 >= shared_half_pbc[0]:
            x13 -= shared_pbc[0]
        elif x13 <= -shared_half_pbc[0]:
            x13 += shared_pbc[0]
        y13 = position_3_y - position_1_y
        if y13 >= shared_half_pbc[1]:
            y13 -= shared_pbc[1]
        elif y13 <= -shared_half_pbc[1]:
            y13 += shared_pbc[1]
        z13 = position_3_z - position_1_z
        if z13 >= shared_half_pbc[2]:
            z13 -= shared_pbc[2]
        elif z13 <= -shared_half_pbc[2]:
            z13 += shared_pbc[2]
        l13 = math.sqrt(x13**2 + y13**2 + z13**2)
        scaled_x13 = x13 / l13
        scaled_y13 = y13 / l13
        scaled_z13 = z13 / l13
        # Harmonic angle
        # theta dot(r21, r23) / (l21 * l23)
        theta = math.acos(
            scaled_x21*scaled_x23 +
            scaled_y21*scaled_y23 +
            scaled_z21*scaled_z23
        )
        delta_theta = theta - theta0
        energy += k * delta_theta**2
        # vec_norm cross(r21, r23)
        vec_norm_x = scaled_y21*scaled_z23 - scaled_y23*scaled_z21
        vec_norm_y = - scaled_x21*scaled_z23 + scaled_x23*scaled_z21
        vec_norm_z = scaled_x21*scaled_y23 - scaled_x23*scaled_y21
        # force_vec1 cross(r21, vec_norm)
        force_vec1_x = scaled_y21*vec_norm_z - vec_norm_y*scaled_z21
        force_vec1_y = - scaled_x21*vec_norm_z + vec_norm_x*scaled_z21
        force_vec1_z = scaled_x21*vec_norm_y - vec_norm_x*scaled_y21
        force_vec1_norm = math.sqrt(
            force_vec1_x**2 + force_vec1_y**2 + force_vec1_z**2
        )
        # force_vec3 cross(-r23, vec_norm)
        force_vec3_x = - scaled_y23*vec_norm_z + vec_norm_y*scaled_z23
        force_vec3_y = scaled_x23*vec_norm_z - vec_norm_x*scaled_z23
        force_vec3_z = - scaled_x23*vec_norm_y + vec_norm_x*scaled_y23
        force_vec3_norm = math.sqrt(
            force_vec3_x**2 + force_vec3_y**2 + force_vec3_z**2
        )

        force_val = - 2 * k * delta_theta
        # particle1
        force_val1 = force_val / l21 / force_vec1_norm
        force_1_x += force_val1 * force_vec1_x
        force_1_y += force_val1 * force_vec1_y
        force_1_z += force_val1 * force_vec1_z
        # particle3
        force_val3 = force_val / l23 / force_vec3_norm
        force_3_x += force_val3 * force_vec3_x
        force_3_y += force_val3 * force_vec3_y
        force_3_z += force_val3 * force_vec3_z
        # particle2
        force_2_x -= force_1_x + force_3_x
        force_2_y -= force_1_y + force_3_y
        force_2_z -= force_1_z + force_3_z

        # Urey-Bradley
        delta_u = l13 - u0
        energy += ku * delta_u**2
        force_val = 2 * ku * delta_u
        force_val_x = force_val * scaled_x13
        force_val_y = force_val * scaled_y13
        force_val_z = force_val * scaled_z13
        # particle1
        force_1_x += force_val_x
        force_1_y += force_val_y
        force_1_z += force_val_z
        # particle1
        force_3_x -= force_val_x
        force_3_y -= force_val_y
        force_3_z -= force_val_z

        # Summary
        cuda.atomic.add(forces, (id1, 0), force_1_x)
        cuda.atomic.add(forces, (id1, 1), force_1_y)
        cuda.atomic.add(forces, (id1, 2), force_1_z)
        cuda.atomic.add(forces, (id2, 0), force_2_x)
        cuda.atomic.add(forces, (id2, 1), force_2_y)
        cuda.atomic.add(forces, (id2, 2), force_2_z)
        cuda.atomic.add(forces, (id3, 0), force_3_x)
        cuda.atomic.add(forces, (id3, 1), force_3_y)
        cuda.atomic.add(forces, (id3, 2), force_3_z)
        cuda.atomic.add(potential_energy, 0, energy)

    def update(self):
        self._check_bound_state()
        self._forces = cp.zeros_like(self._parent_ensemble.state.positions, CUPY_FLOAT)
        self._potential_energy = cp.zeros([1], CUPY_FLOAT)
        # Device
        block_per_grid = (int(np.ceil(
            self._parent_ensemble.topology.num_angles / THREAD_PER_BLOCK
        )))
        self._update[block_per_grid, THREAD_PER_BLOCK](
            self._device_int_parameters,
            self._device_float_parameters,
            self._parent_ensemble.state.device_positions,
            self._parent_ensemble.state.device_pbc_matrix,
            self._forces, self._potential_energy
        )

    @property
    def num_angles(self):
        return self._num_angles