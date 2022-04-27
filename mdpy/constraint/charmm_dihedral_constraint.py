#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : charmm_dihedral_constraint.py
created time : 2021/10/11
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import math
from threading import local
import numpy as np
import numba as nb
import numba.cuda as cuda
from mdpy import SPATIAL_DIM
from mdpy.core import Ensemble
from mdpy.environment import *
from mdpy.constraint import Constraint
from mdpy.utils import *
from mdpy.unit import *

THREAD_PER_BLOCK = (32)

class CharmmDihedralConstraint(Constraint):
    def __init__(self, parameter_dict: dict) -> None:
        super().__init__()
        self._parameter_dict = parameter_dict
        self._int_parameters = []
        self._float_parameters = []
        self._num_dihedrals = 0
        # Kernel
        self._update = cuda.jit(nb.void(
            NUMBA_INT[:, ::1], # int_parameters
            NUMBA_FLOAT[:, ::1], # float_parameters
            NUMBA_FLOAT[:, ::1], # positions
            NUMBA_FLOAT[:, ::1], # pbc_matrix
            NUMBA_FLOAT[:, ::1], # forces
            NUMBA_FLOAT[::1] # potential_energy
        ))(self._update_kernel)

    def __repr__(self) -> str:
        return '<mdpy.constraint.CharmmDihedralConstraint object>'

    def __str__(self) -> str:
        return 'Dihedral constraint'

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._constraint_id = ensemble.constraints.index(self)
        self._int_parameters = []
        self._float_parameters = []
        self._num_dihedrals = 0
        for dihedral in self._parent_ensemble.topology.dihedrals:
            dihedral_type = '%s-%s-%s-%s' %(
                self._parent_ensemble.topology.particles[dihedral[0]].particle_type,
                self._parent_ensemble.topology.particles[dihedral[1]].particle_type,
                self._parent_ensemble.topology.particles[dihedral[2]].particle_type,
                self._parent_ensemble.topology.particles[dihedral[3]].particle_type
            )
            for float_param in self._parameter_dict[dihedral_type]:
                # matrix_id of 4 particles which form the dihedral
                self._int_parameters.append([
                    self._parent_ensemble.topology.particles[dihedral[0]].matrix_id,
                    self._parent_ensemble.topology.particles[dihedral[1]].matrix_id,
                    self._parent_ensemble.topology.particles[dihedral[2]].matrix_id,
                    self._parent_ensemble.topology.particles[dihedral[3]].matrix_id
                ])
                # dihedral coefficient
                self._float_parameters.append(float_param)
            self._num_dihedrals += 1
        self._int_parameters = np.vstack(self._int_parameters).astype(NUMPY_INT)
        self._float_parameters = np.vstack(self._float_parameters).astype(NUMPY_FLOAT)
        self._device_int_parameters = cuda.to_device(self._int_parameters)
        self._device_float_parameters = cuda.to_device(self._float_parameters)

    @staticmethod
    def _update_kernel(
        int_parameters, float_parameters,
        positions, pbc_matrix,
        forces, potential_energy
    ):
        dihedral_id = cuda.grid(1)
        num_dihedrals = int_parameters.shape[0]
        if dihedral_id >= num_dihedrals:
            return None
        shared_pbc = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        shared_half_pbc = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        if cuda.threadIdx.x == 0:
            shared_pbc[0] = pbc_matrix[0, 0]
            shared_pbc[1] = pbc_matrix[1, 1]
            shared_pbc[2] = pbc_matrix[2, 2]
            shared_half_pbc[0] = shared_pbc[0] * 0.5
            shared_half_pbc[1] = shared_pbc[1] * 0.5
            shared_half_pbc[2] = shared_pbc[2] * 0.5
        cuda.syncthreads()
        particle_ids = cuda.local.array(shape=(4), dtype=NUMBA_INT)
        local_positions = cuda.local.array(shape=(4, SPATIAL_DIM), dtype=NUMBA_FLOAT)
        for i in range(4):
            particle_ids[i] = int_parameters[dihedral_id, i]
            local_positions[i, 0] = positions[particle_ids[i], 0]
            local_positions[i, 1] = positions[particle_ids[i], 1]
            local_positions[i, 2] = positions[particle_ids[i], 2]
        k, n, delta = float_parameters[dihedral_id, :]

        v12 = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        v23 = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        vo3 = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        v34 = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        r12, r23, ro3_square, r34 = 0, 0, 0, 0
        for i in range(SPATIAL_DIM):
            v12[i] = local_positions[1, i] - local_positions[0, i]
            if v12[i] >= shared_half_pbc[i]:
                v12[i] -= shared_pbc[i]
            elif v12[i] <= -shared_half_pbc[i]:
                v12[i] += shared_pbc[i]
            r12 += v12[i]**2

            v23[i] = local_positions[2, i] - local_positions[1, i]
            if v23[i] >= shared_half_pbc[i]:
                v23[i] -= shared_pbc[i]
            elif v23[i] <= -shared_half_pbc[i]:
                v23[i] += shared_pbc[i]
            r23 += v23[i]**2

            vo3[i] = v23[i] * 0.5
            ro3_square += vo3[i]**2

            v34[i] = local_positions[3, i] - local_positions[2, i]
            if v34[i] >= shared_half_pbc[i]:
                v34[i] -= shared_pbc[i]
            elif v34[i] <= -shared_half_pbc[i]:
                v34[i] += shared_pbc[i]
            r34 += v34[i]**2
        r12 = math.sqrt(r12)
        r23 = math.sqrt(r23)
        r34 = math.sqrt(r34)

        # Dihedral
        n1 = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT) # v12 x v23
        n1[0] = v12[1]*v23[2] - v12[2]*v23[1]
        n1[1] = - v12[0]*v23[2] + v12[2]*v23[0]
        n1[2] = v12[0]*v23[1] - v12[1]*v23[0]
        rn1 = math.sqrt(n1[0]**2 + n1[1]**2 + n1[2]**2)

        n2 = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT) # v23 x v34
        n2[0] = v23[1]*v34[2] - v23[2]*v34[1]
        n2[1] = - v23[0]*v34[2] + v23[2]*v34[0]
        n2[2] = v23[0]*v34[1] - v23[1]*v34[0]
        rn2 = math.sqrt(n2[0]**2 + n2[1]**2 + n2[2]**2)

        x, y = 0, 0
        for i in range(SPATIAL_DIM):
            x += v12[i] * n2[i]
            y += n1[i] * n2[i]
        x *= r23
        theta = math.atan2(x, y)
        # Angles
        cos_theta_123, cos_theta_234 = 0, 0
        for i in range(SPATIAL_DIM):
            cos_theta_123 -= v12[i] * v23[i]
            cos_theta_234 -= v23[i] * v34[i]
        cos_theta_123 /= r12 * r23
        cos_theta_234 /= r23 * r34
        # Force
        factor = n*theta - delta
        force_val = - k * (1 - n * math.sin(factor))
        local_forces = cuda.local.array(shape=(4, SPATIAL_DIM), dtype=NUMBA_FLOAT)
        r12_times_sin_theta_123 = r12 * math.sqrt(1 - cos_theta_123**2)
        r23_times_sin_theta_234 = r23 * math.sqrt(1 - cos_theta_234**2)
        for i in range(SPATIAL_DIM):
            # force_1 = force_val / (r12 * np.sin(theta_123)) * get_unit_vec(np.cross(-v12, v23))
            local_forces[0, i] = force_val / r12_times_sin_theta_123 * (-n1[i] / rn1)
            # force_4 = force_val / (r34 * np.sin(theta_234)) * get_unit_vec(np.cross(v34, -v23))
            local_forces[3, i] = force_val / r23_times_sin_theta_234 * (n2[i] / rn2)
        # forces[2, i] =  np.cross(
        #     -(np.cross(vo3,force_4)+np.cross(v34,force_4)/2+np.cross(-v12, force_1)/2), vo3
        # ) / ro3**2
        torque = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        torque[0] = - (
            (vo3[1]*local_forces[3, 2] - vo3[2]*local_forces[3, 1]) +
            (v34[1]*local_forces[3, 2] - v34[2]*local_forces[3, 1]) * 0.5 -
            (v12[1]*local_forces[0, 2] - v12[2]*local_forces[0, 1]) * 0.5
        )
        torque[1] = - (
            - (vo3[0]*local_forces[3, 2] - vo3[2]*local_forces[3, 0]) -
            (v34[0]*local_forces[3, 2] - v34[2]*local_forces[3, 0]) * 0.5 +
            (v12[0]*local_forces[0, 2] - v12[2]*local_forces[0, 0]) * 0.5
        )
        torque[2] = - (
            (vo3[0]*local_forces[3, 1] - vo3[1]*local_forces[3, 0]) +
            (v34[0]*local_forces[3, 1] - v34[1]*local_forces[3, 0]) * 0.5 -
            (v12[0]*local_forces[0, 1] - v12[1]*local_forces[0, 0]) * 0.5
        )
        local_forces[2, 0] = (torque[1]*vo3[2] - torque[2]*vo3[1]) / ro3_square
        local_forces[2, 1] = - (torque[0]*vo3[2] - torque[2]*vo3[0]) / ro3_square
        local_forces[2, 2] = (torque[0]*vo3[1] - torque[1]*vo3[0]) / ro3_square
        for i in range(SPATIAL_DIM):
            local_forces[1, i] = - (
                local_forces[0, i] + local_forces[2, i] + local_forces[3, i]
            )
        energy = k * (1 + math.cos(factor))
        for i in range(4):
            for j in range(SPATIAL_DIM):
                cuda.atomic.add(forces, (particle_ids[i], j), local_forces[i, j])
        cuda.atomic.add(potential_energy, (0), energy)

    def update(self):
        self._check_bound_state()
        # V(dihedral) = Kchi(1 + cos(n(chi) - delta))
        self._forces = cp.zeros_like(self._parent_ensemble.state.positions, CUPY_FLOAT)
        self._potential_energy = cp.zeros([1], CUPY_FLOAT)
        # Device
        block_per_grid = (int(np.ceil(
            self._parent_ensemble.topology.num_dihedrals / THREAD_PER_BLOCK
        )))
        self._update[block_per_grid, THREAD_PER_BLOCK](
            self._device_int_parameters,
            self._device_float_parameters,
            self._parent_ensemble.state.device_positions,
            self._parent_ensemble.state.device_pbc_matrix,
            self._forces, self._potential_energy
        )

    @property
    def num_dihedrals(self):
        return self._num_dihedrals