#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : amber_bond_constraint.py
created time : 2021/10/09
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import math
import numpy as np
import numba as nb
import numba.cuda as cuda
from mdpy import env, SPATIAL_DIM
from mdpy.core import Ensemble
from mdpy.constraint import Constraint
from mdpy.utils import *

THREAD_PER_BLOCK = (32)

class CharmmBondConstraint(Constraint):
    def __init__(self, parameter_dict: dict) -> None:
        super().__init__()
        self._parameter_dict = parameter_dict
        self._int_parameters = []
        self._float_parameters = []
        self._num_bonds = 0
        self._update = cuda.jit(nb.void(
            env.NUMBA_INT[:, ::1], # int_parameters
            env.NUMBA_FLOAT[:, ::1], # float_parameters
            env.NUMBA_FLOAT[:, ::1], # positions
            env.NUMBA_FLOAT[:, ::1], # pbc_matrix
            env.NUMBA_FLOAT[:, ::1], # forces
            env.NUMBA_FLOAT[::1] # potential_energy
        ))(self._update_kernel)

    def __repr__(self) -> str:
        return '<mdpy.constraint.CharmmBondConstraint object>'

    def __str__(self) -> str:
        return 'Bond constraint'

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._force_id = ensemble.constraints.index(self)
        self._int_parameters = []
        self._float_parameters = []
        self._num_bonds = 0
        for bond in self._parent_ensemble.topology.bonds:
            bond_type = '%s-%s' %(
                self._parent_ensemble.topology.particles[bond[0]].particle_type,
                self._parent_ensemble.topology.particles[bond[1]].particle_type
            )
            # Matrix id of two bonded particles
            self._int_parameters.append([
                self._parent_ensemble.topology.particles[bond[0]].matrix_id,
                self._parent_ensemble.topology.particles[bond[1]].matrix_id
            ])
            # Bond parameters
            self._float_parameters.append(self._parameter_dict[bond_type])
            self._num_bonds += 1
        self._int_parameters = np.vstack(self._int_parameters).astype(env.NUMPY_INT)
        self._float_parameters = np.vstack(self._float_parameters).astype(env.NUMPY_FLOAT)
        self._device_int_parameters = cuda.to_device(self._int_parameters)
        self._device_float_parameters =cuda.to_device(self._float_parameters)

    @staticmethod
    def kernel(int_parameters, float_parameters, positions, pbc_matrix, pbc_inv):
        forces = np.zeros_like(positions)
        potential_energy = forces[0, 0]
        num_parameters = int_parameters.shape[0]
        for bond in range(num_parameters):
            id1, id2, = int_parameters[bond, :]
            k, r0 = float_parameters[bond, :]
            force_vec = unwrap_vec(
                positions[id2, :] - positions[id1, :],
                pbc_matrix, pbc_inv
            )
            r = np.linalg.norm(force_vec)
            force_vec /= r
            # Forces
            force_val = 2 * k * (r - r0)
            force = force_val * force_vec
            forces[id1, :] += force
            forces[id2, :] -= force
            # Potential energy
            potential_energy += k * (r - r0)**2
        return forces, potential_energy

    @staticmethod
    def _update_kernel(
        int_parameters, float_parameters,
        positions, pbc_matrix,
        forces, potential_energy
    ):
        bond_id = cuda.grid(1)
        shared_num_bonds = cuda.shared.array(shape=(1), dtype=nb.int32)
        shared_pbc = cuda.shared.array(shape=(SPATIAL_DIM), dtype=nb.float32)
        shared_half_pbc = cuda.shared.array(shape=(SPATIAL_DIM), dtype=nb.float32)
        if cuda.threadIdx.x == 0:
            shared_num_bonds[0] = int_parameters.shape[0]
        if cuda.threadIdx.x == 1:
            shared_half_pbc[0] = pbc_matrix[0, 0]
            shared_half_pbc[1] = pbc_matrix[1, 1]
            shared_half_pbc[2] = pbc_matrix[2, 2]
            shared_half_pbc[0] = shared_half_pbc[0] / 2
            shared_half_pbc[1] = shared_half_pbc[1] / 2
            shared_half_pbc[2] = shared_half_pbc[2] / 2
        cuda.syncthreads()
        if bond_id >= shared_num_bonds[0]:
            return None
        id1, id2, = int_parameters[bond_id, :]
        k, r0 = float_parameters[bond_id, :]
        # Positions
        position_1_x = positions[id1, 0]
        position_1_y = positions[id1, 1]
        position_1_z = positions[id1, 2]
        position_2_x = positions[id2, 0]
        position_2_y = positions[id2, 1]
        position_2_z = positions[id2, 2]
        # vec
        x21 = position_2_x - position_1_x
        if x21 >= shared_half_pbc[0]:
            x21 -= shared_pbc[0]
        elif x21 <= -shared_half_pbc[0]:
            x21 += shared_pbc[0]
        y21 = position_2_y - position_1_y
        if y21 >= shared_half_pbc[1]:
            y21 -= shared_pbc[1]
        elif y21 <= -shared_half_pbc[1]:
            y21 += shared_pbc[1]
        z21 = position_2_z - position_1_z
        if z21 >= shared_half_pbc[2]:
            z21 -= shared_pbc[2]
        elif z21 <= -shared_half_pbc[2]:
            z21 += shared_pbc[2]
        l21 = math.sqrt(x21**2 + y21**2 + z21**2)
        scaled_x21 = x21 / l21
        scaled_y21 = y21 / l21
        scaled_z21 = z21 / l21
        # Energy
        delta_r = l21 - r0
        energy = k * delta_r**2
        force_val = 2 * k * delta_r
        force_x = force_val * scaled_x21
        force_y = force_val * scaled_y21
        force_z = force_val * scaled_z21
        # Summary
        cuda.atomic.add(forces, (id1, 0), force_x)
        cuda.atomic.add(forces, (id1, 1), force_y)
        cuda.atomic.add(forces, (id1, 2), force_z)
        cuda.atomic.add(forces, (id2, 0), -force_x)
        cuda.atomic.add(forces, (id2, 1), -force_y)
        cuda.atomic.add(forces, (id2, 2), -force_z)
        cuda.atomic.add(potential_energy, 0, energy)

    def update(self):
        self._check_bound_state()
        self._forces = np.zeros_like(self._parent_ensemble.state.positions)
        self._potential_energy = np.zeros([1], dtype=env.NUMPY_FLOAT)
        # Device
        device_forces = cuda.to_device(self._forces)
        device_potential_energy = cuda.to_device(self._potential_energy)
        block_per_grid = (int(np.ceil(
            self._parent_ensemble.topology.num_bonds / THREAD_PER_BLOCK
        )))
        self._update[block_per_grid, THREAD_PER_BLOCK](
            self._device_int_parameters,
            self._device_float_parameters,
            self._parent_ensemble.state.device_positions,
            self._parent_ensemble.state.device_pbc_matrix,
            device_forces, device_potential_energy
        )
        self._forces = device_forces.copy_to_host()
        self._potential_energy = device_potential_energy.copy_to_host()[0]

    @property
    def num_bonds(self):
        return self._num_bonds