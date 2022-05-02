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
import cupy as cp
import numba.cuda as cuda
from mdpy import SPATIAL_DIM
from mdpy.environment import *
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
            NUMBA_INT[:, ::1], # int_parameters
            NUMBA_FLOAT[:, ::1], # float_parameters
            NUMBA_FLOAT[:, ::1], # positions
            NUMBA_FLOAT[:, ::1], # pbc_matrix
            NUMBA_FLOAT[:, ::1], # forces
            NUMBA_FLOAT[::1] # potential_energy
        ))(self._update_charmm_bond_kernel)

    def __repr__(self) -> str:
        return '<mdpy.constraint.CharmmBondConstraint object>'

    def __str__(self) -> str:
        return 'Bond constraint'

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._constraint_id = ensemble.constraints.index(self)
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
        self._device_int_parameters = cp.array(np.vstack(self._int_parameters), CUPY_INT)
        self._device_float_parameters = cp.array(np.vstack(self._float_parameters), CUPY_FLOAT)
        self._block_per_grid = (int(np.ceil(
            self._parent_ensemble.topology.num_bonds / THREAD_PER_BLOCK
        )))

    @staticmethod
    def _update_charmm_bond_kernel(
        int_parameters, float_parameters,
        positions, pbc_matrix,
        forces, potential_energy
    ):
        bond_id = cuda.grid(1)
        num_bonds = int_parameters.shape[0]
        if bond_id >= num_bonds:
            return None
        shared_pbc = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        shared_half_pbc = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        if cuda.threadIdx.x == 0:
            shared_pbc[0] = pbc_matrix[0, 0]
            shared_pbc[1] = pbc_matrix[1, 1]
            shared_pbc[2] = pbc_matrix[2, 2]
            shared_half_pbc[0] = shared_pbc[0] / 2
            shared_half_pbc[1] = shared_pbc[1] / 2
            shared_half_pbc[2] = shared_pbc[2] / 2
        cuda.syncthreads()
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
    def num_bonds(self):
        return self._num_bonds