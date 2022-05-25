#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : amber_bond_constraint.py
created time : 2021/10/09
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

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

THREAD_PER_BLOCK = 128


class CharmmBondConstraint(Constraint):
    def __init__(self, parameter_dict: dict) -> None:
        super().__init__()
        self._parameter_dict = parameter_dict
        self._int_parameters = []
        self._float_parameters = []
        self._num_bonds = 0
        self._update = cuda.jit(
            nb.void(
                NUMBA_INT[:, ::1],  # int_parameters
                NUMBA_FLOAT[:, ::1],  # float_parameters
                NUMBA_FLOAT[:, ::1],  # positions
                NUMBA_FLOAT[:, ::1],  # pbc_matrix
                NUMBA_FLOAT[:, ::1],  # forces
                NUMBA_FLOAT[::1],  # potential_energy
            ),
            fastmath=True,
        )(self._update_charmm_bond_kernel)

    def __repr__(self) -> str:
        return "<mdpy.constraint.CharmmBondConstraint object>"

    def __str__(self) -> str:
        return "Bond constraint"

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._constraint_id = ensemble.constraints.index(self)
        self._int_parameters = []
        self._float_parameters = []
        self._num_bonds = 0
        for bond in self._parent_ensemble.topology.bonds:
            bond_type = "%s-%s" % (
                self._parent_ensemble.topology.particles[bond[0]].particle_type,
                self._parent_ensemble.topology.particles[bond[1]].particle_type,
            )
            # Matrix id of two bonded particles
            self._int_parameters.append(
                [
                    self._parent_ensemble.topology.particles[bond[0]].matrix_id,
                    self._parent_ensemble.topology.particles[bond[1]].matrix_id,
                ]
            )
            # Bond parameters
            self._float_parameters.append(self._parameter_dict[bond_type])
            self._num_bonds += 1
        self._device_int_parameters = cp.array(
            np.vstack(self._int_parameters), CUPY_INT
        )
        self._device_float_parameters = cp.array(
            np.vstack(self._float_parameters), CUPY_FLOAT
        )
        self._block_per_grid = int(
            np.ceil(self._parent_ensemble.topology.num_bonds / THREAD_PER_BLOCK)
        )

    @staticmethod
    def _update_charmm_bond_kernel(
        int_parameters,
        float_parameters,
        positions,
        pbc_matrix,
        forces,
        potential_energy,
    ):
        bond_id = cuda.grid(1)
        num_bonds = int_parameters.shape[0]
        if bond_id >= num_bonds:
            return None
        shared_pbc = cuda.shared.array((SPATIAL_DIM), NUMBA_FLOAT)
        shared_half_pbc = cuda.shared.array((SPATIAL_DIM), NUMBA_FLOAT)
        local_thread_x = cuda.threadIdx.x
        if local_thread_x <= 2:
            shared_pbc[local_thread_x] = pbc_matrix[local_thread_x, local_thread_x]
            shared_half_pbc[local_thread_x] = shared_pbc[local_thread_x] * NUMBA_FLOAT(
                0.5
            )
        cuda.syncthreads()
        (
            id1,
            id2,
        ) = int_parameters[bond_id, :]
        k, r0 = float_parameters[bond_id, :]
        # Positions
        positions_1 = cuda.local.array((SPATIAL_DIM), NUMBA_FLOAT)
        positions_2 = cuda.local.array((SPATIAL_DIM), NUMBA_FLOAT)
        vec = cuda.local.array((SPATIAL_DIM), NUMBA_FLOAT)
        for i in range(SPATIAL_DIM):
            positions_1[i] = positions[id1, i]
            positions_2[i] = positions[id2, i]
        # vec
        r = NUMBA_FLOAT(0)
        for i in range(SPATIAL_DIM):
            vec[i] = positions_2[i] - positions_1[i]
            if vec[i] < -shared_half_pbc[i]:
                vec[i] += shared_pbc[i]
            elif vec[i] > shared_half_pbc[i]:
                vec[i] -= shared_pbc[i]
            r += vec[i] ** 2
        r = math.sqrt(r)
        # Energy
        delta_r = r - r0
        energy = k * delta_r**2
        force_val = NUMBA_FLOAT(2) * k * delta_r / r
        for i in range(SPATIAL_DIM):
            temp = force_val * vec[i]
            cuda.atomic.add(forces, (id1, i), temp)
            cuda.atomic.add(forces, (id2, i), -temp)
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
            self._forces,
            self._potential_energy,
        )

    @property
    def num_bonds(self):
        return self._num_bonds
