#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : electrostatic_constraint.py
created time : 2021/10/13
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import math
import numpy as np
import numba as nb
from numba import cuda
from .. import env
from . import Constraint
from ..ensemble import Ensemble
from ..math import *
from ..unit import *

epsilon0 = EPSILON0.value

class ElectrostaticConstraint(Constraint):
    def __init__(self, params=None, force_id: int = 0, force_group: int = 0) -> None:
        super().__init__(params, force_id=force_id, force_group=force_group)
        self._int_params = []
        self._float_params = []
        if env.platform == 'CPU':
            self._kernel = nb.njit((
                env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[:, ::1], env.NUMBA_INT[:, ::1], 
                env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[:, ::1]
            ))(self.cpu_kernel)
        elif env.platform == 'CUDA':
            self._kernel = cuda.jit(nb.void(
                env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[::1],
                env.NUMBA_INT[:, ::1], env.NUMBA_FLOAT[:, ::1], 
                env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[::1]
            ))(self.cuda_kernel)

    def __repr__(self) -> str:
        return '<mdpy.constraint.ElectrostaticConstraint object>'

    __str__ = __repr__

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._force_id = ensemble.constraints.index(self)
        self._device_charges = cuda.to_device(self._parent_ensemble.topology.charges)
        self._device_k = cuda.to_device(np.array([4 * np.pi * epsilon0], dtype=env.NUMPY_FLOAT))

    @staticmethod
    def cpu_kernel(
        positions, charges, bonded_particles, 
        pbc_matrix, pbc_inv
    ):
        forces = np.zeros_like(positions)
        potential_energy = forces[0, 0]
        num_particles = positions.shape[0]
        k = 4 * np.pi * epsilon0
        for id1 in range(num_particles):
            cur_bonded_particles = bonded_particles[id1, :][bonded_particles[id1, :] != -1]
            for id2 in range(id1+1, num_particles):
                if not id2 in cur_bonded_particles:
                    e1 = charges[id1, 0]
                    e2 = charges[id2, 0]
                    force_vec = unwrap_vec(
                        positions[id2, :] - positions[id1, :],
                        pbc_matrix, pbc_inv
                    )
                    dist = np.linalg.norm(force_vec)
                    force_vec /= dist
                    force_val = - e1 * e2 / k / dist**2
                    force = force_vec * force_val
                    forces[id1, :] += force
                    forces[id2, :] -= force
                    # Potential energy
                    potential_energy += e1 * e2 / k / dist
        return forces, potential_energy

    @staticmethod
    def cuda_kernel(
        positions, charges, k, 
        bonded_particles, pbc_matrix, 
        forces, potential_energy
    ):
        thread_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        thread_y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        num_particles = positions.shape[0]
        id1, id2 = thread_x, thread_y
        if id1 >= num_particles:
            return None
        elif id2 >= num_particles:
            return None
        elif id1 == id2:
            return None
        for i in bonded_particles[id1, :]:
            if i == -1:
                break
            elif id2 == i:
                return None
        e1 = charges[id1, 0]
        e2 = charges[id2, 0]
        x = (positions[id2, 0] - positions[id1, 0]) / pbc_matrix[0, 0]
        x = (x - round(x)) * pbc_matrix[0, 0]
        y = (positions[id2, 1] - positions[id1, 1]) / pbc_matrix[1, 1]
        y = (y - round(y)) * pbc_matrix[1, 1]
        z = (positions[id2, 2] - positions[id1, 2]) / pbc_matrix[2, 2]
        z = (z - round(z)) * pbc_matrix[2, 2]
        r = math.sqrt(x**2 + y**2 + z**2)
        scaled_x, scaled_y, scaled_z = x / r, y / r, z / r
        force_val = - e1 * e2 / k[0] / r**2
        force_x = scaled_x * force_val / 2
        force_y = scaled_y * force_val / 2
        force_z = scaled_z * force_val / 2
        cuda.atomic.add(forces, (id1, 0), force_x)
        cuda.atomic.add(forces, (id1, 1), force_y)
        cuda.atomic.add(forces, (id1, 2), force_z)
        cuda.atomic.add(forces, (id2, 0), -force_x)
        cuda.atomic.add(forces, (id2, 1), -force_y)
        cuda.atomic.add(forces, (id2, 2), -force_z)
        energy = e1 * e2 / k[0] / r / 2
        cuda.atomic.add(potential_energy, 0, energy)

    def update(self):
        self._check_bound_state()
        if env.platform == 'CPU':
            self._forces, self._potential_energy = self._kernel(
                self._parent_ensemble.state.positions,
                self._parent_ensemble.topology.charges,
                self._parent_ensemble.topology.bonded_particles, 
                *self._parent_ensemble.state.pbc_info
            )
        elif env.platform == 'CUDA':
            self._forces = np.zeros_like(self._parent_ensemble.state.positions)
            self._potential_energy = np.zeros([1], dtype=env.NUMPY_FLOAT)
            d_positions = cuda.to_device(self._parent_ensemble.state.positions)
            d_bonded_particles = cuda.to_device(self._parent_ensemble.topology.bonded_particles)
            d_pbc_matrix = cuda.to_device(self._parent_ensemble.state.pbc_matrix)
            d_forces = cuda.to_device(self._forces)
            d_potential_energy = cuda.to_device(self._potential_energy)

            thread_per_block = (32, 32)
            block_per_grid = int(np.ceil(
                self._parent_ensemble.topology.num_particles / thread_per_block[0]
            ))
            block_per_grid = (block_per_grid, block_per_grid)
            self._kernel[block_per_grid, thread_per_block](
                d_positions, self._device_charges, self._device_k,
                d_bonded_particles, d_pbc_matrix,
                d_forces, d_potential_energy
            )
            self._forces = d_forces.copy_to_host()
            self._potential_energy = d_potential_energy.copy_to_host()[0]