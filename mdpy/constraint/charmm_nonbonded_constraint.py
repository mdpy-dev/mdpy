#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : charmm_nonbonded_constraint.py
created time : 2021/10/12
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

from cupy.cuda.nvtx import RangePush, RangePop
import numpy as np
import numba as nb
from . import Constraint
from .. import env, SPATIAL_DIM
from ..ensemble import Ensemble
from ..math import *
from ..unit import *

NUM_NEIGHBOR_CELLS = 27
NEIGHBOR_CELL_TEMPLATE = np.zeros([NUM_NEIGHBOR_CELLS, SPATIAL_DIM], dtype=env.NUMPY_INT)
index = 0
for i in range(-1, 2):
    for j in range(-1, 2):
        for k in range(-1, 2):
            NEIGHBOR_CELL_TEMPLATE[index, :] = [i, j, k]
            index += 1

class CharmmNonbondedConstraint(Constraint):
    def __init__(self, params, cutoff_radius=12, force_id: int = 0, force_group: int = 0) -> None:
        super().__init__(params, force_id=force_id, force_group=force_group)
        self._cutoff_radius = check_quantity_value(cutoff_radius, default_length_unit)
        self._params_list = []
        self._neighbor_list = []
        self._neighbor_distance = []
        self._num_nonbonded_pairs = 0
        self._kernel = nb.njit((
            env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[:, ::1],
            env.NUMBA_FLOAT, env.NUMBA_INT[:, ::1], env.NUMBA_INT[:, ::1],
            env.NUMBA_INT[:, ::1], env.NUMBA_INT[:, :, :, ::1], env.NUMBA_INT[::1], env.NUMBA_INT[:, ::1]
        ))(self.kernel)

    def __repr__(self) -> str:
        return '<mdpy.constraint.CharmmNonbondedConstraint object>'

    __str__ = __repr__

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._force_id = ensemble.constraints.index(self)
        self._params_list = []
        for particle in self._parent_ensemble.topology.particles:
            param = self._params[particle.particle_name]
            if len(param) == 2:
                epsilon, sigma = param
                self._params_list.append([epsilon, sigma, epsilon, sigma])
            elif len(param) == 4:
                epsilon, sigma, epsilon14, sigma14 = param
                self._params_list.append([epsilon, sigma, epsilon14, sigma14])
        self._params_list = np.vstack(self._params_list).astype(env.NUMPY_FLOAT)

    @staticmethod
    def kernel(
        positions, params, pbc_matrix, pbc_inv, cutoff_radius,
        bonded_particles, scaling_particles, 
        particle_cell_index, cell_list, num_cell_vec, neighbor_cell_template
    ):
        int_type = particle_cell_index.dtype
        forces = np.zeros_like(positions)
        potential_energy = forces[0, 0]
        num_particles = positions.shape[0]
        num_neighbor_cells = neighbor_cell_template.shape[0]
        for id1 in range(num_particles):
            cur_bonded_particles = bonded_particles[id1][bonded_particles[id1] != -1]
            cur_scaling_particles = scaling_particles[id1][scaling_particles[id1] != -1]
            neighbor_cell = neighbor_cell_template + particle_cell_index[id1]
            wrap_flag = (neighbor_cell >= num_cell_vec).astype(int_type)
            neighbor_cell -= wrap_flag * num_cell_vec
            for cell_index in range(num_neighbor_cells):
                i, j, k = neighbor_cell[cell_index, :]
                neighbors = [i for i in cell_list[i, j, k, :] if not i in cur_bonded_particles and i != -1 and i != id1]
                for id2 in neighbors:
                    force_vec = unwrap_vec(
                        positions[id2] - positions[id1],
                        pbc_matrix, pbc_inv
                    )
                    r = np.linalg.norm(force_vec)
                    if r <= cutoff_radius:
                        force_vec /= r
                        if id2 in cur_scaling_particles:
                            epsilon1, sigma1 = params[id1, 2:]
                            epsilon2, sigma2 = params[id2, 2:]
                        else:
                            epsilon1, sigma1 = params[id1, :2]
                            epsilon2, sigma2 = params[id2, :2]
                        epsilon, sigma = (
                            np.sqrt(epsilon1 * epsilon2),
                            (sigma1 + sigma2) / 2
                        )
                        scaled_r = sigma / r
                        force_val = - (2 * scaled_r**12 - scaled_r**6) / r * epsilon * 24 # Sequence for small number divide small number
                        force = force_vec * force_val / 2
                        forces[id1, :] += force
                        forces[id2, :] -= force
                        potential_energy += 4 * epsilon * (scaled_r**12 - scaled_r**6) / 2
        return forces, potential_energy

    def update(self):
        self._check_bound_state()
        self._forces, self._potential_energy = self._kernel(
            self._parent_ensemble.state.positions, self._params_list, 
            *self._parent_ensemble.state.pbc_info, self._cutoff_radius, 
            self._parent_ensemble.topology.bonded_particles, 
            self._parent_ensemble.topology.scaling_particles,
            self._parent_ensemble.state.cell_list.particle_cell_index,
            self._parent_ensemble.state.cell_list.cell_list,
            self._parent_ensemble.state.cell_list.num_cell_vec,
            NEIGHBOR_CELL_TEMPLATE
        )

    @property
    def num_nonbonded_pairs(self):
        return self._num_nonbonded_pairs