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
from .. import env
from ..ensemble import Ensemble
from ..math import *
from ..unit import *

class CharmmNonbondedConstraint(Constraint):
    def __init__(self, params, cutoff_radius=12, force_id: int = 0, force_group: int = 0) -> None:
        super().__init__(params, force_id=force_id, force_group=force_group)
        self._cutoff_radius = check_quantity_value(cutoff_radius, default_length_unit)
        self._param_list = []
        self._neighbor_list = []
        self._neighbor_distance = []
        self._num_nonbonded_pairs = 0
        self._kernel = nb.njit((
            env.NUMBA_INT[:, :], env.NUMBA_FLOAT[:, :], env.NUMBA_FLOAT[:, ::1], 
            env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT
        ))(self.kernel)

    def __repr__(self) -> str:
        return '<mdpy.constraint.CharmmNonbondedConstraint object>'

    __str__ = __repr__

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._force_id = ensemble.constraints.index(self)
        self._param_list = []
        for particle in self._parent_ensemble.topology.particles:
            param = self._params[particle.particle_name]
            if len(param) == 2:
                epsilon, sigma = param
                self._param_list.append([epsilon, sigma, epsilon, sigma])
            elif len(param) == 4:
                epsilon, sigma, epsilon14, sigma14 = param
                self._param_list.append([epsilon, sigma, epsilon14, sigma14])
        self._num_nonbonded_pairs = int((
            self._parent_ensemble.topology.num_particles**2 -
            self._parent_ensemble.topology.num_particles
        ) / 2)

    def _mix_params(self, id1, id2, is_14=False):
        # Mix rule: 
        # - Eps,i,j = sqrt(eps,i * eps,j)
        # - Rmin,i,j = Rmin/2,i + Rmin/2,j
        # Turn r_min to sigma
        if is_14:
            epsilon1, sigma1 = self._param_list[id1][2:]
            epsilon2, sigma2 = self._param_list[id2][2:]
        else:
            epsilon1, sigma1 = self._param_list[id1][:2]
            epsilon2, sigma2 = self._param_list[id2][:2]
        return (
            np.sqrt(epsilon1 * epsilon2),
            (sigma1 + sigma2) / 2
        )

    def _update_neighbor(self):
        self._check_bound_state()
        self._neighbor_list, self._neighbor_distance = [], []
        scaled_position = np.dot(
            self._parent_ensemble.state.positions,
            self._parent_ensemble.state.pbc_inv
        )
        for particle in self._parent_ensemble.topology.particles:
            scaled_position_diff = scaled_position[particle.matrix_id, :] - scaled_position[particle.matrix_id+1:, :]
            scaled_position_diff -= np.round(scaled_position_diff)
            dist = np.sqrt(((np.dot(
                scaled_position_diff, 
                self._parent_ensemble.state.pbc_matrix
            ))**2).sum(1))
            index = np.argwhere(dist <= self._cutoff_radius).reshape(-1)
            self._neighbor_list.append(index + particle.matrix_id + 1)
            self._neighbor_distance.append(dist[index])

    @staticmethod
    def kernel(int_params, float_params, positions, pbc_matrix, pbc_inv, cutoff_radius):
        forces = np.zeros_like(positions)
        potential_energy = forces[0, 0]
        num_params = int_params.shape[0]
        for i in range(num_params):
            id1, id2 = int_params[i, :]
            force_vec = unwrap_vec(
                positions[id2] - positions[id1],
                pbc_matrix, pbc_inv
            )
            r = np.linalg.norm(force_vec)
            if r >= cutoff_radius:
                continue
            else:
                force_vec /= r
                epsilon, sigma, scaling_factor = float_params[i, :]
                scaled_r = sigma / r
                force_val = - (2 * scaled_r**12 - scaled_r**6) / r * epsilon * 24 # Sequence for small number divide small number
                force = scaling_factor * force_vec * force_val 
                forces[id1, :] += force
                forces[id2, :] -= force
                potential_energy += scaling_factor * 4 * epsilon * (scaled_r**12 - scaled_r**6)
        return forces, potential_energy

    def update(self):
        self._check_bound_state()
        RangePush('Nonbonded IO')
        params = []
        for particle in self._parent_ensemble.topology.particles:
            id1 = particle.matrix_id
            particle1 = self._parent_ensemble.topology.particles[id1]
            neighbors = self._parent_ensemble.state.cell_list.get_neighbors(
                self._parent_ensemble.state.positions[id1, :]
            )
            for i in particle1.bonded_particles:
                del neighbors[neighbors.index(i)]
            neighbors.sort()
            neighbors = neighbors[neighbors.index(id1)+1:]
            for id2 in neighbors:
                if id2 in particle1.scaling_particles:
                    scaling_factor = particle1.scaling_factors[particle.scaling_particles.index(id2)]
                    epsilon, sigma = self._mix_params(id1, id2, is_14=True)
                else:
                    scaling_factor = 1
                    epsilon, sigma = self._mix_params(id1, id2, is_14=False)
                params.append([id1, id2, epsilon, sigma, scaling_factor])
            
        params = np.vstack(params)
        self._int_params = params[:, :2].astype(env.NUMPY_INT)
        float_params = params[:, 2:].astype(env.NUMPY_FLOAT)
        RangePop()
        self._forces, self._potential_energy = self._kernel(
            self._int_params, float_params, 
            self._parent_ensemble.state.positions, 
            *self._parent_ensemble.state.pbc_info,
            self._cutoff_radius
        )

    @property
    def num_nonbonded_pairs(self):
        return self._num_nonbonded_pairs