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

import numpy as np
from . import Constraint
from .. import SPATIAL_DIM
from ..ensemble import Ensemble
from ..math import *
from ..unit import *

RMIN_TO_SIGMA_FACTOR = 2**(-1/6)

class CharmmNonbondedConstraint(Constraint):
    def __init__(self, cutoff_radius=12, force_id: int = 0, force_group: int = 0) -> None:
        super().__init__(force_id=force_id, force_group=force_group)
        self._cutoff_radius = check_quantity_value(cutoff_radius, default_length_unit)
        self._nonbonded_pair_type, self._nonbonded_matrix_id = [], []
        self._nonbonded_pair_info = {}
        self._neighbor_list = []
        self._neighbor_distance = []
        self._num_nonbonded_pairs = 0

    def bind_ensemble(self, ensemble: Ensemble):
        ensemble.topology.check_pbc_matrix()
        ensemble.add_constraints(self)
        self._nonbonded_pair_type, self._nonbonded_matrix_id = [], []
        self._num_nonbonded_pairs = 0
        for index, particle1 in enumerate(self._parent_ensemble.topology.particles):
            for particle2 in self._parent_ensemble.topology.particles[index+1:]:
                self._nonbonded_pair_type.append([
                    particle1.particle_name, 
                    particle2.particle_name
                ])
                self._nonbonded_matrix_id.append([
                    particle1.matrix_id, 
                    particle2.matrix_id
                ])
                self._num_nonbonded_pairs += 1

    def set_params(self, params):
        self._check_bound_state()
        self._nonbonded_pair_info = {}
        for index, nonbonded_pair in enumerate(self._nonbonded_pair_type):
            id1, id2 = self._nonbonded_matrix_id[index]
            epsilon1, half_rmin1 = params[nonbonded_pair[0]]
            epsilon2, half_rmin2 = params[nonbonded_pair[1]]
            
            # Mix rule: 
            # - Eps,i,j = sqrt(eps,i * eps,j)
            # - Rmin,i,j = Rmin/2,i + Rmin/2,j
            # Turn r_min to sigma
            self._nonbonded_pair_info['%s-%s' %(id1, id2)] = [
                np.sqrt(epsilon1 * epsilon2), (half_rmin1 + half_rmin2) * RMIN_TO_SIGMA_FACTOR
            ]

    def update_neighbor(self):
        self._check_bound_state()
        self._neighbor_list, self._neighbor_distance = [], []
        scaled_position = np.dot(
            self._parent_ensemble.positions,
            self._parent_ensemble.topology.pbc_inv
        )
        for particle in self._parent_ensemble.topology.particles:
            scaled_position_diff = scaled_position[particle.matrix_id] - scaled_position[particle.matrix_id+1:]
            scaled_position_diff -= np.round(scaled_position_diff)
            dist = np.sqrt(((np.dot(
                scaled_position_diff, 
                self._parent_ensemble.topology.pbc_matrix
            ))**2).sum(1))
            index = np.argwhere(dist <= self._cutoff_radius).reshape(-1)
            self._neighbor_list.append(index + particle.matrix_id + 1)
            self._neighbor_distance.append(dist[index])

    def get_forces(self):
        self._check_bound_state()
        # V(Lennard-Jones) = Eps,i,j[(Rmin,i,j/ri,j)**12 - 2(Rmin,i,j/ri,j)**6]
        forces = np.zeros([self._parent_ensemble.topology.num_particles, SPATIAL_DIM])
        for particle in self._parent_ensemble.topology.particles:
            id1 = particle.matrix_id
            for i, id2 in enumerate(self._neighbor_list[id1]):
                epsilon, sigma = self._nonbonded_pair_info['%s-%s' %(id1, id2)]
                r = self._neighbor_distance[id1][i]
                scaled_r = r / sigma
                force_val = 24 * epsilon / r * (2 * scaled_r**12 - scaled_r**6)
                force_vec = get_unit_vec(self._parent_ensemble.positions[id2] - self._parent_ensemble.positions[id1])
                force = force_vec * force_val
                forces[id1, :] += force
                forces[id2, :] -= force
        return forces


    def get_potential_energy(self):
        self._check_bound_state()
        # V(Lennard-Jones) = Eps,i,j[(Rmin,i,j/ri,j)**12 - 2(Rmin,i,j/ri,j)**6]
        potential_energy = 0
        for particle in self._parent_ensemble.topology.particles:
            id1 = particle.matrix_id
            for i, id2 in enumerate(self._neighbor_list[id1]):
                epsilon, sigma = self._nonbonded_pair_info['%s-%s' %(id1, id2)]
                r = self._neighbor_distance[id1][i]
                scaled_r = r / sigma
                potential_energy += 4 * epsilon * (scaled_r**12 - scaled_r**6) 
        return potential_energy

    @property
    def num_nonbonded_pairs(self):
        return self._num_nonbonded_pairs

    @property
    def cutoff_radius(self):
        return self._cutoff_radius

    @cutoff_radius.setter
    def cutoff_radius(self, val):
        self._cutoff_radius = check_quantity_value(val, default_length_unit)