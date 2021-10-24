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
    def __init__(self, params, cutoff_radius=12, force_id: int = 0, force_group: int = 0) -> None:
        super().__init__(params, force_id=force_id, force_group=force_group)
        self._cutoff_radius = check_quantity_value(cutoff_radius, default_length_unit)
        self._param_list = []
        self._neighbor_list = []
        self._neighbor_distance = []
        self._num_nonbonded_pairs = 0

    def __repr__(self) -> str:
        return '<mdpy.constraint.CharmmNonbondedConstraint object>'

    __str__ = __repr__

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._force_id = ensemble.constraints.index(self)
        self._param_list = []
        for particle in self._parent_ensemble.topology.particles:
            epsilon, sigma = self._params[particle.particle_name]
            self._param_list.append([epsilon, sigma])
        self._num_nonbonded_pairs = int((
            self._parent_ensemble.topology.num_particles**2 -
            self._parent_ensemble.topology.num_particles
        ) / 2)

    def _mix_params(self, id1, id2):
        # Mix rule: 
        # - Eps,i,j = sqrt(eps,i * eps,j)
        # - Rmin,i,j = Rmin/2,i + Rmin/2,j
        # Turn r_min to sigma
        epsilon1, sigma1 = self._param_list[id1]
        epsilon2, sigma2 = self._param_list[id2]
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

    def update(self):
        self._check_bound_state()
        self._update_neighbor()
        self._forces = np.zeros([self._parent_ensemble.topology.num_particles, SPATIAL_DIM])
        self._potential_energy = 0
        # V(Lennard-Jones) = Eps,i,j[(Rmin,i,j/ri,j)**12 - 2(Rmin,i,j/ri,j)**6]
        for particle in self._parent_ensemble.topology.particles:
            id1 = particle.matrix_id
            particle1 = self._parent_ensemble.topology.particles[id1]
            for i, id2 in enumerate(self._neighbor_list[id1]):
                if not id2 in particle1.bonded_particles:
                    epsilon, sigma = self._mix_params(id1, id2)
                    r = self._neighbor_distance[id1][i]
                    scaled_r = sigma / r
                    force_val = - (2 * scaled_r**12 - scaled_r**6) / r * epsilon * 24 # Sequence for small number divide small number
                    force_vec = unwrap_vec(get_unit_vec(
                        self._parent_ensemble.state.positions[id2] - 
                        self._parent_ensemble.state.positions[id1]
                    ), *self._parent_ensemble.state.pbc_info)
                    if id2 in particle1.scaling_particles:
                        scaling_factor = particle1.scaling_factors[particle.scaling_particles.index(id2)]
                    else:
                        scaling_factor = 1
                    force = scaling_factor * force_vec * force_val
                    self._forces[id1, :] += force
                    self._forces[id2, :] -= force
                    self._potential_energy += scaling_factor * 4 * epsilon * (scaled_r**12 - scaled_r**6) 

    @property
    def num_nonbonded_pairs(self):
        return self._num_nonbonded_pairs

    @property
    def cutoff_radius(self):
        return self._cutoff_radius

    @cutoff_radius.setter
    def cutoff_radius(self, val):
        self._cutoff_radius = check_quantity_value(val, default_length_unit)