#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : ensemble.py
created time : 2021/09/28
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import numpy as np
import cupy as cp
import numba.cuda as cuda
import mdpy as md
from mdpy.environment import *
from mdpy.core import Topology, State, TileList
from mdpy.error import *
from mdpy.unit import *


class Ensemble:
    def __init__(self, topology: Topology, pbc_matrix: np.ndarray) -> None:
        if not topology.is_joined:
            topology.join()
        # Read input
        self._topology = topology
        self._state = State(self._topology, pbc_matrix.copy())
        self._tile_list = TileList(pbc_matrix.copy())
        self._matrix_shape = self._state.matrix_shape
        self._forces = cp.zeros(self._matrix_shape)

        self._total_energy = 0
        self._potential_energy = 0
        self._kinetic_energy = 0
        self._constraints = []
        self._num_constraints = 0

    def __repr__(self) -> str:
        return "<mdpy.Ensemble object: %d constraints at %x>" % (
            self._num_constraints,
            id(self),
        )

    __str__ = __repr__

    def add_constraints(self, *constraints):
        for constraint in constraints:
            if constraint in self._constraints:
                raise ConstraintConflictError(
                    "%s has added twice to %s" % (constraint, self)
                )
            self._constraints.append(constraint)
            constraint.bind_ensemble(self)
            if constraint.cutoff_radius > self._tile_list.cutoff_radius:
                self._tile_list.set_cutoff_radius(constraint.cutoff_radius)
            self._num_constraints += 1

    def update_constraints(self):
        self._forces = cp.zeros(self._matrix_shape, CUPY_FLOAT)
        self._potential_energy = cp.zeros([1], CUPY_FLOAT)
        self._state.sorted_positions = self._tile_list.sort_matrix(
            self._state.positions
        )
        self._total_energy, self._kinetic_energy = 0, 0
        for constraint in self._constraints:
            constraint.update()
        cuda.synchronize()
        for constraint in self._constraints:
            self._forces += constraint.forces
            self._potential_energy += constraint.potential_energy
        self._update_kinetic_energy()
        self._total_energy = self._potential_energy + self._kinetic_energy

    def update_tile_list(self):
        self._tile_list.update(self._state.positions)
        # sort topology attribute
        self._topology.device_sorted_charges = self._tile_list.sort_matrix(
            self._topology.device_charges
        )
        self._topology.device_exclusion_map = (
            self._tile_list.generate_exclusion_mask_map(
                self._topology.device_excluded_particles
            )
        )
        # sort constraint attribute
        for constraint in self._constraints:
            constraint.sort_attributes()

    def _update_kinetic_energy(self):
        # Without reshape, the result of the first sum will be a 1d vector
        # , which will be a matrix after multiple with a 2d vector
        self._kinetic_energy = (
            (self._state.velocities**2).sum(1) * self._topology.device_masses[:, 0]
        ).sum() / 2

    @property
    def topology(self) -> Topology:
        return self._topology

    @property
    def state(self) -> State:
        return self._state

    @property
    def tile_list(self) -> TileList:
        return self._tile_list

    @property
    def constraints(self):
        return self._constraints

    @property
    def num_constraints(self) -> int:
        return self._num_constraints

    @property
    def forces(self) -> cp.ndarray:
        return self._forces

    @property
    def total_energy(self) -> float:
        return self._total_energy

    @property
    def potential_energy(self) -> float:
        return self._potential_energy

    @property
    def kinetic_energy(self) -> float:
        return self._kinetic_energy
