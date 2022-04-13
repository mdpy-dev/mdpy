#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : electrostatic_fdpbe_constraint.py
created time : 2022/04/11
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

from cmath import pi
import time
import math
import numpy as np
import numba as nb
import numba.cuda as cuda
from mdpy import env
from mdpy.core import Ensemble
from mdpy.constraint import Constraint
from mdpy.utils import *
from mdpy.unit import *
from mdpy.error import *

class ElectrostaticFDPEConstraint(Constraint):
    def __init__(self, grid_width=Quantity(0.5, angstrom), cavity_relative_permittivity=2) -> None:
        super().__init__()
        # Input
        self._grid_width = check_quantity_value(grid_width, default_length_unit)
        self._cavity_relative_permittivity = cavity_relative_permittivity
        # Attribute
        self._grid_volume = self._grid_width**3
        self._inner_grid_size = np.zeros([3], env.NUMPY_INT)
        self._total_grid_size = np.zeros([3], env.NUMPY_INT)
        self._k0 = 4 * np.pi * EPSILON0.value
        self._epsilon0 = EPSILON0.value
        # device attributes
        self._device_grid_width = cuda.to_device(np.array([self._grid_width], dtype=env.NUMPY_FLOAT))
        self._device_cavity_relative_permittivity = cuda.to_device(np.array([self._cavity_relative_permittivity], dtype=env.NUMPY_FLOAT))
        self._device_k0 = cuda.to_device(np.array([self._k0], dtype=env.NUMPY_FLOAT))
        self._device_epsilon0 = cuda.to_device(np.array([self._epsilon0], dtype=env.NUMPY_FLOAT))
        # Kernel
        self._update_coulombic_electric_potential_map = cuda.jit(nb.void(
            env.NUMBA_FLOAT[:, ::1], # positions
            env.NUMBA_FLOAT[::1], # charges
            env.NUMBA_INT[::1], # num_particles
            env.NUMBA_FLOAT[::1], # k0
            env.NUMBA_FLOAT[::1], # grid_width
            env.NUMBA_INT[::1], # inner_grid_size,
            env.NUMBA_FLOAT[::1], # cavity_relative_permittivity
            env.NUMBA_FLOAT[:, :, ::1] # direct_electric_potential_map
        ))(self._update_coulombic_electric_potential_map_kernel)
        self._update_reaction_field_electric_potential_map = cuda.jit(nb.void(
            env.NUMBA_FLOAT[:, :, ::1], # relative_permittivity_map
            env.NUMBA_FLOAT[::1], # cavity_relative_permittivity
            env.NUMBA_INT[::1], # inner_grid_size,
            env.NUMBA_FLOAT[::1], # grid_width
            env.NUMBA_FLOAT[:, :, ::1], # coulombic_electric_potential_map
            env.NUMBA_FLOAT[:, :, ::1], # reaction_field_electric_potential_map
        ))(self._update_reaction_field_electric_potential_map_kernel)

    def __repr__(self) -> str:
        return '<mdpy.constraint.ElectrostaticFDPEConstraint object>'

    def __str__(self) -> str:
        return 'FDPE electrostatic constraint'

    def _get_boundary_vec(self):
        pass

    def _get_epsilon_map(self):
        pass

    def _get_epsilon_derivative_map(self):
        pass

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._force_id = ensemble.constraints.index(self)
        self._pbc_diag = np.diagonal(self._parent_ensemble.state.pbc_matrix)
        self._total_grid_size = np.ceil(self._pbc_diag / self._grid_width).astype(env.NUMPY_INT) + 1
        self._inner_grid_size = self._total_grid_size - 2
        self._charges = self._parent_ensemble.topology.charges[:, 0]
        # if self._charges.sum() != 0:
        #     raise EnsemblePoorDefinedError(
        #         'mdpy.constraint.ElectrostaticFDPEConstraint is bound to a non-neutralized ensemble'
        #     )
        # device attributes
        self._device_num_particles = cuda.to_device(np.array(
            [self._parent_ensemble.topology.num_particles], dtype=env.NUMPY_INT
        ))
        self._device_inner_grid_size = cuda.to_device(self._inner_grid_size.astype(env.NUMPY_INT))
        self._device_charges = cuda.to_device(self._charges.astype(env.NUMPY_FLOAT))
        self._device_pbc_matrix = cuda.to_device(self._parent_ensemble.state.pbc_matrix)

    def set_relative_permittivity_map(self, relative_permittivity_map: np.ndarray) -> None:
        map_shape = relative_permittivity_map.shape
        for i, j in zip(self._inner_grid_size, map_shape):
            if i != j:
                raise ArrayDimError(
                    'Relative permittivity map requires a [%d, %d, %d] array' %(
                        self._inner_grid_size[0], self._inner_grid_size[1], self._inner_grid_size[2]
                    ) + ', while an [%d, %d, %d] array is provided' %(
                        map_shape[0], map_shape[1], map_shape[2]
                    )
                )
        self._relative_permittivity_map = relative_permittivity_map.astype(env.NUMPY_FLOAT)
        self._device_relative_permittivity_map = cuda.to_device(self._relative_permittivity_map)

    @staticmethod
    def _update_coulombic_electric_potential_map_kernel(
        positions, charges, num_particles, k,
        grid_width, inner_grid_size,
        cavity_relative_permittivity,
        direct_electric_potential_map
    ):
        grid_x, grid_y = cuda.grid(2)
        if grid_x >= inner_grid_size[0]:
            return None
        if grid_y >= inner_grid_size[1]:
            return None
        num_particles = num_particles[0]
        k = k[0]
        cavity_relative_permittivity = cavity_relative_permittivity[0]
        grid_width = grid_width[0]
        for particle_id in range(num_particles):
            particle_grid_x = positions[particle_id, 0] / grid_width
            particle_grid_y = positions[particle_id, 1] / grid_width
            particle_grid_z = positions[particle_id, 2] / grid_width
            dist_x = abs(particle_grid_x - grid_x) * grid_width
            dist_y = abs(particle_grid_y - grid_y) * grid_width
            dist_xy2 = dist_x**2 + dist_y**2
            for grid_z in range(inner_grid_size[2]):
                dist_z = abs(particle_grid_z - grid_z) * grid_width
                dist = math.sqrt(dist_xy2 + dist_z**2)
                if dist < 1e-3:
                    dist = grid_width
                electric_potential = charges[particle_id] / k / dist
                electric_potential /= cavity_relative_permittivity
                cuda.atomic.add(direct_electric_potential_map, (grid_x, grid_y, grid_z), electric_potential)

    @staticmethod
    def _update_reaction_field_electric_potential_map_kernel(
        relative_permittivity_map,
        cavity_relative_permittivity,
        inner_grid_size, grid_width,
        coulombic_electric_potential_map,
        reaction_field_electric_potential_map
    ):
        grid_x, grid_y = cuda.grid(2)
        if grid_x >= inner_grid_size[0]:
            return None
        if grid_y >= inner_grid_size[1]:
            return None
        cavity_relative_permittivity = cavity_relative_permittivity[0]
        grid_size_x = inner_grid_size[0]
        grid_size_y = inner_grid_size[1]
        grid_size_z = inner_grid_size[2]
        for grid_z in range(grid_size_z):
            new_val = 0.
            denominator = 0.
            self_relative_permittivity = relative_permittivity_map[grid_x, grid_y, grid_z]
            # Left
            if grid_x == 0:
                new_val += 0
                denominator += self_relative_permittivity
            else:
                relative_permittivity = 0.5 * (
                    relative_permittivity_map[grid_x-1, grid_y, grid_z] +
                    self_relative_permittivity
                )
                new_val += (
                    relative_permittivity *
                    reaction_field_electric_potential_map[grid_x-1, grid_y, grid_z]
                )
                new_val += (
                    (relative_permittivity - cavity_relative_permittivity) *
                    coulombic_electric_potential_map[grid_x-1, grid_y, grid_z]
                )
                denominator += relative_permittivity
            # Right
            if grid_x == grid_size_x - 1:
                new_val += 0
                denominator += self_relative_permittivity
            else:
                relative_permittivity = 0.5 * (
                    relative_permittivity_map[grid_x+1, grid_y, grid_z] +
                    self_relative_permittivity
                )
                new_val += (
                    relative_permittivity *
                    reaction_field_electric_potential_map[grid_x+1, grid_y, grid_z]
                )
                new_val += (
                    (relative_permittivity - cavity_relative_permittivity) *
                    coulombic_electric_potential_map[grid_x+1, grid_y, grid_z]
                )
                denominator += relative_permittivity
            # Back
            if grid_y == 0:
                new_val += 0
                denominator += self_relative_permittivity
            else:
                relative_permittivity = 0.5 * (
                    relative_permittivity_map[grid_x, grid_y-1, grid_z] +
                    self_relative_permittivity
                )
                new_val += (
                    relative_permittivity *
                    reaction_field_electric_potential_map[grid_x, grid_y-1, grid_z]
                )
                new_val += (
                    (relative_permittivity - cavity_relative_permittivity) *
                    coulombic_electric_potential_map[grid_x, grid_y-1, grid_z]
                )
                denominator += relative_permittivity
            # Front
            if grid_y == grid_size_y - 1:
                new_val += 0
                denominator += self_relative_permittivity
            else:
                relative_permittivity = 0.5 * (
                    relative_permittivity_map[grid_x, grid_y+1, grid_z] +
                    self_relative_permittivity
                )
                new_val += (
                    relative_permittivity *
                    reaction_field_electric_potential_map[grid_x, grid_y+1, grid_z]
                )
                new_val += (
                    (relative_permittivity - cavity_relative_permittivity) *
                    coulombic_electric_potential_map[grid_x, grid_y+1, grid_z]
                )
                denominator += relative_permittivity
            # Bottom
            if grid_z == 0:
                new_val += 0
                denominator += self_relative_permittivity
            else:
                relative_permittivity = 0.5 * (
                    relative_permittivity_map[grid_x, grid_y, grid_z-1] +
                    self_relative_permittivity
                )
                new_val += (
                    relative_permittivity *
                    reaction_field_electric_potential_map[grid_x, grid_y, grid_z-1]
                )
                new_val += (
                    (relative_permittivity - cavity_relative_permittivity) *
                    coulombic_electric_potential_map[grid_x, grid_y, grid_z-1]
                )
                denominator += relative_permittivity
            # Top
            if grid_z == grid_size_z - 1:
                new_val += 0
                denominator += self_relative_permittivity
            else:
                relative_permittivity = 0.5 * (
                    relative_permittivity_map[grid_x, grid_y, grid_z+1] +
                    self_relative_permittivity
                )
                new_val += (
                    relative_permittivity *
                    reaction_field_electric_potential_map[grid_x, grid_y, grid_z+1]
                )
                new_val += (
                    (relative_permittivity - cavity_relative_permittivity) *
                    coulombic_electric_potential_map[grid_x, grid_y, grid_z+1]
                )
                denominator += relative_permittivity
            # Other term
            new_val += 6 * (
                cavity_relative_permittivity *
                coulombic_electric_potential_map[grid_x, grid_y, grid_z]
            )
            new_val /= denominator
            new_val -= coulombic_electric_potential_map[grid_x, grid_y, grid_z]
            old_val = reaction_field_electric_potential_map[grid_x, grid_y, grid_z]
            cuda.atomic.add(
                reaction_field_electric_potential_map,
                (grid_x, grid_y, grid_z), 0.9*new_val - 0.9*old_val
            )

    def update(self):
        positive_positions = self._parent_ensemble.state.positions + self._pbc_diag / 2
        device_positions = cuda.to_device(positive_positions)
        thread_per_block = (16, 16)
        block_per_grid = (
            int(np.ceil(self._inner_grid_size[0] / thread_per_block[0])),
            int(np.ceil(self._inner_grid_size[1] / thread_per_block[1]))
        )
        # columbic electric potential
        device_coulombic_electric_potential_map = cuda.to_device(np.zeros(self._inner_grid_size, dtype=env.NUMPY_FLOAT))
        self._update_coulombic_electric_potential_map[thread_per_block, block_per_grid](
            device_positions, self._device_charges, self._device_num_particles,
            self._device_k0, self._device_grid_width, self._device_inner_grid_size,
            device_coulombic_electric_potential_map
        )
        # reaction field electric potential
        device_reaction_field_electric_potential = cuda.to_device(np.zeros(self._inner_grid_size, dtype=env.NUMPY_FLOAT))
        self._update_coulombic_electric_potential_map[thread_per_block, block_per_grid](
            device_positions,
            self._device_cavity_relative_permittivity,
            self._device_inner_grid_size,
            self._device_grid_width,
            device_coulombic_electric_potential_map,
            device_reaction_field_electric_potential
        )

    @property
    def inner_grid_size(self):
        return self._inner_grid_size

    @property
    def num_inner_grids(self):
        return self._inner_grid_size.prod()

    @property
    def total_grid_size(self):
        return self._total_grid_size

    @property
    def num_total_grids(self):
        return self._total_grid_size.prod()

    @property
    def grid_width(self):
        return self._grid_width

    @property
    def grid_volume(self):
        return self._grid_volume

import mdpy as md
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    topology = md.core.Topology()
    topology.add_particles([
        md.core.Particle(charge=-1),
        md.core.Particle(charge=10),
        md.core.Particle(charge=-1),
        md.core.Particle(charge=1),
    ])
    positions = np.array([
        [0., 0, 0],
        [-2, -2, 0],
        [1, -3, 0],
        [2, 2, 0],
    ])
    positions = positions
    ensemble = md.core.Ensemble(topology, np.eye(3) * 50)
    ensemble.state.set_positions(positions)
    # constraint
    constraint = ElectrostaticFDPEConstraint(0.5)
    ensemble.add_constraints(constraint)
    relative_permittivity_map = np.ones(constraint._inner_grid_size) * 2
    relative_permittivity_map[:35, :, :] = 80
    relative_permittivity_map[-33:, :, :] = 80
    constraint.set_relative_permittivity_map(relative_permittivity_map)
    # Visualization
    fig = plt.figure(figsize=[12, 18])
    x = np.linspace(
        ensemble.state.pbc_matrix[0, 0] / -2,
        ensemble.state.pbc_matrix[0, 0] / 2,
        constraint.total_grid_size[0], endpoint=True
    )[1:-1]
    y = np.linspace(
        ensemble.state.pbc_matrix[1, 1] / -2,
        ensemble.state.pbc_matrix[1, 1] / 2,
        constraint.total_grid_size[1], endpoint=True
    )[1:-1]
    X, Y = np.meshgrid(x, y)
    # update
    positive_positions = constraint._parent_ensemble.state.positions + constraint._pbc_diag / 2
    device_positions = cuda.to_device(positive_positions)
    thread_per_block = (16, 16)
    block_per_grid = (
        int(np.ceil(constraint._inner_grid_size[0] / thread_per_block[0])),
        int(np.ceil(constraint._inner_grid_size[1] / thread_per_block[1]))
    )

    s = time.time()
    device_coulombic_electric_potential_map = cuda.to_device(np.zeros(constraint._inner_grid_size, dtype=env.NUMPY_FLOAT))
    constraint._update_coulombic_electric_potential_map[thread_per_block, block_per_grid](
        device_positions, constraint._device_charges, constraint._device_num_particles,
        constraint._device_k0, constraint._device_grid_width, constraint._device_inner_grid_size,
        constraint._device_cavity_relative_permittivity,
        device_coulombic_electric_potential_map
    )
    coulombic_electric_potential_map = device_coulombic_electric_potential_map.copy_to_host()
    e = time.time()
    print('Run coulombic electric potential for %s s' %(e-s))

    device_reaction_filed_electric_potential_map = cuda.to_device(np.zeros(constraint._inner_grid_size, dtype=env.NUMPY_FLOAT))
    s = time.time()
    for i in range(300):
        constraint._update_reaction_field_electric_potential_map[thread_per_block, block_per_grid](
            constraint._device_relative_permittivity_map,
            constraint._device_cavity_relative_permittivity,
            constraint._device_inner_grid_size,
            constraint._device_grid_width,
            device_coulombic_electric_potential_map,
            device_reaction_filed_electric_potential_map
        )
    reaction_filed_electric_potential_map = device_reaction_filed_electric_potential_map.copy_to_host()
    e = time.time()
    print('Run reaction field electric potential for %s s' %(e-s))

    ax1 = fig.add_subplot(211)
    c = ax1.contourf(X, Y, coulombic_electric_potential_map[:, :, constraint.total_grid_size[1]//2].T, 100, cmap='RdBu')
    plt.colorbar(c)

    ax2 = fig.add_subplot(212)
    c = ax2.contourf(X, Y, reaction_filed_electric_potential_map[:, :, constraint.total_grid_size[1]//2].T, 100, cmap='RdBu')
    plt.colorbar(c)
    plt.show()