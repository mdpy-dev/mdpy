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
from mdpy import env, SPATIAL_DIM
from mdpy.core import Ensemble
from mdpy.constraint import Constraint
from mdpy.utils import *
from mdpy.unit import *
from mdpy.error import *

BSPLINE_ORDER = 4

class ElectrostaticFDPEConstraint(Constraint):
    def __init__(self, grid_width=Quantity(0.5, angstrom)) -> None:
        super().__init__()
        # Input
        self._grid_width = check_quantity_value(grid_width, default_length_unit)
        self._grid_volume = self._grid_width**3
        # Attribute
        self._inner_grid_size = np.zeros([3], env.NUMPY_INT)
        self._total_grid_size = np.zeros([3], env.NUMPY_INT)
        self._k0 = 4 * np.pi * EPSILON0.value
        self._epsilon0 = EPSILON0.value
        # device attributes
        self._device_grid_width = cuda.to_device(np.array([self._grid_width], dtype=env.NUMPY_FLOAT))
        self._device_k0 = cuda.to_device(np.array([self._k0], dtype=env.NUMPY_FLOAT))
        self._device_epsilon0 = cuda.to_device(np.array([self._epsilon0], dtype=env.NUMPY_FLOAT))
        # Kernel
        self._update_bspline = self._update_bspline_kernel
        self._update_charge_map = cuda.jit(nb.void(
            env.NUMBA_INT[::1], # num_particles
            env.NUMBA_FLOAT[:, :, ::1], # spline_coefficient
            env.NUMBA_INT[:, :, ::1], # grid_map
            env.NUMBA_FLOAT[::1], # charges
            env.NUMBA_FLOAT[:, :, ::1] # charge_map
        ))(self._update_charge_map_kernel)
        self._update_coulombic_electric_potential_map = cuda.jit(nb.void(
            env.NUMBA_FLOAT[:, ::1], # positions
            env.NUMBA_FLOAT[::1], # charges
            env.NUMBA_INT[::1], # num_particles
            env.NUMBA_FLOAT[::1], # k0
            env.NUMBA_FLOAT[::1], # grid_width
            env.NUMBA_INT[::1], # inner_grid_size
            env.NUMBA_FLOAT[:, :, ::1] # direct_electric_potential_map
        ))(self._update_coulombic_electric_potential_map_kernel)
        self._update_reaction_field_electric_potential = cuda.jit(nb.void(
            env.NUMBA_FLOAT[:, :, ::1], # charge_map
            # env.NUMBA_FLOAT[:, :, ::1], # epsilon_map
            # env.NUMBA_FLOAT[:, ::1], # boundary_vec
            env.NUMBA_INT[::1], # inner_grid_size
            env.NUMBA_FLOAT[::1], # grid_width
            env.NUMBA_FLOAT[::1], # epsilon0
            env.NUMBA_FLOAT[:, :, ::1]# electric_potential_map
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

    @staticmethod
    def _update_bspline_kernel(positions, total_grid_size, pbc_matrix, pbc_inv):
        num_particles = positions.shape[0]
        positions = positions + np.diagonal(pbc_matrix) / 2
        # spline_coefficient: [num_particles, SPATIAL_DIM, PME_ORDER] The spline coefficient of particles
        spline_coefficient = np.zeros((num_particles, SPATIAL_DIM, BSPLINE_ORDER))
        # spline_derivative_coefficient: [num_particles, SPATIAL_DIM, PME_ORDER] The derivative spline coefficient of particles
        spline_derivative_coefficient = np.zeros((num_particles, SPATIAL_DIM, BSPLINE_ORDER))
        scaled_positions = np.dot(positions, pbc_inv)
        # grid_indice: [num_particles, SPATIAL_DIM] The index of grid where particle assigned to
        grid_indice = scaled_positions * total_grid_size
        # grid_friction: [num_particles, SPATIAL_DIM] The fraction part of particles position relative to gird
        grid_fraction = grid_indice - np.floor(grid_indice)
        grid_indice -= grid_fraction

        # 3 order B-spline
        spline_coefficient[:, :, 2] = 0.5 * grid_fraction**2
        spline_coefficient[:, :, 0] = 0.5 * (1 - grid_fraction)**2
        spline_coefficient[:, :, 1] = 1 - spline_coefficient[:, :, 2] - spline_coefficient[:, :, 0]

        # 4 order derivative coefficient
        spline_derivative_coefficient[:, :, 0] = - spline_coefficient[:, :, 0]
        spline_derivative_coefficient[:, :, 1] = spline_coefficient[:, :, 0] - spline_coefficient[:, :, 1]
        spline_derivative_coefficient[:, :, 2] = spline_coefficient[:, :, 1] - spline_coefficient[:, :, 2]
        spline_derivative_coefficient[:, :, 3] = spline_coefficient[:, :, 2]
        # 4 order spline coefficient
        spline_coefficient[:, :, 3] = grid_fraction * spline_coefficient[:, :, 2] / 3
        spline_coefficient[:, :, 2] = (
            (1 + grid_fraction) * spline_coefficient[:, :, 1] +
            (3 - grid_fraction) * spline_coefficient[:, :, 2]
        ) / 3
        spline_coefficient[:, :, 0] = (1 - grid_fraction) * spline_coefficient[:, :, 0] / 3
        spline_coefficient[:, :, 1] = (
            1 - spline_coefficient[:, :, 0] -
            spline_coefficient[:, :, 2] -
            spline_coefficient[:, :, 3]
        )
        # grid_map: [num_particles, SPATIL_DIM, PME_ORDER]: The indice of grid to add charge of each particles
        grid_map = np.stack([grid_indice+i for i in [-1, 0, 1, 2]]).transpose(1, 2, 0)
        for i in range(SPATIAL_DIM):
            cur_axis_map = grid_map[:, i, :]
            cur_axis_map[cur_axis_map<0] += total_grid_size[i]
            cur_axis_map[cur_axis_map>=total_grid_size[i]] -= total_grid_size[i]
            grid_map[:, i, :] = cur_axis_map
        return (
            (spline_coefficient).astype(env.NUMPY_FLOAT),
            (spline_derivative_coefficient).astype(env.NUMPY_FLOAT),
            np.ascontiguousarray(grid_map.astype(env.NUMPY_INT))
        )

    @staticmethod
    def _update_charge_map_kernel(num_particles, spline_coefficient, grid_map, charges, charge_map):
        '''
        spline_coefficient: [num_particles, SPATIAL_DIM, PME_ORDER] The spline coefficient of particles
        grid_map: [num_particles, SPATIL_DIM, PME_ORDER]: The indice of grid to add charge of each particles
        charges: [num_particles,]: The charges of particles
        charge_map: [grid_size_x, grid_size_y, grid_size_z]
        '''
        particle_id = cuda.grid(1)
        num_particles = num_particles[0]
        if particle_id >= num_particles:
            return None
        charge = charges[particle_id]
        for i in range(BSPLINE_ORDER):
            grid_x = grid_map[particle_id, 0, i]
            spline_coefficient_x = spline_coefficient[particle_id, 0, i]
            for j in range(BSPLINE_ORDER):
                grid_y = grid_map[particle_id, 1, j]
                spline_coefficient_y = spline_coefficient[particle_id, 1, j]
                for k in range(BSPLINE_ORDER):
                    grid_z = grid_map[particle_id, 2, k]
                    spline_coefficient_z = spline_coefficient[particle_id, 2, k]
                    grid_charge = (
                        charge * spline_coefficient_x *
                        spline_coefficient_y * spline_coefficient_z
                    )
                    cuda.atomic.add(charge_map, (grid_x, grid_y, grid_z), grid_charge)

    @staticmethod
    def _update_coulombic_electric_potential_map_kernel(
        positions, charges, num_particles, k,
        grid_width, inner_grid_size,
        direct_electric_potential_map
    ):
        grid_x, grid_y = cuda.grid(2)
        if grid_x >= inner_grid_size[0]:
            return None
        if grid_y >= inner_grid_size[1]:
            return None
        num_particles = num_particles[0]
        k = k[0]
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
                electric_potential = charges[particle_id] / k / dist
                cuda.atomic.add(direct_electric_potential_map, (grid_x, grid_y, grid_z), electric_potential)

    @staticmethod
    def _update_reaction_field_electric_potential_map_kernel(
        charge_map,
        inner_grid_size, grid_width, epsilon0,
        electric_potential_map
    ):
        grid_x, grid_y = cuda.grid(2)
        if grid_x >= inner_grid_size[0]:
            return None
        if grid_y >= inner_grid_size[1]:
            return None
        grid_width = grid_width[0]
        epsilon0 = epsilon0[0]
        for grid_z in range(inner_grid_size[2]):
            if grid_x == 0:
                left = 0
            else:
                left = electric_potential_map[grid_x-1, grid_y, grid_z]
            if grid_x == inner_grid_size[0] - 1:
                right = 0
            else:
                right = electric_potential_map[grid_x+1, grid_y, grid_z]

            if grid_y == 0:
                back = 0
            else:
                back = electric_potential_map[grid_x, grid_y-1, grid_z]
            if grid_y == inner_grid_size[1] - 1:
                front = 0
            else:
                front = electric_potential_map[grid_x, grid_y+1, grid_z]

            if grid_z == 0:
                bottom = 0
            else:
                bottom = electric_potential_map[grid_x, grid_y, grid_z-1]
            if grid_z == inner_grid_size[2] - 1:
                top = 0
            else:
                top = electric_potential_map[grid_x, grid_y, grid_z+1]

            new_val = (
                left + right +
                back + front +
                bottom + top +
                charge_map[grid_x, grid_y, grid_z] / grid_width / epsilon0
            ) / 6
            old_val = electric_potential_map[grid_x, grid_y, grid_z]
            # cuda.atomic.add(electric_potential_map, (grid_x, grid_y, grid_z), new_val - old_val)
            cuda.atomic.add(electric_potential_map, (grid_x, grid_y, grid_z), 0.95 * new_val - 0.95*old_val)


    def update(self):
        # Bspline
        spline_coefficient, spline_derivative_coefficient, grid_map = self._update_bspline(
            self._parent_ensemble.state.positions, self._total_grid_size,
            self._parent_ensemble.state.pbc_matrix, self._parent_ensemble.state.pbc_inv
        )
        device_spline_coefficient = cuda.to_device(spline_coefficient)
        device_grid_map = cuda.to_device(grid_map)
        # Map charge
        thread_per_block = 512
        block_per_grid = int(np.ceil(
            self._parent_ensemble.topology.num_particles / thread_per_block
        ))
        device_num_particles = cuda.to_device(np.array(
            [self._parent_ensemble.topology.num_particles], dtype=env.NUMPY_INT
        ))
        device_charge_map = cuda.to_device(np.zeros(self._total_grid_size, dtype=env.NUMPY_FLOAT))
        self._update_charge_map[thread_per_block, block_per_grid](
            device_num_particles, device_spline_coefficient, device_grid_map,
            self._device_charges, device_charge_map
        )
        # Solve equation
        device_electric_potential = cuda.to_device(np.zeros(self._total_grid_size, dtype=env.NUMPY_FLOAT))
        thread_per_block = (8, 8, 8)
        block_per_grid = (
            int(np.ceil(self._inner_grid_size[0] / thread_per_block[0])),
            int(np.ceil(self._inner_grid_size[1] / thread_per_block[1])),
            int(np.ceil(self._inner_grid_size[2] / thread_per_block[2]))
        )
        self._update_electric_potential_map[thread_per_block, block_per_grid](
            device_charge_map,
            self._device_inner_grid_size, self._device_grid_width, self._device_k,
            device_electric_potential
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
        md.core.Particle(charge=-1),
        md.core.Particle(charge=1),
        md.core.Particle(charge=1),
    ])
    positions = np.array([
        [-2, 0, 0],
        [2, 0, 0],
        [0, -2, 0],
        [0, 2, 0],
    ])
    ensemble = md.core.Ensemble(topology, np.eye(3) * 50)
    ensemble.state.set_positions(positions)
    # constraint
    constraint = ElectrostaticFDPEConstraint(0.5)
    ensemble.add_constraints(constraint)
    # Visualization
    fig = plt.figure(figsize=[12, 18])
    x = np.linspace(
        ensemble.state.pbc_matrix[0, 0] / -2,
        ensemble.state.pbc_matrix[0, 0]  / 2,
        constraint.inner_grid_size[0]
    )
    y = np.linspace(
        ensemble.state.pbc_matrix[1, 1] / -2,
        ensemble.state.pbc_matrix[1, 1]  / 2,
        constraint.inner_grid_size[1]
    )
    X, Y = np.meshgrid(x, y)
    # update
    s = time.time()
    positive_positions = constraint._parent_ensemble.state.positions + constraint._pbc_diag / 2
    device_positions = cuda.to_device(positive_positions)
    device_electric_potential_map = cuda.to_device(np.zeros(constraint._inner_grid_size, dtype=env.NUMPY_FLOAT))
    thread_per_block = (16, 16)
    block_per_grid = (
        int(np.ceil(constraint._inner_grid_size[0] / thread_per_block[0])),
        int(np.ceil(constraint._inner_grid_size[1] / thread_per_block[1]))
    )
    constraint._update_coulombic_electric_potential_map[thread_per_block, block_per_grid](
        device_positions, constraint._device_charges, constraint._device_num_particles,
        constraint._device_k0, constraint._device_grid_width, constraint._device_inner_grid_size,
        device_electric_potential_map
    )
    e = time.time()
    print('Run direct for %s s' %(e-s))
    direct_electric_potential_map = device_electric_potential_map.copy_to_host()
    ax2 = fig.add_subplot(211)
    c = ax2.contourf(X, Y, direct_electric_potential_map[:, :, constraint.total_grid_size[1]//2].T, 100, cmap='RdBu')
    plt.colorbar(c)

    # Bspline
    s = time.time()
    spline_coefficient, spline_derivative_coefficient, grid_map = constraint._update_bspline(
        constraint._parent_ensemble.state.positions, constraint._total_grid_size,
        constraint._parent_ensemble.state.pbc_matrix, constraint._parent_ensemble.state.pbc_inv
    )
    device_spline_coefficient = cuda.to_device(spline_coefficient)
    device_grid_map = cuda.to_device(grid_map)
    # Map charge
    thread_per_block = 512
    block_per_grid = int(np.ceil(
        constraint._parent_ensemble.topology.num_particles / thread_per_block
    ))
    device_num_particles = cuda.to_device(np.array(
        [constraint._parent_ensemble.topology.num_particles], dtype=env.NUMPY_INT
    ))
    device_charge_map = cuda.to_device(np.zeros(constraint._total_grid_size, dtype=env.NUMPY_FLOAT))
    constraint._update_charge_map[thread_per_block, block_per_grid](
        device_num_particles, device_spline_coefficient, device_grid_map,
        constraint._device_charges, device_charge_map
    )
    # Solve equation
    device_electric_potential_map = cuda.to_device(np.zeros(constraint._inner_grid_size, dtype=env.NUMPY_FLOAT))
    thread_per_block = (16, 16)
    block_per_grid = (
        int(np.ceil(constraint._inner_grid_size[0] / thread_per_block[0])),
        int(np.ceil(constraint._inner_grid_size[1] / thread_per_block[1]))
    )
    cur_map = np.zeros(constraint._inner_grid_size, dtype=env.NUMPY_FLOAT)
    for i in range(200):
        constraint._update_reaction_field_electric_potential[thread_per_block, block_per_grid](
            device_charge_map,
            constraint._device_inner_grid_size, constraint._device_grid_width, constraint._device_epsilon0,
            device_electric_potential_map
        )
        cur_map, pre_map = device_electric_potential_map.copy_to_host(), cur_map
        print(np.sqrt((cur_map - pre_map)**2).mean())
    e = time.time()
    print('Run update for %s s' %(e-s))
    # Electric potential map
    reaction_field_electric_potential = device_electric_potential_map.copy_to_host()
    ax2 = fig.add_subplot(212)
    c = ax2.contourf(X, Y, reaction_field_electric_potential[:, :, constraint.total_grid_size[1]//2].T, 100, cmap='RdBu')
    plt.colorbar(c)
    plt.show()

    # print((np.abs(direct_electric_potential_map - reaction_field_electric_potential)).std())


    # Visualization
    # # Charge map
    # charge_map = device_charge_map.copy_to_host()
    # ax1 = fig.add_subplot(211)
    # c = ax1.contourf(X, Y, charge_map[1:-1, 1:-1, constraint.total_grid_size[1]//2].T, 100, cmap='RdBu')
    # plt.colorbar(c)
    # e = time.time()