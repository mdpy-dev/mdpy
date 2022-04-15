#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : electrostatic_fdpbe_constraint.py
created time : 2022/04/11
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

from signal import pthread_kill
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

THREAD_PER_BLOCK = (16, 16)

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
            env.NUMBA_INT[::1], # inner_grid_size
            env.NUMBA_FLOAT[:, :, ::1], # coulombic_electric_potential_map
            env.NUMBA_FLOAT[:, :, ::1], # reaction_field_electric_potential_map
        ))(self._update_reaction_field_electric_potential_map_kernel)
        self._update_coulombic_force_and_potential_energy = cuda.jit(nb.void(
            env.NUMBA_FLOAT[:, ::1], # positions
            env.NUMBA_FLOAT[::1], # charges
            env.NUMBA_INT[::1], # num_particles
            env.NUMBA_FLOAT[::1], # grid_width
            env.NUMBA_FLOAT[::1], # k0
            env.NUMBA_FLOAT[::1], # cavity_relative_permittivity
            env.NUMBA_FLOAT[:, ::1], # forces
            env.NUMBA_FLOAT[::1] # potential_energy
        ))(self._update_coulombic_force_and_potential_energy_kernel)
        self._update_reaction_field_force_and_potential_energy = nb.njit((
            env.NUMBA_FLOAT[:, ::1], # positions
            env.NUMBA_FLOAT[::1], # charges
            env.NUMBA_FLOAT[::1], # grid_width
            env.NUMBA_FLOAT[:, :, ::1], # reaction_field_electric_potential_map
        ))

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
            # positions[particle_id, x] / grid_width is the grid id of total grid
            particle_grid_x = positions[particle_id, 0] / grid_width - 1
            particle_grid_y = positions[particle_id, 1] / grid_width - 1
            particle_grid_z = positions[particle_id, 2] / grid_width - 1
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
        inner_grid_size,
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
                (grid_x, grid_y, grid_z), 0.85*new_val - 0.85*old_val
            )

    @staticmethod
    def _update_coulombic_force_and_potential_energy_kernel(
        positions, charges, num_particles, grid_width,
        k0, cavity_relative_permittivity,
        forces, potential_energy
    ):
        particle_id1, particle_id2 = cuda.grid(2)
        num_particles = num_particles[0]
        if particle_id1 >= num_particles:
            return None
        if particle_id2 >= num_particles:
            return None
        grid_width = grid_width[0]
        k0 = k0[0]
        cavity_relative_permittivity = cavity_relative_permittivity[0]
        factor = k0 * cavity_relative_permittivity
        e1 = charges[particle_id1]
        e2 = charges[particle_id2]
        x = (positions[particle_id2, 0] - positions[particle_id1, 0])
        y = (positions[particle_id2, 1] - positions[particle_id1, 1])
        z = (positions[particle_id2, 2] - positions[particle_id1, 2])
        r = math.sqrt(x**2 + y**2 + z**2)
        scaled_x, scaled_y, scaled_z = x / r, y / r, z / r
        force_val = - e1 * e2 / factor / r**2
        force_x = scaled_x * force_val / 2
        force_y = scaled_y * force_val / 2
        force_z = scaled_z * force_val / 2
        cuda.atomic.add(forces, (particle_id1, 0), force_x)
        cuda.atomic.add(forces, (particle_id1, 1), force_y)
        cuda.atomic.add(forces, (particle_id1, 2), force_z)
        cuda.atomic.add(forces, (particle_id2, 0), -force_x)
        cuda.atomic.add(forces, (particle_id2, 1), -force_y)
        cuda.atomic.add(forces, (particle_id2, 2), -force_z)
        energy = e1 * e2 / factor / r / 2
        cuda.atomic.add(potential_energy, 0, energy)

    @staticmethod
    def _update_reaction_field_force_and_potential_energy_kernel(
        positions, charges, grid_width,
        reaction_field_electric_potential_map
    ):
        '''
        Least square solution:
        f(x, y, z) = a0x^2 + a1y^2 + a2z^2 + a3xy + a4xz + a5yz + a6x + a7y + a8z + a9

        Solution:
        A^TA a = A^Tb
        '''
        forces = np.zeros_like(positions)
        potential_energy = 0
        for particle in range(positions.shape[0]):
            grid_index = positions[particle, :] / grid_width - 1
            grid_index_int = np.floor(grid_index).astype(np.int32)
            grid_index_float = grid_index - grid_index_int
            A = np.zeros([64, 10])
            b = np.zeros([64, 1])
            index = 0
            for i, int_x in enumerate(range(-1, 3)):
                for j, int_y in enumerate(range(-1, 3)):
                    for k, int_z in enumerate(range(-1, 3)):
                        x = int_x - grid_index_float[0]
                        y = int_y - grid_index_float[1]
                        z = int_z - grid_index_float[2]
                        A[index, :] = [
                            x**2, y**2, z**2, x*y, x*z, y*z, x, y, z, 1
                        ]
                        b[index, 0] = reaction_field_electric_potential_map[
                            int_x+grid_index_int[0],
                            int_y+grid_index_int[1],
                            int_z+grid_index_int[2]
                        ]
                        index += 1
            a = np.linalg.solve(A.T.dot(A), A.T.dot(b))
            potential_energy += charges[particle] * (
                a[0]*grid_index_float[0]**2 + a[1]*grid_index_float[1]**2 + a[2]*grid_index_float[2]**2 +
                a[3]*grid_index_float[0]*grid_index_float[1] +
                a[4]*grid_index_float[0]*grid_index_float[2] +
                a[5]*grid_index_float[1]*grid_index_float[2] +
                a[6]*grid_index_float[0] + a[7]*grid_index_float[1] + a[8]*grid_index_float[2] + a[9]
            )
            forces[particle, 0] = -charges[particle] * (
                2*a[0] + a[3]*grid_index_float[1] + a[4]*grid_index_float[2] + a[6]
            )
            forces[particle, 1] = -charges[particle] * (
                2*a[1] + a[3]*grid_index_float[0] + a[5]*grid_index_float[2] + a[7]
            )
            forces[particle, 2] = -charges[particle] * (
                2*a[2] + a[4]*grid_index_float[0] + a[5]*grid_index_float[1] + a[8]
            )
        return forces, potential_energy

    def update(self):
        positive_positions = self._parent_ensemble.state.positions + self._pbc_diag / 2
        device_positions = cuda.to_device(positive_positions)
        # Columbic electric potential map
        device_coulombic_electric_potential_map = cuda.to_device(np.zeros(self._inner_grid_size, dtype=env.NUMPY_FLOAT))
        self._update_coulombic_electric_potential_map[block_per_grid, thread_per_block](
            device_positions, self._device_charges, self._device_num_particles,
            self._device_k0, self._device_grid_width, self._device_inner_grid_size,
            self._device_cavity_relative_permittivity,
            device_coulombic_electric_potential_map
        )
        # Reaction field potential map
        device_reaction_filed_electric_potential_map = cuda.to_device(np.zeros(self._inner_grid_size, dtype=env.NUMPY_FLOAT))
        for _ in range(500):
            constraint._update_reaction_field_electric_potential_map[block_per_grid, thread_per_block](
                constraint._device_relative_permittivity_map,
                constraint._device_cavity_relative_permittivity,
                constraint._device_inner_grid_size,
                device_coulombic_electric_potential_map,
                device_reaction_filed_electric_potential_map
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
        md.core.Particle(charge=1),
        # md.core.Particle(charge=-1),
        # md.core.Particle(charge=1),
    ])
    positions = np.array([
        [0., 2, 0],
        [0, -2, 0],
        # [0, -4, 0],
        # [0, 4, 0],
    ])
    positions = positions
    ensemble = md.core.Ensemble(topology, np.eye(3) * 50)
    ensemble.state.set_positions(positions)
    # constraint
    constraint = ElectrostaticFDPEConstraint(0.5)
    ensemble.add_constraints(constraint)
    relative_permittivity_map = np.ones(constraint._inner_grid_size) * 2
    relative_permittivity_map[:45, :, :] = 180
    # relative_permittivity_map[-45:, :, :] = 2
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
    print(x.shape)
    X, Y = np.meshgrid(x, y)
    # update
    positive_positions = constraint._parent_ensemble.state.positions + constraint._pbc_diag / 2
    device_positions = cuda.to_device(positive_positions)
    thread_per_block = (THREAD_PER_BLOCK[0], THREAD_PER_BLOCK[1])
    block_per_grid = (
        int(np.ceil(constraint._inner_grid_size[0] / thread_per_block[0])),
        int(np.ceil(constraint._inner_grid_size[1] / thread_per_block[1]))
    )
    print(block_per_grid)
    s = time.time()
    device_coulombic_electric_potential_map = cuda.to_device(np.zeros(constraint._inner_grid_size, dtype=env.NUMPY_FLOAT))
    constraint._update_coulombic_electric_potential_map[block_per_grid, thread_per_block](
        device_positions, constraint._device_charges, constraint._device_num_particles,
        constraint._device_k0, constraint._device_grid_width, constraint._device_inner_grid_size,
        constraint._device_cavity_relative_permittivity,
        device_coulombic_electric_potential_map
    )
    coulombic_electric_potential_map = device_coulombic_electric_potential_map.copy_to_host()
    e = time.time()
    print('Run coulombic electric potential for %s s' %(e-s))

    thread_per_block = (THREAD_PER_BLOCK[0], THREAD_PER_BLOCK[1])
    block_per_grid = (
        int(np.ceil(constraint._inner_grid_size[0] / thread_per_block[0])),
        int(np.ceil(constraint._inner_grid_size[1] / thread_per_block[1]))
    )
    cur = np.zeros(constraint._inner_grid_size, dtype=env.NUMPY_FLOAT)
    device_reaction_filed_electric_potential_map = cuda.to_device(cur)
    s = time.time()
    for i in range(500):
        constraint._update_reaction_field_electric_potential_map[block_per_grid, thread_per_block](
            constraint._device_relative_permittivity_map,
            constraint._device_cavity_relative_permittivity,
            constraint._device_inner_grid_size,
            device_coulombic_electric_potential_map,
            device_reaction_filed_electric_potential_map
        )
    reaction_field_electric_potential_map = device_reaction_filed_electric_potential_map.copy_to_host()
    e = time.time()
    print('Run reaction field electric potential for %s s' %(e-s))

    s = time.time()
    forces, potential_energy = constraint._update_reaction_field_force_and_potential_energy_kernel(
        constraint._parent_ensemble.state.positions,
        constraint._charges, constraint._grid_width,
        reaction_field_electric_potential_map
    )
    e = time.time()
    print('Run reaction field electric force for %s s' %(e-s))
    print(forces)
    print(potential_energy)

    ax1 = fig.add_subplot(311)
    Ey, Ex = np.gradient(-coulombic_electric_potential_map[:, :, constraint.total_grid_size[1]//2].T)
    ax1.streamplot(x, y, Ex, Ey, linewidth=1, cmap='RdBu', density=2)
    c = ax1.contourf(X, Y, coulombic_electric_potential_map[:, :, constraint.total_grid_size[1]//2].T, 100, cmap='RdBu')
    plt.colorbar(c)

    ax2 = fig.add_subplot(312)
    c = ax2.contourf(X, Y, reaction_field_electric_potential_map[:, :, constraint.total_grid_size[1]//2].T, 100, cmap='RdBu')
    Ey, Ex = np.gradient(-reaction_field_electric_potential_map[:, :, constraint.total_grid_size[1]//2].T)
    ax2.streamplot(x, y, Ex, Ey, linewidth=1, cmap='RdBu', density=2)
    for particle in range(ensemble.topology.num_particles):
        ax2.scatter(
            [positions[particle, 0]], [positions[particle, 1]],
            c='navy' if constraint._charges[particle] > 0 else 'brown'
        )
        ax2.plot(
            [positions[particle, 0], positions[particle, 0] + forces[particle, 0]*1e4],
            [positions[particle, 1], positions[particle, 1] + forces[particle, 1]*1e4],
            c='navy' if constraint._charges[particle] > 0 else 'brown'
        )
    plt.colorbar(c)

    ax3 = fig.add_subplot(313)
    c = ax3.contourf(X, Y, reaction_field_electric_potential_map[:, :, constraint.total_grid_size[1]//2].T + coulombic_electric_potential_map[:, :, constraint.total_grid_size[1]//2].T, 100, cmap='RdBu')
    Ey, Ex = np.gradient(-(
        reaction_field_electric_potential_map[:, :, constraint.total_grid_size[1]//2] +
        coulombic_electric_potential_map[:, :, constraint.total_grid_size[1]//2]
    ).T)
    ax3.streamplot(x, y, Ex, Ey, linewidth=1, cmap='RdBu', density=2)
    for particle in range(ensemble.topology.num_particles):
        ax3.scatter(
            [positions[particle, 0]], [positions[particle, 1]],
            c='navy' if constraint._charges[particle] > 0 else 'brown'
        )
    fig.tight_layout()
    plt.colorbar(c)
    plt.show()