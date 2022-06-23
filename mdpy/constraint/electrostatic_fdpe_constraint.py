#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : electrostatic_fdpe_constraint.py
created time : 2022/04/11
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import math
import numpy as np
import numba as nb
import numba.cuda as cuda
import cupy as cp
from mdpy import SPATIAL_DIM
from mdpy.environment import *
from mdpy.core import Ensemble
from mdpy.constraint import Constraint
from mdpy.utils import *
from mdpy.unit import *
from mdpy.error import *

GRID_POINTS_PER_BLOCK = 8  # 8*8*8 grids point will be solved in one block
TOTAL_POINTS_PER_BLOCK = GRID_POINTS_PER_BLOCK + 2  # 10*10*10 neighbors are needed
THREAD_PER_BLOCK = (16, 16)


class ElectrostaticFDPEConstraint(Constraint):
    def __init__(
        self,
        grid_width=Quantity(0.5, angstrom),
        cavity_relative_permittivity: float = 2,
    ) -> None:
        super().__init__()
        # Input
        self._grid_width = check_quantity_value(grid_width, default_length_unit)
        self._cavity_relative_permittivity = cavity_relative_permittivity
        # Attribute
        self._inner_grid_size = np.zeros([3], NUMPY_INT)
        self._total_grid_size = np.zeros([3], NUMPY_INT)
        self._k0 = 4 * np.pi * EPSILON0.value
        self._epsilon0 = EPSILON0.value
        # device attributes
        self._device_grid_width = cp.array([self._grid_width], CUPY_FLOAT)
        self._device_cavity_relative_permittivity = cp.array(
            [self._cavity_relative_permittivity], CUPY_FLOAT
        )
        self._device_k0 = cp.array([self._k0], CUPY_FLOAT)
        self._device_epsilon0 = cp.array([self._epsilon0], CUPY_FLOAT)
        # Kernel
        self._update_coulombic_electric_potential_map = cuda.jit(
            nb.void(
                NUMBA_FLOAT[:, ::1],  # positions
                NUMBA_FLOAT[:, ::1],  # charges
                NUMBA_FLOAT[::1],  # k0
                NUMBA_FLOAT[::1],  # grid_width
                NUMBA_INT[::1],  # inner_grid_size,
                NUMBA_FLOAT[::1],  # cavity_relative_permittivity
                NUMBA_FLOAT[:, :, ::1],  # coulombic_electric_potential_map
            )
        )(self._update_coulombic_electric_potential_map_kernel)
        self._update_reaction_field_electric_potential_map = cuda.jit(
            nb.void(
                NUMBA_FLOAT[:, :, ::1],  # relative_permittivity_map
                NUMBA_FLOAT[:, :, ::1],  # coulombic_electric_potential_map
                NUMBA_FLOAT[::1],  # cavity_relative_permittivity
                NUMBA_INT[::1],  # inner_grid_size
                NUMBA_FLOAT[:, :, ::1],  # reaction_field_electric_potential_map
            )
        )(self._update_reaction_field_electric_potential_map_kernel)
        self._update_coulombic_force_and_potential_energy = cuda.jit(
            nb.void(
                NUMBA_FLOAT[:, ::1],  # positions
                NUMBA_FLOAT[:, ::1],  # charges
                NUMBA_FLOAT[:, ::1],  # pbc_matrix
                NUMBA_FLOAT[::1],  # k0
                NUMBA_FLOAT[::1],  # cavity_relative_permittivity
                NUMBA_FLOAT[:, ::1],  # forces
                NUMBA_FLOAT[::1],  # potential_energy
            )
        )(self._update_coulombic_force_and_potential_energy_kernel)
        self._update_reaction_field_force_and_potential_energy = nb.njit(
            (
                NUMBA_FLOAT[:, ::1],  # positions
                NUMBA_FLOAT[:, ::1],  # charges
                NUMBA_FLOAT,  # grid_width
                NUMBA_FLOAT[:, :, ::1],  # reaction_field_electric_potential_map
            )
        )(self._update_reaction_field_force_and_potential_energy_kernel)

    def __repr__(self) -> str:
        return "<mdpy.constraint.ElectrostaticFDPEConstraint object>"

    def __str__(self) -> str:
        return "FDPE electrostatic constraint"

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._constraint_id = ensemble.constraints.index(self)
        pbc_diag = np.diagonal(self._parent_ensemble.state.pbc_matrix)
        self._total_grid_size = (
            np.ceil(pbc_diag / self._grid_width).astype(NUMPY_INT) + 1
        )
        self._inner_grid_size = self._total_grid_size - 2
        # device attributes
        self._device_inner_grid_size = cp.array(self._inner_grid_size, CUPY_INT)
        self._device_total_grid_size = cp.array(self._total_grid_size, CUPY_INT)
        self._device_reaction_filed_electric_potential_map = cp.zeros(
            self._inner_grid_size, CUPY_FLOAT
        )

    def set_relative_permittivity_map(
        self, relative_permittivity_map: cp.ndarray
    ) -> None:
        map_shape = relative_permittivity_map.shape
        for i, j in zip(self._inner_grid_size, map_shape):
            if i != j:
                raise ArrayDimError(
                    "Relative permittivity map requires a [%d, %d, %d] array"
                    % (
                        self._inner_grid_size[0],
                        self._inner_grid_size[1],
                        self._inner_grid_size[2],
                    )
                    + ", while an [%d, %d, %d] array is provided"
                    % (map_shape[0], map_shape[1], map_shape[2])
                )
        self._device_relative_permittivity_map = relative_permittivity_map.astype(
            CUPY_FLOAT
        )

    @staticmethod
    def _update_coulombic_electric_potential_map_kernel(
        positions,
        charges,
        k,
        grid_width,
        inner_grid_size,
        cavity_relative_permittivity,
        coulombic_electric_potential_map,
    ):
        grid_x, grid_y = cuda.grid(2)
        if grid_x >= inner_grid_size[0]:
            return
        if grid_y >= inner_grid_size[1]:
            return
        num_particles = positions.shape[0]
        denominator = NUMBA_FLOAT(1) / k[0] / cavity_relative_permittivity[0]
        grid_width = grid_width[0]
        inverse_grid_width = NUMBA_FLOAT(1) / grid_width
        for particle_id in range(num_particles):
            # positions[particle_id, x] / grid_width is the grid id of total grid
            charge = charges[particle_id, 0]
            particle_grid_x = positions[particle_id, 0] * inverse_grid_width - 1
            particle_grid_y = positions[particle_id, 1] * inverse_grid_width - 1
            particle_grid_z = positions[particle_id, 2] * inverse_grid_width - 1
            dist_x = abs(particle_grid_x - grid_x) * grid_width
            dist_y = abs(particle_grid_y - grid_y) * grid_width
            dist_xy2 = dist_x**2 + dist_y**2
            for grid_z in range(inner_grid_size[2]):
                dist_z = abs(particle_grid_z - grid_z) * grid_width
                dist = math.sqrt(dist_xy2 + dist_z**2)
                if dist < 0.01:
                    dist = NUMBA_FLOAT(0.01)
                electric_potential = charge / dist
                electric_potential *= denominator
                cuda.atomic.add(
                    coulombic_electric_potential_map,
                    (grid_x, grid_y, grid_z),
                    electric_potential,
                )

    @staticmethod
    def _update_reaction_field_electric_potential_map_kernel(
        relative_permittivity_map,
        coulombic_electric_potential_map,
        cavity_relative_permittivity,
        inner_grid_size,
        reaction_field_electric_potential_map,
    ):
        # Local index
        local_thread_index = cuda.local.array((SPATIAL_DIM), NUMBA_INT)
        local_thread_index[0] = cuda.threadIdx.x
        local_thread_index[1] = cuda.threadIdx.y
        local_thread_index[2] = cuda.threadIdx.z
        # Global index
        global_thread_index = cuda.local.array((SPATIAL_DIM), NUMBA_INT)
        global_thread_index[0] = (
            local_thread_index[0] + cuda.blockIdx.x * cuda.blockDim.x
        )
        global_thread_index[1] = (
            local_thread_index[1] + cuda.blockIdx.y * cuda.blockDim.y
        )
        global_thread_index[2] = (
            local_thread_index[2] + cuda.blockIdx.z * cuda.blockDim.z
        )
        # Grid size
        grid_size = cuda.local.array((SPATIAL_DIM), NUMBA_INT)
        for i in range(SPATIAL_DIM):
            grid_size[i] = inner_grid_size[i]
            if global_thread_index[i] >= grid_size[i]:
                return
        # Neighbor array index
        local_array_index = cuda.local.array((SPATIAL_DIM), NUMBA_INT)
        for i in range(SPATIAL_DIM):
            local_array_index[i] = local_thread_index[i] + NUMBA_INT(1)
        # Shared array
        shared_relative_permittivity_map = cuda.shared.array(
            (TOTAL_POINTS_PER_BLOCK, TOTAL_POINTS_PER_BLOCK, TOTAL_POINTS_PER_BLOCK),
            NUMBA_FLOAT,
        )
        shared_coulombic_electric_potential_map = cuda.shared.array(
            (TOTAL_POINTS_PER_BLOCK, TOTAL_POINTS_PER_BLOCK, TOTAL_POINTS_PER_BLOCK),
            NUMBA_FLOAT,
        )
        shared_reaction_field_electric_potential_map = cuda.shared.array(
            (TOTAL_POINTS_PER_BLOCK, TOTAL_POINTS_PER_BLOCK, TOTAL_POINTS_PER_BLOCK),
            NUMBA_FLOAT,
        )
        # Load self point data
        shared_relative_permittivity_map[
            local_array_index[0], local_array_index[1], local_array_index[2]
        ] = relative_permittivity_map[
            global_thread_index[0], global_thread_index[1], global_thread_index[2]
        ]
        shared_coulombic_electric_potential_map[
            local_array_index[0], local_array_index[1], local_array_index[2]
        ] = coulombic_electric_potential_map[
            global_thread_index[0], global_thread_index[1], global_thread_index[2]
        ]
        shared_reaction_field_electric_potential_map[
            local_array_index[0], local_array_index[1], local_array_index[2]
        ] = reaction_field_electric_potential_map[
            global_thread_index[0], global_thread_index[1], global_thread_index[2]
        ]
        # Load boundary point
        for i in range(SPATIAL_DIM):
            if local_thread_index[i] == 0:
                tmp_local_array_index = local_array_index[i]
                tmp_global_thread_index = global_thread_index[i]
                local_array_index[i] = 0
                global_thread_index[i] -= 1
                shared_relative_permittivity_map[
                    local_array_index[0], local_array_index[1], local_array_index[2]
                ] = relative_permittivity_map[
                    global_thread_index[0],
                    global_thread_index[1],
                    global_thread_index[2],
                ]
                shared_coulombic_electric_potential_map[
                    local_array_index[0], local_array_index[1], local_array_index[2]
                ] = coulombic_electric_potential_map[
                    global_thread_index[0],
                    global_thread_index[1],
                    global_thread_index[2],
                ]
                shared_reaction_field_electric_potential_map[
                    local_array_index[0], local_array_index[1], local_array_index[2]
                ] = reaction_field_electric_potential_map[
                    global_thread_index[0],
                    global_thread_index[1],
                    global_thread_index[2],
                ]
                local_array_index[i] = tmp_local_array_index
                global_thread_index[i] = tmp_global_thread_index
            elif local_array_index[i] == GRID_POINTS_PER_BLOCK or global_thread_index[
                i
            ] == (grid_size[i] - NUMBA_FLOAT(1)):
                # Equivalent to local_thread_index[i] == GRID_POINTS_PER_BLOCK - 1
                tmp_local_array_index = local_array_index[i]
                tmp_global_thread_index = global_thread_index[i]
                local_array_index[i] += 1
                global_thread_index[i] += 1
                global_thread_index[i] *= NUMBA_INT(
                    global_thread_index[i] != grid_size[i]
                )
                shared_relative_permittivity_map[
                    local_array_index[0], local_array_index[1], local_array_index[2]
                ] = relative_permittivity_map[
                    global_thread_index[0],
                    global_thread_index[1],
                    global_thread_index[2],
                ]
                shared_coulombic_electric_potential_map[
                    local_array_index[0], local_array_index[1], local_array_index[2]
                ] = coulombic_electric_potential_map[
                    global_thread_index[0],
                    global_thread_index[1],
                    global_thread_index[2],
                ]
                shared_reaction_field_electric_potential_map[
                    local_array_index[0], local_array_index[1], local_array_index[2]
                ] = reaction_field_electric_potential_map[
                    global_thread_index[0],
                    global_thread_index[1],
                    global_thread_index[2],
                ]
                local_array_index[i] = tmp_local_array_index
                global_thread_index[i] = tmp_global_thread_index
        cuda.syncthreads()
        # Load constant
        cavity_relative_permittivity = cavity_relative_permittivity[0]
        # Calculate
        new_val = NUMBA_FLOAT(0)
        denominator = NUMBA_FLOAT(0)
        # Self term
        self_relative_permittivity = shared_relative_permittivity_map[
            local_array_index[0],
            local_array_index[1],
            local_array_index[2],
        ]
        self_coulombic_electric_potential = shared_coulombic_electric_potential_map[
            local_array_index[0],
            local_array_index[1],
            local_array_index[2],
        ]
        new_val += (
            NUMBA_FLOAT(6)
            * cavity_relative_permittivity
            * self_coulombic_electric_potential
        )
        # Neighbor term
        for i in range(SPATIAL_DIM):
            for j in [-1, 1]:
                local_array_index[i] += j
                relative_permittivity = NUMBA_FLOAT(0.5) * (
                    self_relative_permittivity
                    + shared_relative_permittivity_map[
                        local_array_index[0],
                        local_array_index[1],
                        local_array_index[2],
                    ]
                )
                new_val += (
                    relative_permittivity
                    * shared_reaction_field_electric_potential_map[
                        local_array_index[0],
                        local_array_index[1],
                        local_array_index[2],
                    ]
                )
                new_val += shared_coulombic_electric_potential_map[
                    local_array_index[0],
                    local_array_index[1],
                    local_array_index[2],
                ] * (relative_permittivity - cavity_relative_permittivity)
                denominator += relative_permittivity
                local_array_index[i] -= j
        # Update
        new_val /= denominator
        new_val -= self_coulombic_electric_potential
        old_val = reaction_field_electric_potential_map[
            global_thread_index[0],
            global_thread_index[1],
            global_thread_index[2],
        ]
        cuda.atomic.add(
            reaction_field_electric_potential_map,
            (
                global_thread_index[0],
                global_thread_index[1],
                global_thread_index[2],
            ),
            NUMBA_FLOAT(0.9) * new_val - NUMBA_FLOAT(0.9) * old_val,
        )

    @staticmethod
    def _update_coulombic_force_and_potential_energy_kernel(
        positions,
        charges,
        pbc_matrix,
        k0,
        cavity_relative_permittivity,
        forces,
        potential_energy,
    ):
        particle_id1, particle_id2 = cuda.grid(2)
        num_particles = positions.shape[0]
        if particle_id1 >= num_particles:
            return
        if particle_id2 >= num_particles:
            return
        if particle_id1 == particle_id2:
            return
        local_thread_x = cuda.threadIdx.x
        shared_pbc_matrix = cuda.shared.array((SPATIAL_DIM), NUMBA_FLOAT)
        shared_half_pbc_matrix = cuda.shared.array((SPATIAL_DIM), NUMBA_FLOAT)
        if local_thread_x <= 2:
            shared_pbc_matrix[local_thread_x] = pbc_matrix[
                local_thread_x, local_thread_x
            ]
            shared_half_pbc_matrix[local_thread_x] = shared_pbc_matrix[
                local_thread_x
            ] * NUMBA_FLOAT(0.5)
        k0 = k0[0]
        cavity_relative_permittivity = cavity_relative_permittivity[0]
        factor = k0 * cavity_relative_permittivity
        factor = charges[particle_id1, 0] * charges[particle_id2, 0] / factor

        r = NUMBA_FLOAT(0)
        vec = cuda.local.array((SPATIAL_DIM), NUMBA_FLOAT)
        for i in range(SPATIAL_DIM):
            vec[i] = positions[particle_id2, i] - positions[particle_id1, i]
            if vec[i] < -pbc_matrix[i, i] / 2:
                vec[i] += shared_pbc_matrix[i]
            elif vec[i] > shared_half_pbc_matrix[i]:
                vec[i] -= shared_pbc_matrix[i]
            r += vec[i] ** 2
            vec[i] = positions[particle_id2, i] - positions[particle_id1, i]
            r += vec[i] ** 2
        r = math.sqrt(r)
        force_val = -factor / r**2
        for i in range(SPATIAL_DIM):
            force = vec[i] * force_val * NUMBA_FLOAT(0.5) / r
            cuda.atomic.add(forces, (particle_id1, i), force)
            cuda.atomic.add(forces, (particle_id2, i), -force)
        energy = factor / r * NUMBA_FLOAT(0.5)
        cuda.atomic.add(potential_energy, 0, energy)

    @staticmethod
    def _update_reaction_field_force_and_potential_energy_kernel(
        positions, charges, grid_width, reaction_field_electric_potential_map
    ):
        """
        Least square solution:
        f(x, y, z) = a0x^2 + a1y^2 + a2z^2 + a3xy + a4xz + a5yz + a6x + a7y + a8z + a9

        Solution:
        A^TA a = A^Tb
        """
        forces = np.zeros_like(positions)
        potential_energy = 0
        for particle in range(positions.shape[0]):
            grid_index = positions[particle, :] / grid_width - 1
            grid_index_int = np.floor(grid_index).astype(np.int32)
            grid_index_float = grid_index - grid_index_int
            gradient = 0
            for i in range(SPATIAL_DIM):
                grid_index_int[i] += 1
                gradient = reaction_field_electric_potential_map[
                    grid_index_int[0], grid_index_int[1], grid_index_int[2]
                ]
                grid_index_int[i] -= 2
                gradient -= reaction_field_electric_potential_map[
                    grid_index_int[0], grid_index_int[1], grid_index_int[2]
                ]
                grid_index_int[i] += 1
                forces[particle, i] = -charges[particle, 0] * gradient / grid_width / 2

            potential_energy += (
                reaction_field_electric_potential_map[
                    grid_index_int[0], grid_index_int[1], grid_index_int[2]
                ]
                * charges[particle, 0]
            )
        return forces, potential_energy

    def update(self):
        positive_positions = (
            self._parent_ensemble.state.positions
            + self._parent_ensemble.state.device_half_pbc_diag
        )
        # Columbic electric potential map
        block_per_grid = (
            int(np.ceil(self._inner_grid_size[0] / THREAD_PER_BLOCK[0])),
            int(np.ceil(self._inner_grid_size[1] / THREAD_PER_BLOCK[1])),
        )
        self._device_coulombic_electric_potential_map = cp.zeros(
            self._inner_grid_size, CUPY_FLOAT
        )
        self._update_coulombic_electric_potential_map[block_per_grid, THREAD_PER_BLOCK](
            positive_positions,
            self._parent_ensemble.topology.device_charges,
            self._device_k0,
            self._device_grid_width,
            self._device_inner_grid_size,
            self._device_cavity_relative_permittivity,
            self._device_coulombic_electric_potential_map,
        )
        # Reaction field potential map
        thread_per_block = (
            GRID_POINTS_PER_BLOCK,
            GRID_POINTS_PER_BLOCK,
            GRID_POINTS_PER_BLOCK,
        )
        block_per_grid = (
            int(np.ceil(self._inner_grid_size[0] / GRID_POINTS_PER_BLOCK)),
            int(np.ceil(self._inner_grid_size[1] / GRID_POINTS_PER_BLOCK)),
            int(np.ceil(self._inner_grid_size[2] / GRID_POINTS_PER_BLOCK)),
        )
        for _ in range(500):
            self._update_reaction_field_electric_potential_map[
                block_per_grid, thread_per_block
            ](
                self._device_relative_permittivity_map,
                self._device_coulombic_electric_potential_map,
                self._device_cavity_relative_permittivity,
                self._device_inner_grid_size,
                self._device_reaction_filed_electric_potential_map,
            )
            cuda.synchronize()
        # Coulombic force and potential energy
        self._columbic_forces = cp.zeros(
            (self._parent_ensemble.topology.num_particles, SPATIAL_DIM), CUPY_FLOAT
        )
        self._columbic_potential_energy = cp.zeros((1), CUPY_FLOAT)
        block_per_grid = (
            int(
                np.ceil(
                    self._parent_ensemble.topology.num_particles / THREAD_PER_BLOCK[0]
                )
            ),
            int(
                np.ceil(
                    self._parent_ensemble.topology.num_particles / THREAD_PER_BLOCK[1]
                )
            ),
        )
        self._update_coulombic_force_and_potential_energy[
            block_per_grid, THREAD_PER_BLOCK
        ](
            positive_positions,
            self._parent_ensemble.topology.device_charges,
            self._parent_ensemble.state.device_pbc_matrix,
            self._device_k0,
            self._device_cavity_relative_permittivity,
            self._columbic_forces,
            self._columbic_potential_energy,
        )
        # Reaction field force and potential
        (
            self._reaction_field_forces,
            self._reaction_field_potential_energy,
        ) = self._update_reaction_field_force_and_potential_energy(
            positive_positions.get(),
            self._parent_ensemble.topology.charges,
            self._grid_width,
            self._device_reaction_filed_electric_potential_map.get(),
        )
        self._reaction_field_forces = cp.array(self._reaction_field_forces, CUPY_FLOAT)
        self._reaction_field_potential_energy = cp.array(
            [self._reaction_field_potential_energy], CUPY_FLOAT
        )
        self._forces = self._reaction_field_forces + self._columbic_forces
        self._potential_energy = (
            self._reaction_field_potential_energy + self._columbic_potential_energy
        )

    @property
    def inner_grid_size(self) -> np.ndarray:
        return self._inner_grid_size

    @property
    def num_inner_grids(self) -> np.ndarray:
        return self._inner_grid_size.prod()

    @property
    def total_grid_size(self) -> np.ndarray:
        return self._total_grid_size

    @property
    def num_total_grids(self) -> int:
        return self._total_grid_size.prod()

    @property
    def grid_width(self) -> float:
        return self._grid_width

    @property
    def device_coulombic_electric_potential_map(self) -> cp.ndarray:
        return self._device_coulombic_electric_potential_map

    @property
    def device_reaction_field_electric_potential_map(self) -> cp.ndarray:
        return self._device_reaction_filed_electric_potential_map

    @property
    def columbic_forces(self) -> cp.ndarray:
        return self._columbic_forces

    @property
    def columbic_potential_energy(self) -> cp.ndarray:
        return self._columbic_potential_energy

    @property
    def reaction_field_forces(self) -> cp.ndarray:
        return self._reaction_field_forces

    @property
    def reaction_field_potential_energy(self) -> cp.ndarray:
        return self._reaction_field_potential_energy
