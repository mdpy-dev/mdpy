#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : electrostatic_fdpe_constraint.py
created time : 2022/04/11
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

from concurrent.futures import thread
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


MAX_NUM_PARTICLES = 250
GRID_POINTS_PER_BLOCK = 8  # 8*8*8 grids point will be solved in one block
TOTAL_POINTS_PER_BLOCK = GRID_POINTS_PER_BLOCK + 2  # 10*10*10 neighbors are needed


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
                NUMBA_INT[::1],  # shared_array_allocation_parameters,
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
        shared_array_allocation_parameters,
        cavity_relative_permittivity,
        coulombic_electric_potential_map,
    ):
        grid_x, grid_y, grid_z = cuda.grid(3)
        num_particles = positions.shape[0]
        grid_width = grid_width[0]
        inverse_grid_width = NUMBA_FLOAT(1) / grid_width
        # Shared array
        shared_particle_grid_index = cuda.shared.array(
            (MAX_NUM_PARTICLES, SPATIAL_DIM), NUMBA_FLOAT
        )
        shared_particle_charges = cuda.shared.array((MAX_NUM_PARTICLES), NUMBA_FLOAT)
        # Load data before skipping threads
        thread_hashing_index = (
            cuda.blockDim.z * (cuda.threadIdx.x * cuda.blockDim.y + cuda.threadIdx.y)
            + cuda.threadIdx.z
        )
        num_target_particles_each_thread = shared_array_allocation_parameters[0]
        num_threads_per_block = shared_array_allocation_parameters[1]
        for i in range(num_target_particles_each_thread):
            particle_index = i * num_threads_per_block + thread_hashing_index
            if particle_index < num_particles:
                for j in range(SPATIAL_DIM):
                    # positions[particle_id, x] / grid_width is the grid id of total grid
                    shared_particle_grid_index[particle_index, j] = (
                        positions[particle_index, j] * inverse_grid_width
                    )
                shared_particle_charges[particle_index] = charges[particle_index, 0]
        if grid_x >= inner_grid_size[0]:
            return
        if grid_y >= inner_grid_size[1]:
            return
        if grid_z >= inner_grid_size[2]:
            return
        denominator = NUMBA_FLOAT(1) / k[0] / cavity_relative_permittivity[0]
        float_grid_index = cuda.local.array((SPATIAL_DIM), NUMBA_FLOAT)
        float_grid_index[0] = NUMBA_FLOAT(grid_x + 1)
        float_grid_index[1] = NUMBA_FLOAT(grid_y + 1)
        vec = cuda.local.array((SPATIAL_DIM), NUMBA_FLOAT)
        r = NUMBA_FLOAT(0)
        electric_potential = NUMBA_FLOAT(0)
        float_grid_index[2] = NUMBA_FLOAT(grid_z + 1)
        cuda.syncthreads()
        for particle_id in range(num_particles):
            r = NUMBA_FLOAT(0)
            for i in range(SPATIAL_DIM):
                vec[i] = (
                    abs(
                        shared_particle_grid_index[particle_id, i] - float_grid_index[i]
                    )
                    * grid_width
                )
                r += vec[i] ** 2
            r = math.sqrt(r)
            if r < NUMBA_FLOAT(0.01):
                r = NUMBA_FLOAT(0.01)
            electric_potential += shared_particle_charges[particle_id] / r * denominator
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
        thread_per_block = (8, 8, 8)
        block_per_grid = (
            int(np.ceil(self._inner_grid_size[0] / thread_per_block[0])),
            int(np.ceil(self._inner_grid_size[1] / thread_per_block[1])),
            int(np.ceil(self._inner_grid_size[2] / thread_per_block[2])),
        )
        self._device_coulombic_electric_potential_map = cp.zeros(
            self._inner_grid_size, CUPY_FLOAT
        )
        num_threads_per_block = (
            thread_per_block[0] * thread_per_block[1] * thread_per_block[2]
        )
        target_particles_per_thread = np.ceil(
            self._parent_ensemble.topology.num_particles / num_threads_per_block
        )
        shared_array_allocation_parameters = cp.array(
            [
                target_particles_per_thread,
                num_threads_per_block,
            ],
            CUPY_INT,
        )
        self._update_coulombic_electric_potential_map[block_per_grid, thread_per_block](
            positive_positions,
            self._parent_ensemble.topology.device_charges,
            self._device_k0,
            self._device_grid_width,
            self._device_inner_grid_size,
            shared_array_allocation_parameters,
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
        for _ in range(100):
            self._update_reaction_field_electric_potential_map[
                block_per_grid, thread_per_block
            ](
                self._device_relative_permittivity_map,
                self._device_coulombic_electric_potential_map,
                self._device_cavity_relative_permittivity,
                self._device_inner_grid_size,
                self._device_reaction_filed_electric_potential_map,
            )
        # Coulombic force and potential energy
        self._columbic_forces = cp.zeros(
            (self._parent_ensemble.topology.num_particles, SPATIAL_DIM), CUPY_FLOAT
        )
        self._columbic_potential_energy = cp.zeros((1), CUPY_FLOAT)
        thread_per_block = (8, 8)
        block_per_grid = (
            int(
                np.ceil(
                    self._parent_ensemble.topology.num_particles / thread_per_block[0]
                )
            ),
            int(
                np.ceil(
                    self._parent_ensemble.topology.num_particles / thread_per_block[1]
                )
            ),
        )
        self._update_coulombic_force_and_potential_energy[
            block_per_grid, thread_per_block
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
