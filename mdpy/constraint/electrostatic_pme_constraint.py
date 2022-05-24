#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : electrostatic_pme_constraint.py
created time : 2022/04/07
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import math
import numpy as np
import numba as nb
import cupy as cp
from numba import cuda
from mdpy import SPATIAL_DIM
from mdpy.environment import *
from mdpy.core import Ensemble
from mdpy.core import NUM_PARTICLES_PER_TILE, MAX_NUM_EXCLUDED_PARTICLES
from mdpy.constraint import Constraint
from mdpy.utils import *
from mdpy.unit import *
from mdpy.error import *

PME_ORDER = 4
THREAD_PER_BLOCK = (64)
TILES_PER_THREAD = 4

def bspline(x, order):
    if order == 2:
        return 1 - np.abs(x-1) if x >= 0 and x <= 2 else 0
    else:
        return (
            x * bspline(x, order-1) / (order - 1) +
            (order - x) * bspline(x-1, order-1) / (order-1)
        )

def gamma_sum(m, num_grids, order):
    x = np.pi * m / num_grids
    k = np.arange(1, 51)
    res = 1
    res += ((x / (x + np.pi*k))**order).sum()
    res += ((x / (x - np.pi*k))**order).sum()
    return res

class ElectrostaticPMEConstraint(Constraint):
    def __init__(self, cutoff_radius=Quantity(8, angstrom), direct_sum_energy_tolerance=1e-5) -> None:
        super().__init__()
        self._cutoff_radius = check_quantity_value(cutoff_radius, default_length_unit)
        self._device_cutoff_radius = cp.array([self._cutoff_radius], CUPY_FLOAT)
        self._direct_sum_energy_tolerance = direct_sum_energy_tolerance
        self._ewald_coefficient = self._get_ewald_coefficient()
        self._device_ewald_coefficient = cp.array([self._ewald_coefficient], CUPY_FLOAT)
        self._k = 4 * np.pi * EPSILON0.value
        self._device_k = cp.array([self._k], CUPY_FLOAT)
        self._device_inverse_k = cp.array([1 / self._k], CUPY_FLOAT)
        # Attribute
        self._grid_size = None
        self._b_grid = None
        self._c_grid = None
        # Kernel
        self._update_pme_direct_part = cuda.jit(nb.void(
            NUMBA_FLOAT[::1], # inverse_k
            NUMBA_FLOAT[::1], # ewald_coefficient
            NUMBA_FLOAT[::1], # cutoff_radius
            NUMBA_FLOAT[:, ::1], # pbc_matrix
            NUMBA_FLOAT[:, ::1], # sorted_positions
            NUMBA_FLOAT[:, ::1], # sorted_charges
            NUMBA_BIT[:, ::1], # exclusion_map
            NUMBA_INT[:, ::1], # tile_neighbors
            NUMBA_FLOAT[:, ::1], # sorted_forces
            NUMBA_FLOAT[::1] # potential_energy
        ), fastmath=True, max_registers=32)(self._update_pme_direct_part_kernel)
        self._update_excluded_pme_direct_part = cuda.jit(nb.void(
            NUMBA_FLOAT[::1], # inverse_k
            NUMBA_FLOAT[::1], # ewald_coefficient
            NUMBA_FLOAT[::1], # cutoff_radius
            NUMBA_FLOAT[:, ::1], # pbc_matrix
            NUMBA_FLOAT[:, ::1], # positions
            NUMBA_FLOAT[:, ::1], # charges
            NUMBA_INT[:, ::1], # excluded_particles
            NUMBA_FLOAT[:, ::1], # forces
            NUMBA_FLOAT[::1] # potential_energy
        ))(self._update_excluded_pme_direct_part_kernel)
        self._update_bspline = cuda.jit(nb.void(
            NUMBA_FLOAT[:, ::1], # position
            NUMBA_INT[::1], # grid_size,
            NUMBA_FLOAT[:, ::1], # pbc_matrix
            NUMBA_FLOAT[:, :, ::1], # spline_coefficient
            NUMBA_FLOAT[:, :, ::1], # spline_derivative_coefficient
            NUMBA_INT[:, :, ::1] # grid_map
        ))(self._update_bspline_kernel)
        self._update_charge_map = cuda.jit(nb.void(
            NUMBA_INT[::1], # num_particles
            NUMBA_FLOAT[:, :, ::1], # spline_coefficient
            NUMBA_INT[:, :, ::1], # grid_map
            NUMBA_FLOAT[:, ::1], # charges
            NUMBA_FLOAT[:, :, ::1] # charge_map
        ))(self._update_charge_map_kernel)
        self._update_electric_potential_map = self._update_reciprocal_electric_potential_map_kernel
        self._update_reciprocal_force = cuda.jit(nb.void(
            NUMBA_INT[::1], # num_particles
            NUMBA_FLOAT[:, :, ::1], # spline_coefficient
            NUMBA_FLOAT[:, :, ::1], # spline_derivative_coefficient
            NUMBA_INT[:, :, ::1], # grid_map
            NUMBA_FLOAT[:, :, ::1], # electric_potential_map
            NUMBA_FLOAT[:, ::1], # charges
            NUMBA_FLOAT[:, ::1], # forces
            NUMBA_FLOAT[::1] # potential_energy
        ))(self._update_reciprocal_force_kernel)

    def __repr__(self) -> str:
        return '<mdpy.constraint.ElectrostaticPMEConstraint object>'

    def __str__(self) -> str:
        return 'PME electrostatic constraint'

    def _get_ewald_coefficient(self):
        '''
        Using Newton interation solve ewald coefficient

        Essentially, the Ewald coefficient is mathematical shorthand for 1/2s,
        where s is the width of the Gaussian used to smooth out charges on the grid.
        That width is chosen such that, at the direct space interaction `cutoff_radius`,
        the interaction of two Gaussian-smoothed charges and the interaction of
        two point charges are identical to a precision of `direct_sum_energy_tolerance`,
        which mean the interaction between these 4 charge site are small enough,
        proving truncation yielding no big error

        f(alpha) = erfc(alpha*cutoff_radius) / cutoff_radius - direct_sum_energy_tolerance
        f'(alpha) = - 2/sqrt(pi) * exp[-(alpha*cutoff_radius)^2]

        alpha_new = alpha_old - f(alpha_old) / f'(alpha_old)
        '''
        alpha = 0.1
        sqrt_pi = np.sqrt(np.pi)
        while True:
            f = (
                math.erfc(alpha*self._cutoff_radius)/self._cutoff_radius -
                self._direct_sum_energy_tolerance
            )
            df = - 2 * np.exp(-(alpha*self._cutoff_radius)**2) / sqrt_pi
            d_alpha = f / df
            if np.abs(d_alpha/alpha) < 1e-5:
                break
            alpha -= d_alpha
        return alpha

    def _get_grid_size(self):
        '''
        Set the grid size with spacing around 0.5 angstrom and power to 2 for better fft performance
        '''
        pbc_diag = np.diag(Quantity(
            self._parent_ensemble.state.pbc_matrix, default_length_unit
        ).convert_to(angstrom).value)
        grid_size = np.ceil(pbc_diag/32) * 32
        return grid_size.astype(NUMPY_INT)

    def _get_b_grid(self):
        b_grid = np.zeros(self._grid_size)
        b_factor = []
        for i in range(SPATIAL_DIM):
            num_grids = self._grid_size[i]
            half_num_grids = num_grids // 2
            # denominator
            # Calculate the abs of denminator directly
            denominator = np.zeros([num_grids])
            prefactor = np.zeros([3])
            prefactor[0] = bspline(1, PME_ORDER)
            prefactor[1] = bspline(2, PME_ORDER)
            prefactor[2] = bspline(3, PME_ORDER)
            k_vec = np.arange(3)
            for m in np.arange(num_grids):
                factor = 2 * np.pi * m * k_vec / num_grids
                sin_term = (prefactor * np.sin(factor)).sum()
                cos_term = (prefactor * np.cos(factor)).sum()
                denominator[m] = sin_term**2 + cos_term**2
            # Handle small denominator
            for i in range(num_grids):
                if denominator[i] <= 1e-7:
                    if i != 0 and i != num_grids-1:
                        # Replace with mean value of two neighbor points
                        denominator[i] = (denominator[i-1] + denominator[i+1]) / 2
                    else:
                        # Boundary point set to a very high value, 0 for denominator
                        denominator[i] = 1e7
            # nominator
            nominator = np.zeros([num_grids])
            for index in range(num_grids):
                if index >= half_num_grids:
                    m = index - num_grids
                else:
                    m = index
                if m == 0:
                    nominator[index] = 1
                else:
                    nominator[index] = gamma_sum(m, num_grids, PME_ORDER) / gamma_sum(m, num_grids, 8)
            b_factor.append(nominator**2/denominator)
        for i in range(self._grid_size[0]):
            for j in range(self._grid_size[1]):
                for k in range(self._grid_size[2]):
                    b_grid[i, j, k] = (
                        b_factor[0][i] * b_factor[1][j] * b_factor[2][k]
                    )
        return b_grid

    def _get_c_grid(self):
        c_grid = np.zeros(self._grid_size)
        freqency_factor, exp_factor = [], []
        for i in range(SPATIAL_DIM):
            num_grids = self._grid_size[i]
            half_num_grids = num_grids // 2
            m_vec = np.arange(num_grids)
            m_vec[m_vec > half_num_grids] -= num_grids
            m_vec = m_vec / self._parent_ensemble.state.pbc_matrix[i, i]
            exp_vec = np.exp(-(np.pi*m_vec/self._ewald_coefficient)**2)
            freqency_factor.append(m_vec)
            exp_factor.append(exp_vec)
        for i in range(self._grid_size[0]):
            for j in range(self._grid_size[1]):
                for k in range(self._grid_size[2]):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    c_grid[i, j, k] = (
                        exp_factor[0][i] * exp_factor[1][j] * exp_factor[2][k]
                    )
                    c_grid[i, j, k] = c_grid[i, j, k] / (
                        freqency_factor[0][i]**2 +
                        freqency_factor[1][j]**2 +
                        freqency_factor[2][k]**2
                    )
        c_grid = c_grid / ( # / pi V
            np.prod(np.diagonal(self._parent_ensemble.state.pbc_matrix)) /
            self._num_grids_total * np.pi
        )
        return c_grid

    def _get_self_potential_energy(self):
        potential_energy = (self._parent_ensemble.topology.charges**2).sum() * self._ewald_coefficient / self._k / np.sqrt(np.pi)
        return potential_energy

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._constraint_id = ensemble.constraints.index(self)
        if not np.isclose(self._parent_ensemble.topology.charges.sum(), 0, atol=1e-3):
            raise EnsemblePoorDefinedError('mdpy.constraint.ElectrostaticPMEConstraint is bound to a non-neutralized ensemble')
        # Grid size
        self._grid_size = self._get_grid_size()
        self._num_grids_total = np.prod(self._grid_size)
        # create b grid
        self._b_grid = self._get_b_grid()
        # create c grid
        self._c_grid = self._get_c_grid()
        # b_grid * c_grid
        self._bc_grid = self._b_grid * self._c_grid
        # calculate self energy correction
        self._self_potential_energy = self._get_self_potential_energy()
        # device attributes
        self._device_grid_size = cp.array(self._grid_size, CUPY_INT)
        self._device_num_particles = cp.array(
            [self._parent_ensemble.topology.num_particles], CUPY_INT
        )
        self._device_self_potential_energy = cp.array(self._self_potential_energy, CUPY_FLOAT)
        self._device_b_grid = cp.array(self._b_grid, CUPY_FLOAT)
        self._device_c_grid = cp.array(self._c_grid, CUPY_FLOAT)
        self._device_bc_grid = cp.array(self._bc_grid, CUPY_FLOAT)
        self._device_reciprocal_factor = cp.array(
            self._grid_size / np.diagonal(self._parent_ensemble.state.pbc_matrix), CUPY_FLOAT
        )

    @staticmethod
    def _update_pme_direct_part_kernel(
        inverse_k, ewald_coefficient,
        cutoff_radius,
        pbc_matrix,
        sorted_positions,
        sorted_charges,
        exclusion_map,
        tile_neighbors,
        sorted_forces, potential_energy
    ):
        # Particle index information
        local_thread_x = cuda.threadIdx.x
        local_thread_y = cuda.threadIdx.y
        tile_id1 = cuda.blockIdx.x * TILES_PER_THREAD + local_thread_y
        if tile_id1 >= tile_neighbors.shape[0]:
            return
        tile1_particle_index = tile_id1 * NUM_PARTICLES_PER_TILE + local_thread_x
        # shared data
        local_pbc_matrix = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        local_half_pbc_matrix = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        for i in range(SPATIAL_DIM):
            local_pbc_matrix[i] = pbc_matrix[i, i]
            local_half_pbc_matrix[i] = local_pbc_matrix[i] * NUMBA_FLOAT(0.5)
        tile1_positions = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        tile2_positions = cuda.shared.array(shape=(SPATIAL_DIM, TILES_PER_THREAD, NUM_PARTICLES_PER_TILE), dtype=NUMBA_FLOAT)
        tile2_charges = cuda.shared.array(shape=(TILES_PER_THREAD, NUM_PARTICLES_PER_TILE), dtype=NUMBA_FLOAT)
        cuda.syncthreads()
        # Read data
        for i in range(SPATIAL_DIM):
            tile1_positions[i] = sorted_positions[i, tile1_particle_index]
        tile1_charges = sorted_charges[0, tile1_particle_index]
        # Local data
        local_forces = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        vec = cuda.local.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        energy = NUMBA_FLOAT(0)
        inverse_k = inverse_k[0]
        inverse_sqrt_pi = NUMBA_FLOAT(1) / math.sqrt(NUMBA_FLOAT(math.pi))
        ewald_coefficient = ewald_coefficient[0]
        cutoff_radius = cutoff_radius[0]
        for i in range(SPATIAL_DIM):
            local_forces[i] = 0
        for neighbor_index in range(tile_neighbors.shape[1]):
            tile_id2 = tile_neighbors[tile_id1, neighbor_index]
            tile2_particle_index = tile_id2 * NUM_PARTICLES_PER_TILE + local_thread_x
            cuda.syncthreads()
            for i in range(SPATIAL_DIM):
                tile2_positions[i, local_thread_y, local_thread_x] = sorted_positions[i, tile2_particle_index]
            tile2_charges[local_thread_y, local_thread_x] = sorted_charges[0, tile2_particle_index]
            exclusion_flag = exclusion_map[neighbor_index, tile1_particle_index]
            cuda.syncthreads()
            if tile_id2 == -1:
                break
            # Computation
            for particle_index in range(NUM_PARTICLES_PER_TILE):
                if exclusion_flag >> particle_index & 0b1:
                    continue
                r = NUMBA_FLOAT(0)
                for i in range(SPATIAL_DIM):
                    vec[i] = tile2_positions[i, local_thread_y, particle_index] - tile1_positions[i]
                    if vec[i] < - local_half_pbc_matrix[i]:
                        vec[i] += local_pbc_matrix[i]
                    elif vec[i] > local_half_pbc_matrix[i]:
                        vec[i] -= local_pbc_matrix[i]
                    r += vec[i]**2
                r = math.sqrt(r)
                if r < cutoff_radius:
                    ewald_r = ewald_coefficient * r
                    e1e2_over_k = tile1_charges * tile2_charges[local_thread_y, particle_index] * inverse_k
                    inverse_r = NUMBA_FLOAT(1) / r
                    erfc_over_r = math.erfc(ewald_r) * inverse_r
                    energy += e1e2_over_k * erfc_over_r * NUMBA_FLOAT(0.5)
                    force_val = - e1e2_over_k * (
                        NUMBA_FLOAT(2)*ewald_coefficient*math.exp(-(ewald_r)**2) * inverse_sqrt_pi + erfc_over_r
                    ) * inverse_r * inverse_r
                    for i in range(SPATIAL_DIM):
                        local_forces[i] += force_val * vec[i]
        for i in range(SPATIAL_DIM):
            cuda.atomic.add(sorted_forces, (i, tile1_particle_index), local_forces[i])
        cuda.atomic.add(potential_energy, 0, energy)

    @staticmethod
    def _update_excluded_pme_direct_part_kernel(
        inverse_k, ewald_coefficient,
        cutoff_radius,
        pbc_matrix,
        positions,
        charges,
        excluded_particles,
        forces, potential_energy
    ):
        particle1 = cuda.grid(1)
        local_thread_x = cuda.threadIdx.x
        if particle1 >= positions.shape[0]:
            return
        shared_pbc_matrix = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        shared_half_pbc_matrix = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        if local_thread_x <= 2:
            shared_pbc_matrix[local_thread_x] = pbc_matrix[local_thread_x, local_thread_x]
            shared_half_pbc_matrix[local_thread_x] = shared_pbc_matrix[local_thread_x] * NUMBA_FLOAT(0.5)
        cuda.syncthreads()
        local_positions = cuda.local.array(shape=(3), dtype=NUMBA_FLOAT)
        local_forces = cuda.local.array(shape=(3), dtype=NUMBA_FLOAT)
        vec = cuda.local.array(shape=(3), dtype=NUMBA_FLOAT)
        e1 = charges[particle1, 0]
        inverse_k = inverse_k[0]
        ewald_coefficient = ewald_coefficient[0]
        cutoff_radius = cutoff_radius[0]
        inverse_sqrt_pi = NUMBA_FLOAT(1) / math.sqrt(NUMBA_FLOAT(math.pi))
        for i in range(SPATIAL_DIM):
            local_positions[i] = positions[particle1, i]
            local_forces[i] = 0
        energy = NUMBA_FLOAT(0)
        is_excluded = False # Prevent atomic add for no excluded particles
        for i in range(excluded_particles.shape[1]):
            particle2 = excluded_particles[particle1, i]
            if particle2 == -1:
                break
            is_excluded = True
            r = NUMBA_FLOAT(0)
            for i in range(SPATIAL_DIM):
                vec[i] = positions[particle2, i] - local_positions[i]
                if vec[i] < - shared_half_pbc_matrix[i]:
                    vec[i] += shared_pbc_matrix[i]
                elif vec[i] > shared_half_pbc_matrix[i]:
                    vec[i] -= shared_pbc_matrix[i]
                r += vec[i]**2
            r = math.sqrt(r)
            if r < cutoff_radius:
                ewald_r = ewald_coefficient * r
                e1e2_over_k = e1 * charges[particle2, 0] * inverse_k
                inverse_r = NUMBA_FLOAT(1) / r
                erf_over_r = math.erf(ewald_r) * inverse_r
                force_val = e1e2_over_k * (
                    NUMBA_FLOAT(2)*ewald_coefficient*math.exp(-(ewald_r)**2) * inverse_sqrt_pi - erf_over_r
                ) * inverse_r * inverse_r
                for i in range(SPATIAL_DIM):
                    local_forces[i] -= force_val * vec[i]
                energy -= e1e2_over_k * erf_over_r * NUMBA_FLOAT(0.5)
        if is_excluded:
            for i in range(SPATIAL_DIM):
                cuda.atomic.add(forces, (particle1, i), local_forces[i])
            cuda.atomic.add(potential_energy, 0, energy)

    @staticmethod
    def _update_bspline_kernel(
        positions, grid_size, pbc_matrix,
        spline_coefficient,
        spline_derivative_coefficient,
        grid_map
    ):
        particle_id = cuda.grid(1)
        thread_x = cuda.threadIdx.x
        if particle_id >= positions.shape[0]:
            return None
        # Shared array
        # PBC matrix
        shared_pbc_matrix = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        shared_half_pbc_matrix = cuda.shared.array(shape=(SPATIAL_DIM), dtype=NUMBA_FLOAT)
        # num_cell_vec
        shared_grid_size = cuda.shared.array(shape=(3), dtype=NUMBA_INT)
        if thread_x == 0:
            shared_pbc_matrix[0] = pbc_matrix[0, 0]
            shared_pbc_matrix[1] = pbc_matrix[1, 1]
            shared_pbc_matrix[2] = pbc_matrix[2, 2]
            shared_half_pbc_matrix[0] = shared_pbc_matrix[0] / 2
            shared_half_pbc_matrix[1] = shared_pbc_matrix[1] / 2
            shared_half_pbc_matrix[2] = shared_pbc_matrix[2] / 2
        elif thread_x == 1:
            shared_grid_size[0] = grid_size[0]
            shared_grid_size[1] = grid_size[1]
            shared_grid_size[2] = grid_size[2]
        cuda.syncthreads()
        position_x = positions[particle_id, 0] + shared_half_pbc_matrix[0]
        position_y = positions[particle_id, 1] + shared_half_pbc_matrix[1]
        position_z = positions[particle_id, 2] + shared_half_pbc_matrix[2]
        grid_index_x = position_x / shared_pbc_matrix[0] * shared_grid_size[0]
        grid_index_y = position_y / shared_pbc_matrix[1] * shared_grid_size[1]
        grid_index_z = position_z / shared_pbc_matrix[2] * shared_grid_size[2]
        grid_index = cuda.local.array((SPATIAL_DIM), NUMBA_INT)
        grid_index[0] = math.floor(grid_index_x)
        grid_index[1] = math.floor(grid_index_y)
        grid_index[2] = math.floor(grid_index_z)
        grid_fraction = cuda.local.array((SPATIAL_DIM), NUMBA_FLOAT)
        grid_fraction[0] = grid_index_x - grid_index[0]
        grid_fraction[1] = grid_index_y - grid_index[1]
        grid_fraction[2] = grid_index_z - grid_index[2]

        local_spline_coefficient = cuda.local.array((SPATIAL_DIM, PME_ORDER), NUMBA_FLOAT)
        local_spline_derivative_coefficient = cuda.local.array((SPATIAL_DIM, PME_ORDER), NUMBA_FLOAT)
        # 3 order B-spline
        for i in range(SPATIAL_DIM):
            local_spline_coefficient[i, 2] = NUMBA_FLOAT(0.5) * grid_fraction[i]**2
            local_spline_coefficient[i, 0] = NUMBA_FLOAT(0.5)* (1 - grid_fraction[i])**2
            local_spline_coefficient[i, 1] = NUMBA_FLOAT(1) - local_spline_coefficient[i, 0] - local_spline_coefficient[i, 2]
        # 4 order derivative coefficient
        for i in range(SPATIAL_DIM):
            local_spline_derivative_coefficient[i, 0] = - local_spline_coefficient[i, 0]
            local_spline_derivative_coefficient[i, 1] = local_spline_coefficient[i, 0] - local_spline_coefficient[i, 1]
            local_spline_derivative_coefficient[i, 2] = local_spline_coefficient[i, 1] - local_spline_coefficient[i, 2]
            local_spline_derivative_coefficient[i, 3] = local_spline_coefficient[i, 2]
        # 4 order spline coefficient
        for i in range(SPATIAL_DIM):
            local_spline_coefficient[i, 3] = grid_fraction[i] * local_spline_coefficient[i, 2] / NUMBA_FLOAT(3)
            local_spline_coefficient[i, 2] = (
                (NUMBA_FLOAT(1) + grid_fraction[i]) * local_spline_coefficient[i, 1] +
                (NUMBA_FLOAT(3) - grid_fraction[i]) * local_spline_coefficient[i, 2]
            ) / 3
            local_spline_coefficient[i, 0] = (NUMBA_FLOAT(1) - grid_fraction[i]) * local_spline_coefficient[i, 0] / NUMBA_FLOAT(3)
            local_spline_coefficient[i, 1] = NUMBA_FLOAT(1) - (
                local_spline_coefficient[i, 0] + local_spline_coefficient[i, 2] + local_spline_coefficient[i, 3]
            )
        # Set value
        for i in range(SPATIAL_DIM):
            for j in range(PME_ORDER):
                cuda.atomic.add(
                    spline_coefficient,
                    (particle_id, i, j),
                    local_spline_coefficient[i, j]
                )
                cuda.atomic.add(
                    spline_derivative_coefficient,
                    (particle_id, i, j),
                    local_spline_derivative_coefficient[i, j]
                )
                index = grid_index[i] + j - 1
                if index >= shared_grid_size[i]:
                    index -= shared_grid_size[i]
                elif index < 0:
                    index += shared_grid_size[i]
                cuda.atomic.add(grid_map, (particle_id, i, j), index)

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
        charge = charges[particle_id, 0]
        for i in range(PME_ORDER):
            grid_x = grid_map[particle_id, 0, i]
            charge_x = charge * spline_coefficient[particle_id, 0, i]
            for j in range(PME_ORDER):
                grid_y = grid_map[particle_id, 1, j]
                charge_xy = charge_x * spline_coefficient[particle_id, 1, j]
                for k in range(PME_ORDER):
                    grid_z = grid_map[particle_id, 2, k]
                    charge_xyz = charge_xy * spline_coefficient[particle_id, 2, k]
                    cuda.atomic.add(charge_map, (grid_x, grid_y, grid_z), charge_xyz)

    @staticmethod
    def _update_reciprocal_electric_potential_map_kernel(k, charge_map, bc_grid):
        # Convolution
        fft_charge = cp.fft.fftn(charge_map / k)
        fft_charge[0, 0, 0] = 0
        electric_potential_map = cp.real(cp.fft.ifftn(fft_charge*bc_grid))
        return electric_potential_map.astype(CUPY_FLOAT)

    @staticmethod
    def _update_reciprocal_force_kernel(
        num_particles, spline_coefficient, spline_derivative_coefficient,
        grid_map, electric_potential_map, charges, forces, potential_energy
    ):
        '''
        spline_coefficient: [num_particles, SPATIAL_DIM, PME_ORDER] The spline coefficient of particles
        spline_derivative_coefficient: [num_particles, SPATIAL_DIM, PME_ORDER] The spline derivative coefficient of particles
        grid_map: [num_particles, SPATIL_DIM, PME_ORDER]: The indice of grid to add charge of each particles
        electric_potential_map: [grid_size_x, grid_size_y, grid_size_z]: Electric potential on each grid point
        forces: [num_particles, SPATIAL_DIM]
        '''
        particle_id = cuda.grid(1)
        num_particles = num_particles[0]
        if particle_id >= num_particles:
            return None
        charge = charges[particle_id, 0]
        force_x = 0
        force_y = 0
        force_z = 0
        energy = 0
        for i in range(PME_ORDER):
            grid_x = grid_map[particle_id, 0, i]
            spline_coefficient_x = spline_coefficient[particle_id, 0, i]
            spline_derivative_coefficient_x = spline_derivative_coefficient[particle_id, 0, i]
            for j in range(PME_ORDER):
                grid_y = grid_map[particle_id, 1, j]
                spline_coefficient_y = spline_coefficient[particle_id, 1, j]
                spline_derivative_coefficient_y = spline_derivative_coefficient[particle_id, 1, j]
                for k in range(PME_ORDER):
                    grid_z = grid_map[particle_id, 2, k]
                    spline_coefficient_z = spline_coefficient[particle_id, 2, k]
                    spline_derivative_coefficient_z = spline_derivative_coefficient[particle_id, 2, k]
                    cur_energy = (
                        spline_coefficient_x * spline_coefficient_y * spline_coefficient_z *
                        electric_potential_map[grid_x, grid_y, grid_z]
                    )
                    force_x -= (
                        cur_energy / spline_coefficient_x *
                        spline_derivative_coefficient_x
                    )
                    force_y -= (
                        cur_energy / spline_coefficient_y *
                        spline_derivative_coefficient_y
                    )
                    force_z -= (
                        cur_energy / spline_coefficient_z *
                        spline_derivative_coefficient_z
                    )
                    energy += cur_energy
        cuda.atomic.add(forces, (particle_id, 0), force_x * charge)
        cuda.atomic.add(forces, (particle_id, 1), force_y * charge)
        cuda.atomic.add(forces, (particle_id, 2), force_z * charge)
        cuda.atomic.add(potential_energy, 0, energy)

    def update(self):
        self._check_bound_state()
        # Direct part
        self._direct_potential_energy = cp.zeros([1], CUPY_FLOAT)
        sorted_forces = cp.zeros((SPATIAL_DIM, self._parent_ensemble.tile_list.num_tiles * NUM_PARTICLES_PER_TILE), CUPY_FLOAT)
        sorted_positions = self._parent_ensemble.tile_list.sort_matrix(self._parent_ensemble.state.positions)
        thread_per_block = (NUM_PARTICLES_PER_TILE, TILES_PER_THREAD)
        block_per_grid = (int(np.ceil(self._parent_ensemble.tile_list.num_tiles / TILES_PER_THREAD)))
        self._update_pme_direct_part[block_per_grid, thread_per_block](
            self._device_inverse_k,
            self._device_ewald_coefficient,
            self._device_cutoff_radius,
            self._parent_ensemble.state.device_pbc_matrix,
            sorted_positions,
            self._parent_ensemble.topology.device_sorted_charges,
            self._parent_ensemble.topology.device_exclusion_map,
            self._parent_ensemble.tile_list.tile_neighbors,
            sorted_forces, self._direct_potential_energy
        )
        self._direct_forces = self._parent_ensemble.tile_list.unsort_matrix(sorted_forces)
        thread_per_block = 64
        block_per_grid = int(np.ceil(
            self._parent_ensemble.topology.num_particles / thread_per_block
        ))
        self._update_excluded_pme_direct_part[block_per_grid, thread_per_block](
            self._device_inverse_k,
            self._device_ewald_coefficient,
            self._device_cutoff_radius,
            self._parent_ensemble.state.device_pbc_matrix,
            self._parent_ensemble.state.positions,
            self._parent_ensemble.topology.device_charges,
            self._parent_ensemble.topology.device_excluded_particles,
            self._direct_forces, self._direct_potential_energy
        )
        # Reciprocal part
        thread_per_block = (64)
        block_per_grid = int(np.ceil(
            self._parent_ensemble.topology.num_particles / thread_per_block
        ))
        # Bspline
        spline_coefficient = cp.zeros(
            [self._parent_ensemble.topology.num_particles, SPATIAL_DIM, PME_ORDER], CUPY_FLOAT
        )
        spline_derivative_coefficient = cp.zeros(
            [self._parent_ensemble.topology.num_particles, SPATIAL_DIM, PME_ORDER], CUPY_FLOAT
        )
        grid_map = cp.zeros(
            [self._parent_ensemble.topology.num_particles, SPATIAL_DIM, PME_ORDER], CUPY_INT
        )
        self._update_bspline[block_per_grid, thread_per_block](
            self._parent_ensemble.state.positions, self._device_grid_size,
            self._parent_ensemble.state.device_pbc_matrix,
            spline_coefficient, spline_derivative_coefficient, grid_map
        )
        # Map charge
        self._charge_map = cp.zeros(self._grid_size, CUPY_FLOAT)
        self._update_charge_map[block_per_grid, thread_per_block](
            self._device_num_particles, spline_coefficient, grid_map,
            self._parent_ensemble.topology.device_charges, self._charge_map
        )
        # Reciprocal convolution
        self._electric_potential_map = self._update_electric_potential_map(
            self._device_k, self._charge_map, self._device_bc_grid
        )
        # Reciprocal force
        self._reciprocal_forces = cp.zeros(self._parent_ensemble.state.matrix_shape, CUPY_FLOAT)
        self._reciprocal_potential_energy = cp.zeros([1], CUPY_FLOAT)
        self._update_reciprocal_force[block_per_grid, thread_per_block](
            self._device_num_particles,
            spline_coefficient,
            spline_derivative_coefficient,
            grid_map,
            self._electric_potential_map,
            self._parent_ensemble.topology.device_charges,
            self._reciprocal_forces,
            self._reciprocal_potential_energy
        )
        self._reciprocal_forces = self._reciprocal_forces * self._device_reciprocal_factor
        self._reciprocal_forces -= self._reciprocal_forces.mean(0)
        # Summary
        self._potential_energy = (
            self._direct_potential_energy +
            self._reciprocal_potential_energy -
            self._device_self_potential_energy
        )
        self._forces = (self._direct_forces + self._reciprocal_forces)

    @property
    def grid_size(self):
        return self._grid_size

    @property
    def ewald_coefficient(self):
        return self._ewald_coefficient

if __name__ == '__main__':
    import os
    import mdpy as md
    from mdpy.unit import *
    data_dir = '/home/zhenyuwei/nutstore/ZhenyuWei/Note_Research/mdpy/mdpy/benchmark/data'
    psf_file = os.path.join(data_dir, 'str.psf')
    pdb_file = os.path.join(data_dir, 'str.pdb')
    cutoff_radius = Quantity(9, angstrom)
    # IO
    psf = md.io.PSFParser(psf_file)
    pdb = md.io.PDBParser(pdb_file)
    charmm_prm = md.io.CharmmTopparParser(
        os.path.join(data_dir, 'par_all36_prot.prm'),
        os.path.join(data_dir, 'toppar_water_ions_namd.str'),
    )
    # constraint
    ensemble = md.core.Ensemble(psf.topology, pdb.pbc_matrix)
    constraint = ElectrostaticPMEConstraint()
    ensemble.add_constraints(constraint)
    ensemble.state.set_positions(pdb.positions)
    ensemble.update_tile_list()
    constraint.update()
    # print(constraint.forces)
    print(Quantity(constraint.forces, default_force_unit).convert_to(kilojoule_permol_over_nanometer).value)