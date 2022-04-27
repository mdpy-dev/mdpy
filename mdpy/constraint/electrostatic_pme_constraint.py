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
from mdpy.core import MAX_NUM_BONDED_PARTICLES
from mdpy.constraint import Constraint
from mdpy.utils import *
from mdpy.unit import *
from mdpy.error import *

PME_ORDER = 4
THREAD_PER_BLOCK = (64)

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
        self._device_k = cp.array([4 * np.pi * EPSILON0.value], CUPY_FLOAT)
        # Attribute
        self._charges = None
        self._grid_size = None
        self._b_grid = None
        self._c_grid = None
        # Kernel
        self._update_direct_part = cuda.jit(nb.void(
            NUMBA_FLOAT[::1], # charges
            NUMBA_FLOAT[::1], # k
            NUMBA_FLOAT[::1], # ewald_coefficient
            NUMBA_FLOAT[::1], # cutoff_radius
            NUMBA_INT[:, ::1], # bonded_particles
            NUMBA_INT[:, ::1], # neighbor_list
            NUMBA_FLOAT[:, :, ::1], # neighbor_vec_list
            NUMBA_FLOAT[:, ::1], # force
            NUMBA_FLOAT[::1] # potential_energy
        ))(self._update_direct_part_kernel)
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
            NUMBA_FLOAT[::1], # charges
            NUMBA_FLOAT[:, :, ::1] # charge_map
        ))(self._update_charge_map_kernel)
        self._update_electric_potential_map = self._update_electric_potential_map_kernel
        self._update_reciprocal_force = cuda.jit(nb.void(
            NUMBA_INT[::1], # num_particles
            NUMBA_FLOAT[:, :, ::1], # spline_coefficient
            NUMBA_FLOAT[:, :, ::1], # spline_derivative_coefficient
            NUMBA_INT[:, :, ::1], # grid_map
            NUMBA_FLOAT[:, :, ::1], # electric_potential_map
            NUMBA_FLOAT[::1], # charges
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
        potential_energy = (self._charges**2).sum() * self._ewald_coefficient / self._k / np.sqrt(np.pi)
        return potential_energy

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._constraint_id = ensemble.constraints.index(self)
        self._charges = self._parent_ensemble.topology.charges[:, 0]
        # if self._charges.sum() != 0:
        #     raise EnsemblePoorDefinedError('mdpy.constraint.ElectrostaticPMEConstraint is bound to a non-neutralized ensemble')
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
        self._device_charges = cp.array(self._charges, CUPY_FLOAT)
        self._device_cutoff_radius = cp.array([self._cutoff_radius], CUPY_FLOAT)
        self._device_bonded_particles = cp.array(self._parent_ensemble.topology.bonded_particles, CUPY_INT)
        self._device_b_grid = cp.array(self._b_grid, CUPY_FLOAT)
        self._device_c_grid = cp.array(self._c_grid, CUPY_FLOAT)
        self._device_bc_grid = cp.array(self._bc_grid, CUPY_FLOAT)
        self._device_reciprocal_factor = cp.array(
            self._grid_size / np.diagonal(self._parent_ensemble.state.pbc_matrix), CUPY_FLOAT
        )

    @staticmethod
    def _update_direct_part_kernel(
        charges,
        k, ewald_coefficient,
        cutoff_radius,
        bonded_particles,
        neighbor_list,
        neighbor_vec_list,
        forces, potential_energy
    ):
        particle_id1 = cuda.grid(1)
        num_particles = neighbor_list.shape[0]
        if particle_id1 >= num_particles:
            return None
        # Bonded particle
        local_bonded_particles = cuda.local.array(
            shape=(MAX_NUM_BONDED_PARTICLES), dtype=NUMBA_INT
        )
        for i in range(MAX_NUM_BONDED_PARTICLES):
            local_bonded_particles[i] = bonded_particles[particle_id1, i]

        shared_k = cuda.shared.array(shape=(1), dtype=NUMBA_FLOAT)
        shared_ewald_coefficient = cuda.shared.array(shape=(1), dtype=NUMBA_FLOAT)
        shared_cutoff_radius = cuda.shared.array(shape=(1), dtype=NUMBA_FLOAT)
        shared_sqrt_pi = cuda.shared.array(shape=(1), dtype=NUMBA_FLOAT)
        thread_x = cuda.threadIdx.x
        if thread_x == 0:
            shared_k[0] = k[0]
            shared_ewald_coefficient[0] = ewald_coefficient[0]
        elif thread_x == 1:
            shared_cutoff_radius[0] = cutoff_radius[0]
            shared_sqrt_pi[0] = sqrt_pi = math.sqrt(math.pi)
        cuda.syncthreads()

        # id1 attribute
        force_x = 0
        force_y = 0
        force_z = 0
        energy = 0
        e1 = charges[particle_id1]
        for neighbor_index in range(neighbor_list.shape[1]):
            particle_id2 = neighbor_list[particle_id1, neighbor_index]
            if particle_id2 == -1: # self-padding term
                break
            if particle_id1 == particle_id2:
                continue
            if particle_id1 == particle_id2: # self-self term
                continue
            is_bonded = False
            for i in local_bonded_particles:
                if i == -1: # padding of bonded particle
                    break
                elif particle_id2 == i: # self-bonded particle term
                    is_bonded = True
                    r = neighbor_vec_list[particle_id1, neighbor_index, 0]
                    scaled_x = neighbor_vec_list[particle_id1, neighbor_index, 1]
                    scaled_y = neighbor_vec_list[particle_id1, neighbor_index, 2]
                    scaled_z = neighbor_vec_list[particle_id1, neighbor_index, 3]
                    e1e2 = e1 * charges[particle_id2]
                    ewald_r = shared_ewald_coefficient[0] * r
                    erf = math.erf(ewald_r)
                    force_val = e1e2 * (
                        2*shared_ewald_coefficient[0]*math.exp(-(ewald_r)**2) / shared_sqrt_pi[0] - erf / r
                    ) / shared_k[0] / r
                    force_x -= scaled_x * force_val
                    force_y -= scaled_y * force_val
                    force_z -= scaled_z * force_val
                    energy -= e1e2 * erf / shared_k[0] / r / 2
                    is_continue = True
                    break
            if is_bonded:
                continue
            r = neighbor_vec_list[particle_id1, neighbor_index, 0]
            if r <= shared_cutoff_radius[0]:
                scaled_x = neighbor_vec_list[particle_id1, neighbor_index, 1]
                scaled_y = neighbor_vec_list[particle_id1, neighbor_index, 2]
                scaled_z = neighbor_vec_list[particle_id1, neighbor_index, 3]
                e1e2 = e1 * charges[particle_id2]
                ewald_r = shared_ewald_coefficient[0]*r
                erfc = math.erfc(ewald_r)
                force_val = - e1e2 * (
                    2*shared_ewald_coefficient[0]*math.exp(-(ewald_r)**2) / shared_sqrt_pi[0] + erfc / r
                ) / shared_k[0] / r
                force_x += scaled_x * force_val
                force_y += scaled_y * force_val
                force_z += scaled_z * force_val
                energy += e1e2 * erfc / shared_k[0] / r / 2
        cuda.atomic.add(forces, (particle_id1, 0), force_x)
        cuda.atomic.add(forces, (particle_id1, 1), force_y)
        cuda.atomic.add(forces, (particle_id1, 2), force_z)
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
            local_spline_coefficient[i, 2] = 0.5 * grid_fraction[i]**2
            local_spline_coefficient[i, 0] = 0.5 * (1 - grid_fraction[i])**2
            local_spline_coefficient[i, 1] = 1 - local_spline_coefficient[i, 0] - local_spline_coefficient[i, 2]
        # 4 order derivative coefficient
        for i in range(SPATIAL_DIM):
            local_spline_derivative_coefficient[i, 0] = - local_spline_coefficient[i, 0]
            local_spline_derivative_coefficient[i, 1] = local_spline_coefficient[i, 0] - local_spline_coefficient[i, 1]
            local_spline_derivative_coefficient[i, 2] = local_spline_coefficient[i, 1] - local_spline_coefficient[i, 2]
            local_spline_derivative_coefficient[i, 3] = local_spline_coefficient[i, 2]
        # 4 order spline coefficient
        for i in range(SPATIAL_DIM):
            local_spline_coefficient[i, 3] = grid_fraction[i] * local_spline_coefficient[i, 2] / 3
            local_spline_coefficient[i, 2] = (
                (1 + grid_fraction[i]) * local_spline_coefficient[i, 1] +
                (3 - grid_fraction[i]) * local_spline_coefficient[i, 2]
            ) / 3
            local_spline_coefficient[i, 0] = (1 - grid_fraction[i]) * local_spline_coefficient[i, 0] / 3
            local_spline_coefficient[i, 1] = 1 - (
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
        charge = charges[particle_id]
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
    def _update_electric_potential_map_kernel(k, charge_map, bc_grid):
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
        charge = charges[particle_id]
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
        self._direct_forces = cp.zeros_like(self._parent_ensemble.state.positions, CUPY_FLOAT)
        self._direct_potential_energy = cp.zeros([1], CUPY_FLOAT)
        # Direct part
        block_per_grid = int(np.ceil(
            self._parent_ensemble.topology.num_particles / THREAD_PER_BLOCK
        ))
        self._update_direct_part[block_per_grid, THREAD_PER_BLOCK](
            self._device_charges,
            self._device_k,
            self._device_ewald_coefficient,
            self._device_cutoff_radius,
            self._device_bonded_particles,
            self._parent_ensemble.state.neighbor_list.device_neighbor_list,
            self._parent_ensemble.state.neighbor_list.device_neighbor_vec_list,
            self._direct_forces, self._direct_potential_energy
        )
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
            self._parent_ensemble.state.device_positions, self._device_grid_size,
            self._parent_ensemble.state.device_pbc_matrix,
            spline_coefficient, spline_derivative_coefficient, grid_map
        )
        # Map charge
        self._charge_map = cp.zeros(self._grid_size, CUPY_FLOAT)
        self._update_charge_map[block_per_grid, thread_per_block](
            self._device_num_particles, spline_coefficient, grid_map,
            self._device_charges, self._charge_map
        )
        # Reciprocal convolution
        self._electric_potential_map = self._update_electric_potential_map(
            self._device_k, self._charge_map, self._device_bc_grid
        )
        # Reciprocal force
        self._reciprocal_forces = cp.zeros_like(self._parent_ensemble.state.positions, CUPY_FLOAT)
        self._reciprocal_potential_energy = cp.zeros([1], CUPY_FLOAT)
        self._update_reciprocal_force[block_per_grid, thread_per_block](
            self._device_num_particles,
            spline_coefficient,
            spline_derivative_coefficient,
            grid_map,
            self._electric_potential_map,
            self._device_charges,
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
        self._forces = (self._direct_forces +  self._reciprocal_forces)

    @property
    def grid_size(self):
        return self._grid_size

    @property
    def ewald_coefficient(self):
        return self._ewald_coefficient
