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
import scipy.fft as fft
from numba import cuda
from mdpy import env, SPATIAL_DIM
from mdpy.core import Ensemble
from mdpy.core import NUM_NEIGHBOR_CELLS, NEIGHBOR_CELL_TEMPLATE
from mdpy.core import MAX_NUM_BONDED_PARTICLES
from mdpy.constraint import Constraint
from mdpy.utils import *
from mdpy.unit import *
from mdpy.error import *

PME_ORDER = 4
THREAD_PER_BLOCK = (32, 4)

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
        self._device_cutoff_radius = cuda.to_device(np.array([self._cutoff_radius]))
        self._direct_sum_energy_tolerance = direct_sum_energy_tolerance
        self._ewald_coefficient = self._get_ewald_coefficient()
        self._device_ewald_coefficient = cuda.to_device(
            np.array([self._ewald_coefficient], dtype=env.NUMPY_FLOAT)
        )
        self._k = 4 * np.pi * EPSILON0.value
        self._device_k = cuda.to_device(np.array([4 * np.pi * EPSILON0.value], dtype=env.NUMPY_FLOAT))
        self._device_neighbor_cell_template = cuda.to_device(NEIGHBOR_CELL_TEMPLATE.astype(env.NUMPY_INT))
        # Attribute
        self._charges = None
        self._grid_size = None
        self._b_grid = None
        self._c_grid = None
        # Kernel
        self._update_direct_part = cuda.jit(nb.void(
            env.NUMBA_FLOAT[:, ::1], # position
            env.NUMBA_FLOAT[::1], # charges
            env.NUMBA_FLOAT[::1], # k
            env.NUMBA_FLOAT[::1], # ewald_coefficient
            env.NUMBA_FLOAT[::1], # cutoff_radius
            env.NUMBA_FLOAT[:, ::1], # pbc_matrix
            env.NUMBA_INT[:, ::1], # bonded_particles
            env.NUMBA_INT[:, ::1], # particle_cell_index
            env.NUMBA_INT[:, :, :, ::1], # cell_list
            env.NUMBA_INT[::1], # num_cell_vec
            env.NUMBA_INT[:, ::1], # neighbor_cell_template
            env.NUMBA_FLOAT[:, ::1], # force
            env.NUMBA_FLOAT[::1] # potential_energy
        ))(self._update_direct_part_kernel)
        self._update_bspline = self._update_bspline_kernel
        self._update_charge_map = cuda.jit(nb.void(
            env.NUMBA_INT[::1], # num_particles
            env.NUMBA_FLOAT[:, :, ::1], # spline_coefficient
            env.NUMBA_INT[:, :, ::1], # grid_map
            env.NUMBA_FLOAT[::1], # charges
            env.NUMBA_FLOAT[:, :, ::1] # charge_map
        ))(self._update_charge_map_kernel)
        self._update_electric_potential_map = self._update_electric_potential_map_kernel
        self._update_reciprocal_force = cuda.jit(nb.void(
            env.NUMBA_INT[::1], # num_particles
            env.NUMBA_FLOAT[:, :, ::1], # spline_coefficient
            env.NUMBA_FLOAT[:, :, ::1], # spline_derivative_coefficient
            env.NUMBA_INT[:, :, ::1], # grid_map
            env.NUMBA_FLOAT[:, :, ::1], # electric_potential_map
            env.NUMBA_FLOAT[::1], # charges
            env.NUMBA_FLOAT[:, ::1], # force
            env.NUMBA_FLOAT[::1] # potential_energy
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
        return grid_size.astype(env.NUMPY_INT)

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

    def _get_self_energy(self):
        potential_energy = (self._charges**2).sum() * self._ewald_coefficient / self._k / np.sqrt(np.pi)
        return potential_energy

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._constraint_id = ensemble.constraints.index(self)
        self._charges = self._parent_ensemble.topology.charges[:, 0]
        if self._charges.sum() != 0:
            raise EnsemblePoorDefinedError('mdpy.constraint.ElectrostaticPMEConstraint is bound to a non-neutralized ensemble')
        # Grid size
        self._grid_size = self._get_grid_size()
        self._num_grids_total = np.prod(self._grid_size)
        # create b grid
        self._b_grid = self._get_b_grid()
        # create c grid
        self._c_grid = self._get_c_grid()
        # calculate self energy correction
        self._potential_energy_self = self._get_self_energy()
        # device attributes
        self._device_num_particles = cuda.to_device(np.array(
            [self._parent_ensemble.topology.num_particles], dtype=env.NUMPY_INT
        ))
        self._device_charges = cuda.to_device(self._charges)
        self._device_cutoff_radius = cuda.to_device(np.array([self._cutoff_radius]))
        self._device_bonded_particles = cuda.to_device(self._parent_ensemble.topology.bonded_particles)
        self._device_b_grid = cuda.to_device(self._b_grid)
        self._device_c_grid = cuda.to_device(self._c_grid)

    @staticmethod
    def _update_direct_part_kernel(
        positions, charges,
        k, ewald_coefficient,
        cutoff_radius, pbc_matrix,
        bonded_particles,
        particle_cell_index, cell_list,
        num_cell_vec, neighbor_cell_template,
        forces, potential_energy
    ):
        thread_x, thread_y = cuda.grid(2)
        num_particles_per_cell = cell_list.shape[3]
        num_particles = positions.shape[0]
        id1 = thread_x
        if id1 >= num_particles:
            return None
        cell_id = thread_y
        if cell_id >= NUM_NEIGHBOR_CELLS:
            return None
        thread_x = cuda.threadIdx.x
        thread_y = cuda.threadIdx.y
        # PBC matrix
        shared_pbc_matrix = cuda.shared.array(
            shape=(SPATIAL_DIM), dtype=nb.float32
        )
        shared_half_pbc_matrix = cuda.shared.array(
            shape=(SPATIAL_DIM), dtype=nb.float32
        )
        # Bonded particle
        shared_bonded_particles = cuda.shared.array(
            shape=(THREAD_PER_BLOCK[0], MAX_NUM_BONDED_PARTICLES), dtype=nb.int32
        )
        # num_cell_vec
        shared_num_cell_vec = cuda.shared.array(shape=(3), dtype=nb.int32)
        shared_k = cuda.shared.array(shape=(1), dtype=nb.float32)
        shared_ewald_coefficient = cuda.shared.array(shape=(1), dtype=nb.float32)
        shared_cutoff_radius = cuda.shared.array(shape=(1), dtype=nb.float32)
        shared_sqrt_pi = cuda.shared.array(shape=(1), dtype=nb.float32)
        if thread_y == 0:
            if thread_x == 0:
                shared_pbc_matrix[0] = pbc_matrix[0, 0]
                shared_pbc_matrix[1] = pbc_matrix[1, 1]
                shared_pbc_matrix[2] = pbc_matrix[2, 2]
                shared_half_pbc_matrix[0] = shared_pbc_matrix[0] / 2
                shared_half_pbc_matrix[1] = shared_pbc_matrix[1] / 2
                shared_half_pbc_matrix[2] = shared_pbc_matrix[2] / 2
            if thread_x == 1:
                shared_num_cell_vec[0] = num_cell_vec[0]
                shared_num_cell_vec[1] = num_cell_vec[1]
                shared_num_cell_vec[2] = num_cell_vec[2]
                shared_k[0] = k[0]
                shared_ewald_coefficient[0] = ewald_coefficient[0]
                shared_cutoff_radius[0] = cutoff_radius[0]
                shared_sqrt_pi[0] = sqrt_pi = math.sqrt(math.pi)
        elif thread_y == 1:
            for i in range(MAX_NUM_BONDED_PARTICLES):
                shared_bonded_particles[thread_x, i] = bonded_particles[id1, i]
        cuda.syncthreads()
        cell_id_x = particle_cell_index[id1, 0] + neighbor_cell_template[cell_id, 0]
        cell_id_x = cell_id_x - shared_num_cell_vec[0] if cell_id_x >= shared_num_cell_vec[0] else cell_id_x
        cell_id_y = particle_cell_index[id1, 1] + neighbor_cell_template[cell_id, 1]
        cell_id_y = cell_id_y - shared_num_cell_vec[1] if cell_id_y >= shared_num_cell_vec[1] else cell_id_y
        cell_id_z = particle_cell_index[id1, 2] + neighbor_cell_template[cell_id, 2]
        cell_id_z = cell_id_z - shared_num_cell_vec[2] if cell_id_z >= shared_num_cell_vec[2] else cell_id_z
        # id1 attribute
        positions_id1_x = positions[id1, 0]
        positions_id1_y = positions[id1, 1]
        positions_id1_z = positions[id1, 2]
        force_x = 0
        force_y = 0
        force_z = 0
        energy = 0
        e1 = charges[id1]
        for index in range(num_particles_per_cell):
            id2 = cell_list[cell_id_x, cell_id_y, cell_id_z, index]
            if id1 == id2:
                continue
            if id2 == -1:
                break
            is_continue = False
            for i in shared_bonded_particles[thread_x, :]:
                if i == -1:
                    break
                if id2 == i: # Bonded particle
                    x = (positions[id2, 0] - positions_id1_x)
                    if x >= shared_half_pbc_matrix[0]:
                        x -= shared_pbc_matrix[0]
                    elif x <= -shared_half_pbc_matrix[0]:
                        x += shared_pbc_matrix[0]
                    y = (positions[id2, 1] - positions_id1_y)
                    if y >= shared_half_pbc_matrix[1]:
                        y -= shared_pbc_matrix[1]
                    elif y <= -shared_half_pbc_matrix[1]:
                        y += shared_pbc_matrix[1]
                    z = (positions[id2, 2] - positions_id1_z)
                    if z >= shared_half_pbc_matrix[2]:
                        z -= shared_pbc_matrix[2]
                    elif z <= -shared_half_pbc_matrix[2]:
                        z += shared_pbc_matrix[2]
                    r = math.sqrt(x**2 + y**2 + z**2)
                    e1e2 = e1 * charges[id2]
                    scaled_x, scaled_y, scaled_z = x / r, y / r, z / r
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
            if is_continue:
                continue
            x = (positions[id2, 0] - positions_id1_x)
            if x >= shared_half_pbc_matrix[0]:
                x -= shared_pbc_matrix[0]
            elif x <= -shared_half_pbc_matrix[0]:
                x += shared_pbc_matrix[0]
            y = (positions[id2, 1] - positions_id1_y)
            if y >= shared_half_pbc_matrix[1]:
                y -= shared_pbc_matrix[1]
            elif y <= -shared_half_pbc_matrix[1]:
                y += shared_pbc_matrix[1]
            z = (positions[id2, 2] - positions_id1_z)
            if z >= shared_half_pbc_matrix[2]:
                z -= shared_pbc_matrix[2]
            elif z <= -shared_half_pbc_matrix[2]:
                z += shared_pbc_matrix[2]
            r = math.sqrt(x**2 + y**2 + z**2)
            if r <= shared_cutoff_radius[0]:
                e1e2 = e1 * charges[id2]
                scaled_x, scaled_y, scaled_z = x / r, y / r, z / r
                ewald_r = shared_ewald_coefficient[0]*r
                erfc = math.erfc(ewald_r)
                force_val = - e1e2 * (
                    2*shared_ewald_coefficient[0]*math.exp(-(ewald_r)**2) / shared_sqrt_pi[0] + erfc / r
                ) / shared_k[0] / r
                force_x += scaled_x * force_val
                force_y += scaled_y * force_val
                force_z += scaled_z * force_val
                energy += e1e2 * erfc / shared_k[0] / r / 2
        cuda.atomic.add(forces, (id1, 0), force_x)
        cuda.atomic.add(forces, (id1, 1), force_y)
        cuda.atomic.add(forces, (id1, 2), force_z)
        cuda.atomic.add(potential_energy, 0, energy)

    @staticmethod
    def _update_bspline_kernel(positions, grid_size, pbc_matrix, pbc_inv):
        num_particles = positions.shape[0]
        # Change positions from [-0.5, 0.5] pbc_matrix to [0, 1.0]
        positions = positions + np.diagonal(pbc_matrix) / 2
        # spline_coefficient: [num_particles, SPATIAL_DIM, PME_ORDER] The spline coefficient of particles
        spline_coefficient = np.zeros((num_particles, SPATIAL_DIM, PME_ORDER))
        # spline_derivative_coefficient: [num_particles, SPATIAL_DIM, PME_ORDER] The derivative spline coefficient of particles
        spline_derivative_coefficient = np.zeros((num_particles, SPATIAL_DIM, PME_ORDER))
        scaled_positions = np.dot(positions, pbc_inv)
        # grid_indice: [num_particles, SPATIAL_DIM] The index of grid where particle assigned to
        grid_indice = scaled_positions * grid_size
        # grid_friction: [num_particles, SPATIAL_DIM] The fraction part of particles position relative to gird
        grid_fraction = grid_indice - np.floor(grid_indice)
        grid_indice -= grid_fraction

        # 3 order B-spline
        spline_coefficient[:, :, 2] = 0.5 * grid_fraction**2
        spline_coefficient[:, :, 0] = 0.5 * (1 - grid_fraction)**2
        spline_coefficient[:, :, 1] = 1 - spline_coefficient[:, :, 2] - spline_coefficient[:, :, 0]

        # PME_ORDER order derivative coefficient
        spline_derivative_coefficient[:, :, 0] = - spline_coefficient[:, :, 0]
        spline_derivative_coefficient[:, :, 1] = spline_coefficient[:, :, 0] - spline_coefficient[:, :, 1]
        spline_derivative_coefficient[:, :, 2] = spline_coefficient[:, :, 1] - spline_coefficient[:, :, 2]
        spline_derivative_coefficient[:, :, 3] = spline_coefficient[:, :, 2]
        # PME_ORDER order spline coefficient
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
            cur_axis_map[cur_axis_map<0] += grid_size[i]
            cur_axis_map[cur_axis_map>=grid_size[i]] -= grid_size[i]
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
        for i in range(PME_ORDER):
            grid_x = grid_map[particle_id, 0, i]
            spline_coefficient_x = spline_coefficient[particle_id, 0, i]
            for j in range(PME_ORDER):
                grid_y = grid_map[particle_id, 1, j]
                spline_coefficient_y = spline_coefficient[particle_id, 1, j]
                for k in range(PME_ORDER):
                    grid_z = grid_map[particle_id, 2, k]
                    spline_coefficient_z = spline_coefficient[particle_id, 2, k]
                    grid_charge = (
                        charge * spline_coefficient_x *
                        spline_coefficient_y * spline_coefficient_z
                    )
                    cuda.atomic.add(charge_map, (grid_x, grid_y, grid_z), grid_charge)

    @staticmethod
    def _update_electric_potential_map_kernel(
        k, charge_map, b_grid, c_grid,
    ):
        # Convolution
        fft_charge = fft.fftn(charge_map / k)
        fft_charge[0, 0, 0] = 0
        electric_potential_map = fft.ifftn(fft_charge*b_grid*c_grid).real
        return electric_potential_map

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
            for j in range(PME_ORDER):
                grid_y = grid_map[particle_id, 1, j]
                for k in range(PME_ORDER):
                    grid_z = grid_map[particle_id, 2, k]
                    cur_energy = (
                        spline_coefficient[particle_id, 0, i] *
                        spline_coefficient[particle_id, 1, j] *
                        spline_coefficient[particle_id, 2, k] *
                        electric_potential_map[grid_x, grid_y, grid_z]
                    )
                    force_x -= (
                        cur_energy / spline_coefficient[particle_id, 0, i] *
                        spline_derivative_coefficient[particle_id, 0, i]
                    )
                    force_y -= (
                        cur_energy / spline_coefficient[particle_id, 1, j] *
                        spline_derivative_coefficient[particle_id, 1, j]
                    )
                    force_z -= (
                        cur_energy / spline_coefficient[particle_id, 2, k] *
                        spline_derivative_coefficient[particle_id, 2, k]
                    )
                    energy += cur_energy
        cuda.atomic.add(forces, (particle_id, 0), force_x * charge)
        cuda.atomic.add(forces, (particle_id, 1), force_y * charge)
        cuda.atomic.add(forces, (particle_id, 2), force_z * charge)
        cuda.atomic.add(potential_energy, 0, energy)

    def update(self):
        self._forces = np.zeros_like(self._parent_ensemble.state.positions)
        self._potential_energy = np.zeros([1], dtype=env.NUMPY_FLOAT)
        device_forces = cuda.to_device(self._forces)
        device_potential_energy = cuda.to_device(self._potential_energy)
        # Direct part
        block_per_grid_x = int(np.ceil(
            self._parent_ensemble.topology.num_particles / THREAD_PER_BLOCK[0]
        ))
        block_per_grid_y = int(np.ceil(
            NUM_NEIGHBOR_CELLS / THREAD_PER_BLOCK[1]
        ))
        block_per_grid = (block_per_grid_x, block_per_grid_y)
        block_per_grid = (block_per_grid_x, block_per_grid_y)
        self._update_direct_part[block_per_grid, THREAD_PER_BLOCK](
            self._parent_ensemble.state.device_positions, self._device_charges,
            self._device_k, self._device_ewald_coefficient,
            self._device_cutoff_radius, self._parent_ensemble.state.device_pbc_matrix,
            self._device_bonded_particles,
            self._parent_ensemble.state.cell_list.device_particle_cell_index,
            self._parent_ensemble.state.cell_list.device_cell_list,
            self._parent_ensemble.state.cell_list.device_num_cell_vec,
            self._device_neighbor_cell_template,
            device_forces, device_potential_energy
        )
        forces_direct = device_forces.copy_to_host()
        potential_energy_direct = device_potential_energy.copy_to_host()[0]
        # Bspline
        spline_coefficient, spline_derivative_coefficient, grid_map = self._update_bspline(
            self._parent_ensemble.state.positions, self._grid_size,
            self._parent_ensemble.state.pbc_matrix, self._parent_ensemble.state.pbc_inv
        )
        device_spline_coefficient = cuda.to_device(spline_coefficient)
        device_grid_map = cuda.to_device(grid_map)
        # Map charge
        thread_per_block = 64
        block_per_grid = int(np.ceil(
            self._parent_ensemble.topology.num_particles / thread_per_block
        ))
        device_charge_map = cuda.to_device(np.zeros(self._grid_size, dtype=env.NUMPY_FLOAT))
        self._update_charge_map[block_per_grid, thread_per_block](
            self._device_num_particles, device_spline_coefficient, device_grid_map,
            self._device_charges, device_charge_map
        )
        # Reciprocal convolution
        electric_potential_map = self._update_electric_potential_map(
            self._k, device_charge_map.copy_to_host(), self._b_grid, self._c_grid
        )
        # Reciprocal force
        device_spline_derivative_coefficient = cuda.to_device(spline_derivative_coefficient)
        device_electric_potential_map = cuda.to_device(electric_potential_map.astype(env.NUMPY_FLOAT))
        device_forces = cuda.to_device(np.zeros_like(forces_direct, dtype=env.NUMPY_FLOAT))
        device_potential_energy = cuda.to_device(np.zeros([1], dtype=env.NUMPY_FLOAT))
        self._update_reciprocal_force[block_per_grid, thread_per_block](
            self._device_num_particles,
            device_spline_coefficient,
            device_spline_derivative_coefficient,
            device_grid_map,
            device_electric_potential_map,
            self._device_charges,
            device_forces,
            device_potential_energy
        )
        forces_reciprocal = device_forces.copy_to_host()
        forces_reciprocal = forces_reciprocal * self._grid_size / np.diagonal(self._parent_ensemble.state.pbc_matrix)
        forces_reciprocal -= forces_reciprocal.mean(0)
        potential_energy_resciprocal = device_potential_energy.copy_to_host()[0]
        # Summary
        self._potential_energy = potential_energy_direct + potential_energy_resciprocal - self._potential_energy_self
        self._forces = forces_direct + forces_reciprocal

    @property
    def grid_size(self):
        return self._grid_size

    @property
    def ewald_coefficient(self):
        return self._ewald_coefficient
