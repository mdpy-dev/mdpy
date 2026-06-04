from __future__ import annotations

import numpy as np
import cupy as cp

from mdpy.force.force_term import ForceTerm
from mdpy.force.pme_parameters import PMEParameters
from mdpy.force.pme_bspline import (
    get_spread_kernel,
    get_cell_spread_kernel,
    get_gather_kernel,
    get_self_energy_kernel,
    get_exclusion_kernel,
    precompute_bk_factors,
)

COULOMB_CONST = 0.13893556595455
SQRT_PI = 1.772453850905516


class PMEReciprocalForce(ForceTerm):
    name = 'pme_reciprocal'

    def __init__(self, pme_params: PMEParameters, cutoff: float):
        self.pme_params = pme_params
        self.cutoff = cutoff
        self.alpha = pme_params.alpha
        self.order = pme_params.order
        self.grid_x = pme_params.grid_x
        self.grid_y = pme_params.grid_y
        self.grid_z = pme_params.grid_z

        self._d_charges = None
        self._d_bk_factors = None
        self._d_charge_grid = None
        self._self_energy_factor = 0.0

        self._d_pair_i = None
        self._d_pair_j = None
        self._d_pair_scale = None
        self._num_exclusion_pairs = 0

        self._box_x = 0.0
        self._box_y = 0.0
        self._box_z = 0.0

        self._N = 0
        self._fft_warmed = False
        self._subgrid_initialized = False

    def bind(self, topology, parameter_table, pbc_matrix=None):
        N = topology.num_particles
        self._N = N

        charges = parameter_table.particle_parameters['charge'].astype(np.float32)
        self._d_charges = cp.asarray(charges)

        grid_size = self.grid_x * self.grid_y * self.grid_z
        self._d_charge_grid = cp.zeros(grid_size, dtype=np.float32)

        if pbc_matrix is not None:
            pbc_2d = np.asarray(pbc_matrix, dtype=np.float64).reshape(3, 3)
            box_x = abs(float(pbc_2d[0, 0]))
            box_y = abs(float(pbc_2d[1, 1]))
            box_z = abs(float(pbc_2d[2, 2]))
        else:
            box_x = self._box_x
            box_y = self._box_y
            box_z = self._box_z

        self._box_x = box_x
        self._box_y = box_y
        self._box_z = box_z

        bk = precompute_bk_factors(
            self.alpha, self.grid_x, self.grid_y, self.grid_z,
            self.order, box_x, box_y, box_z,
        )
        self._d_bk_factors = cp.asarray(bk)

        self._self_energy_factor = (
            -COULOMB_CONST * self.alpha / SQRT_PI
            * float(np.sum(charges.astype(np.float64) ** 2))
        )

        self._build_exclusion_arrays(topology)

        self._warm_fft()

    def _build_exclusion_arrays(self, topology):
        offset = topology.exclusion_offset
        neighbors = topology.exclusion_neighbors
        scale = topology.exclusion_scale

        pair_i = []
        pair_j = []
        pair_scale = []
        N = topology.num_particles
        for i in range(N):
            start = int(offset[i])
            end = int(offset[i + 1])
            for idx in range(start, end):
                j = int(neighbors[idx])
                if j > i:
                    pair_i.append(i)
                    pair_j.append(j)
                    pair_scale.append(float(scale[idx]))

        self._num_exclusion_pairs = len(pair_i)
        if self._num_exclusion_pairs > 0:
            self._d_pair_i = cp.asarray(np.array(pair_i, dtype=np.int32))
            self._d_pair_j = cp.asarray(np.array(pair_j, dtype=np.int32))
            self._d_pair_scale = cp.asarray(np.array(pair_scale, dtype=np.float32))

    def _warm_fft(self):
        if self._fft_warmed:
            return
        dummy = cp.zeros((self.grid_x, self.grid_y, self.grid_z), dtype=np.float32)
        fft = cp.fft.rfftn(dummy)
        cp.fft.irfftn(fft, s=(self.grid_x, self.grid_y, self.grid_z))
        self._fft_warmed = True

    def compute(self, gpu_context, block_list=None, compute_energy=True):
        N = self._N
        order = self.order
        gx, gy, gz = self.grid_x, self.grid_y, self.grid_z
        box_x = gpu_context._box_x
        box_y = gpu_context._box_y
        box_z = gpu_context._box_z

        tpb = 256
        grid_1d = ((N + tpb - 1) // tpb,)

        self._d_charge_grid[:] = 0

        use_cell_spread = (
            block_list is not None
            and hasattr(block_list, 'nc_total')
            and block_list.nc_total > 0
            and hasattr(block_list, '_sorted_positions')
            and block_list._sorted_positions is not None
        )

        if use_cell_spread:
            if not self._subgrid_initialized:
                block_list.compute_pme_subgrid_dims(gx, gy, gz, order)
                self._subgrid_initialized = True

            sorted_pos_x, sorted_pos_y, sorted_pos_z = block_list._sorted_positions
            sorted_charges = self._d_charges[block_list.d_sorted_to_pdb]

            cell_spread_k = get_cell_spread_kernel()
            shmem = block_list._subgrid_total * 4
            cell_spread_k(
                (block_list.nc_total,), (tpb,),
                (sorted_pos_x, sorted_pos_y, sorted_pos_z, sorted_charges,
                 block_list.d_cell_block_offset, block_list.d_cell_block_count, block_list.d_block_atoms,
                 np.int32(N),
                 np.float32(gpu_context._inv_box_x),
                 np.float32(gpu_context._inv_box_y),
                 np.float32(gpu_context._inv_box_z),
                 np.int32(gx), np.int32(gy), np.int32(gz),
                 np.int32(block_list.nc_x), np.int32(block_list.nc_y), np.int32(block_list.nc_z),
                 np.int32(block_list._subgrid_dx), np.int32(block_list._subgrid_dy), np.int32(block_list._subgrid_dz),
                 np.int32(order),
                 self._d_charge_grid),
                shared_mem=shmem,
            )
        else:
            spread_k = get_spread_kernel()
            spread_k(grid_1d, (tpb,),
                (gpu_context.d_wrapped_positions_x,
                 gpu_context.d_wrapped_positions_y,
                 gpu_context.d_wrapped_positions_z,
                 self._d_charges,
                 np.int32(N),
                 np.float32(gpu_context._inv_box_x),
                 np.float32(gpu_context._inv_box_y),
                 np.float32(gpu_context._inv_box_z),
                 np.int32(gx), np.int32(gy), np.int32(gz), np.int32(order),
                 self._d_charge_grid))

        grid_3d = self._d_charge_grid.reshape(gx, gy, gz)
        grid_complex = cp.fft.rfftn(grid_3d)
        grid_complex = grid_complex * self._d_bk_factors
        grid_3d = cp.fft.irfftn(grid_complex, s=(gx, gy, gz))
        self._d_charge_grid = grid_3d.ravel()

        gather_k = get_gather_kernel()
        gather_k(grid_1d, (tpb,),
            (gpu_context.d_wrapped_positions_x,
             gpu_context.d_wrapped_positions_y,
             gpu_context.d_wrapped_positions_z,
             self._d_charges,
             np.int32(N),
             np.float32(gpu_context._inv_box_x),
             np.float32(gpu_context._inv_box_y),
             np.float32(gpu_context._inv_box_z),
             np.int32(gx), np.int32(gy), np.int32(gz), np.int32(order),
             self._d_charge_grid,
             gpu_context.d_forces_x, gpu_context.d_forces_y, gpu_context.d_forces_z,
             gpu_context.d_energy))

        self_k = get_self_energy_kernel()
        self_k((1,), (1,),
            (np.float32(self._self_energy_factor), gpu_context.d_energy))

        if self._num_exclusion_pairs > 0:
            num_pairs = self._num_exclusion_pairs
            pair_grid = ((num_pairs + tpb - 1) // tpb,)
            excl_k = get_exclusion_kernel()
            excl_k(pair_grid, (tpb,),
                (gpu_context.d_wrapped_positions_x,
                 gpu_context.d_wrapped_positions_y,
                 gpu_context.d_wrapped_positions_z,
                 self._d_charges,
                 self._d_pair_i, self._d_pair_j, self._d_pair_scale,
                 np.int32(num_pairs), np.float32(self.alpha),
                 np.float32(box_x), np.float32(box_y), np.float32(box_z),
                 gpu_context.d_forces_x, gpu_context.d_forces_y, gpu_context.d_forces_z,
                 gpu_context.d_energy))
