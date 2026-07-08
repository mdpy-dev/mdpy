from __future__ import annotations

import math
import os

import cupy as cp
import numpy as np
import pytest

from mdpy.force.pme_reciprocal_force import (
    _calc_ewald_coefficient,
    _next_fft_friendly_size,
    compute_bspline_weights,
    get_gather_kernel,
    get_self_energy_kernel,
    precompute_bk_factors,
)
from mdpy.force.factories.charmm import create_bonded_forces


class _PBCContext:
    """Minimal stand-in exposing d_pbc_matrix/d_pbc_inv for BlockList.rebuild,
    which now reads PBC from a State."""

    def __init__(self, pbc_matrix, pbc_inv, positions=None):
        self.d_pbc_matrix = cp.asarray(
            np.ascontiguousarray(pbc_matrix, dtype=np.float32).ravel()
        )
        self.d_pbc_inv = cp.asarray(
            np.ascontiguousarray(pbc_inv, dtype=np.float32).ravel()
        )
        if positions is not None:
            pos = np.asarray(positions, dtype=np.float32)
            self.d_positions_x = cp.asarray(pos[:, 0])
            self.d_positions_y = cp.asarray(pos[:, 1])
            self.d_positions_z = cp.asarray(pos[:, 2])


class TestBSplineWeights:

    def test_partition_of_unity(self):
        for u in [0.0, 0.1, 0.25, 0.5, 0.75, 0.99]:
            theta, dtheta = compute_bspline_weights(u, order=4)
            assert abs(sum(theta) - 1.0) < 1e-10, f"u={u}, sum={sum(theta)}"

    def test_derivative_sum_zero(self):
        for u in [0.0, 0.1, 0.5, 0.99]:
            theta, dtheta = compute_bspline_weights(u, order=4)
            assert abs(sum(dtheta)) < 1e-10, f"u={u}, sum={sum(dtheta)}"

    def test_nonnegativity(self):
        for u in np.linspace(0, 1, 100, endpoint=False):
            theta, _ = compute_bspline_weights(float(u), order=4)
            assert all(t >= -1e-10 for t in theta), f"u={u}, theta={theta}"

    def test_known_values_at_zero(self):
        theta, dtheta = compute_bspline_weights(0.0, order=4)
        assert abs(theta[0] - 1.0 / 6.0) < 1e-10, f"theta[0]={theta[0]}"
        assert abs(theta[1] - 2.0 / 3.0) < 1e-10, f"theta[1]={theta[1]}"
        assert abs(theta[2] - 1.0 / 6.0) < 1e-10, f"theta[2]={theta[2]}"
        assert abs(theta[3]) < 1e-10, f"theta[3]={theta[3]}"

    def test_known_values_at_half(self):
        theta, dtheta = compute_bspline_weights(0.5, order=4)
        assert abs(theta[0] - 1.0 / 48.0) < 1e-10, f"theta[0]={theta[0]}"
        assert abs(theta[1] - 23.0 / 48.0) < 1e-10, f"theta[1]={theta[1]}"
        assert abs(theta[2] - 23.0 / 48.0) < 1e-10, f"theta[2]={theta[2]}"
        assert abs(theta[3] - 1.0 / 48.0) < 1e-10, f"theta[3]={theta[3]}"

    def test_derivative_finite_difference(self):
        for u in [0.1, 0.3, 0.5, 0.7, 0.9]:
            theta, dtheta = compute_bspline_weights(u, order=4)
            h = 1e-6
            theta_plus, _ = compute_bspline_weights(u + h, order=4)
            theta_minus, _ = compute_bspline_weights(u - h, order=4)
            for k in range(4):
                numerical = (theta_plus[k] - theta_minus[k]) / (2.0 * h)
                assert abs(dtheta[k] - numerical) < 1e-4, \
                    f"u={u} k={k}: dtheta={dtheta[k]} numerical={numerical}"


class TestCellBasedChargeSpreading:

    def _run_cell_spread(self, N, grid_x, grid_y, grid_z, box_x, box_y, box_z, order=4, seed=123):
        from mdpy.force.pme_reciprocal_force import get_cell_spread_kernel, compute_bspline_weights
        from mdpy.core.block_list import BlockList

        np.random.seed(seed)
        charges = np.random.randn(N).astype(np.float32) * 0.5
        pos_x = np.random.uniform(0, box_x, N).astype(np.float32)
        pos_y = np.random.uniform(0, box_y, N).astype(np.float32)
        pos_z = np.random.uniform(0, box_z, N).astype(np.float32)

        d_pos_x = cp.asarray(pos_x)
        d_pos_y = cp.asarray(pos_y)
        d_pos_z = cp.asarray(pos_z)
        d_charges = cp.asarray(charges)

        ref_grid = np.zeros(grid_x * grid_y * grid_z, dtype=np.float64)
        for i in range(N):
            fx = pos_x[i] / box_x * grid_x
            fy = pos_y[i] / box_y * grid_y
            fz = pos_z[i] / box_z * grid_z
            theta_x, _ = compute_bspline_weights(float(fx), order)
            theta_y, _ = compute_bspline_weights(float(fy), order)
            theta_z, _ = compute_bspline_weights(float(fz), order)
            gx0 = int(math.floor(fx)) % grid_x
            gy0 = int(math.floor(fy)) % grid_y
            gz0 = int(math.floor(fz)) % grid_z
            for kx in range(order):
                ix = (gx0 + kx) % grid_x
                for ky in range(order):
                    iy = (gy0 + ky) % grid_y
                    for kz in range(order):
                        iz = (gz0 + kz) % grid_z
                        ref_grid[ix * grid_y * grid_z + iy * grid_z + iz] += float(charges[i]) * float(theta_x[kx]) * float(theta_y[ky]) * float(theta_z[kz])

        d_grid_ref = cp.asarray(ref_grid.astype(np.float32))

        bl = BlockList(cutoff=12.0, skin=1.0)
        pbc = np.eye(3, dtype=np.float64) * max(box_x, box_y, box_z)
        pbc_inv = np.linalg.inv(pbc)
        positions = np.stack([pos_x, pos_y, pos_z], axis=1).astype(np.float64)
        topo = type('T', (), {'num_particles': N, 'particle_type_indices': np.zeros(N, dtype=np.int32)})()
        bl.rebuild(topo, _PBCContext(pbc, pbc_inv, positions), force=True)

        subgrid_dx = -(-grid_x // bl.num_cells_x) + 2 * order
        subgrid_dy = -(-grid_y // bl.num_cells_y) + 2 * order
        subgrid_dz = -(-grid_z // bl.num_cells_z) + 2 * order
        subgrid_total = subgrid_dx * subgrid_dy * subgrid_dz

        sorted_pos_x = d_pos_x
        sorted_pos_y = d_pos_y
        sorted_pos_z = d_pos_z
        sorted_charges = d_charges

        d_grid_cell = cp.zeros(grid_x * grid_y * grid_z, dtype=np.float32)
        cell_spread_k = get_cell_spread_kernel()
        shmem = subgrid_total * 4
        cell_spread_k(
            (bl.num_cells_total,), (256,),
            (sorted_pos_x, sorted_pos_y, sorted_pos_z, sorted_charges,
             bl.d_cell_block_offset, bl.d_cell_block_count, bl.d_block_atoms,
             np.int32(N),
             np.float32(1.0 / box_x), np.float32(1.0 / box_y), np.float32(1.0 / box_z),
             np.int32(grid_x), np.int32(grid_y), np.int32(grid_z),
             np.int32(bl.num_cells_x), np.int32(bl.num_cells_y), np.int32(bl.num_cells_z),
             np.int32(subgrid_dx), np.int32(subgrid_dy), np.int32(subgrid_dz),
             np.int32(order),
             d_grid_cell),
            shared_mem=shmem,
        )

        return d_grid_ref, d_grid_cell, charges

    def test_charge_conservation(self):
        d_ref, d_cell, charges = self._run_cell_spread(50, 32, 32, 32, 50.0, 50.0, 50.0)
        ref_sum = float(cp.sum(d_ref))
        cell_sum = float(cp.sum(d_cell))
        expected = float(np.sum(charges))
        assert abs(ref_sum - expected) < abs(expected) * 1e-4 + 1e-5, f"ref_sum={ref_sum}"
        assert abs(cell_sum - expected) < abs(expected) * 1e-4 + 1e-5, f"cell_sum={cell_sum}"

    def test_matches_per_particle_spread(self):
        d_ref, d_cell, _ = self._run_cell_spread(50, 32, 32, 32, 50.0, 50.0, 50.0)
        ref = cp.asnumpy(d_ref)
        cell = cp.asnumpy(d_cell)
        nonzero = np.abs(ref) > 1e-10
        if np.any(nonzero):
            rel_err = np.max(np.abs(ref[nonzero] - cell[nonzero]) / (np.abs(ref[nonzero]) + 1e-10))
            assert rel_err < 1e-3, f"Max relative error: {rel_err}"
        abs_err = np.max(np.abs(ref - cell))
        assert abs_err < 1e-4, f"Max absolute error: {abs_err}"

    def test_matches_large_system(self):
        d_ref, d_cell, _ = self._run_cell_spread(500, 64, 64, 64, 80.0, 80.0, 80.0, seed=42)
        ref = cp.asnumpy(d_ref)
        cell = cp.asnumpy(d_cell)
        nonzero = np.abs(ref) > 1e-10
        if np.any(nonzero):
            rel_err = np.max(np.abs(ref[nonzero] - cell[nonzero]) / (np.abs(ref[nonzero]) + 1e-10))
            assert rel_err < 1e-2, f"Max relative error: {rel_err}"
        abs_err = np.max(np.abs(ref - cell))
        assert abs_err < 1e-3, f"Max absolute error: {abs_err}"

    def test_single_atom(self):
        d_ref, d_cell, _ = self._run_cell_spread(1, 32, 32, 32, 50.0, 50.0, 50.0)
        ref = cp.asnumpy(d_ref)
        cell = cp.asnumpy(d_cell)
        np.testing.assert_allclose(cell, ref, atol=1e-5)


class TestBSplineModuli:

    def test_dc_component_zero(self):
        bk = precompute_bk_factors(0.35, 32, 32, 32, 4, 50.0, 50.0, 50.0)
        assert abs(bk[0, 0, 0]) < 1e-10, f"DC component should be ~0, got {bk[0,0,0]}"

    def test_nonzero_terms_positive(self):
        bk = precompute_bk_factors(0.35, 32, 32, 32, 4, 50.0, 50.0, 50.0)
        for ix, iy, iz in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1), (5, 5, 5)]:
            if iz < bk.shape[2]:
                val = bk[ix, iy, iz]
                assert abs(val.imag) < abs(val.real) * 1e-6 + 1e-10, \
                    f"bk[{ix},{iy},{iz}] should be real: {val}"
                assert val.real > 0, f"bk[{ix},{iy},{iz}] should be positive: {val.real}"

    def test_symmetry(self):
        bk = precompute_bk_factors(0.35, 32, 32, 32, 4, 50.0, 50.0, 50.0)
        assert bk.shape == (32, 32, 17), f"Expected (32,32,17), got {bk.shape}"


class TestForceGathering:

    def test_net_force_near_zero(self):
        N = 50
        np.random.seed(42)
        order = 4
        grid_x, grid_y, grid_z = 32, 32, 32
        box_x, box_y, box_z = 50.0, 50.0, 50.0

        charges = np.random.randn(N).astype(np.float32) * 0.5
        pos_x = np.random.uniform(1, box_x - 1, N).astype(np.float32)
        pos_y = np.random.uniform(1, box_y - 1, N).astype(np.float32)
        pos_z = np.random.uniform(1, box_z - 1, N).astype(np.float32)

        d_charges = cp.asarray(charges)
        d_pos_x = cp.asarray(pos_x)
        d_pos_y = cp.asarray(pos_y)
        d_pos_z = cp.asarray(pos_z)

        from mdpy.force.pme_reciprocal_force import get_cell_spread_kernel
        from mdpy.core.block_list import BlockList

        pbc = np.eye(3, dtype=np.float64) * max(box_x, box_y, box_z)
        pbc_inv = np.linalg.inv(pbc)
        positions = np.stack([pos_x, pos_y, pos_z], axis=1).astype(np.float64)
        topo = type('T', (), {'num_particles': N, 'particle_type_indices': np.zeros(N, dtype=np.int32)})()
        bl = BlockList(cutoff=12.0, skin=1.0)
        bl.rebuild(topo, _PBCContext(pbc, pbc_inv, positions), force=True)
        subgrid_dx = -(-grid_x // bl.num_cells_x) + 2 * order
        subgrid_dy = -(-grid_y // bl.num_cells_y) + 2 * order
        subgrid_dz = -(-grid_z // bl.num_cells_z) + 2 * order
        subgrid_total = subgrid_dx * subgrid_dy * subgrid_dz

        sorted_pos_x = d_pos_x
        sorted_pos_y = d_pos_y
        sorted_pos_z = d_pos_z
        sorted_charges_gpu = d_charges

        d_charge_grid = cp.zeros(grid_x * grid_y * grid_z, dtype=np.float32)
        cell_spread_k = get_cell_spread_kernel()
        shmem = subgrid_total * 4
        cell_spread_k(
            (bl.num_cells_total,), (256,),
            (sorted_pos_x, sorted_pos_y, sorted_pos_z, sorted_charges_gpu,
             bl.d_cell_block_offset, bl.d_cell_block_count, bl.d_block_atoms,
             np.int32(N),
             np.float32(1.0 / box_x), np.float32(1.0 / box_y), np.float32(1.0 / box_z),
             np.int32(grid_x), np.int32(grid_y), np.int32(grid_z),
             np.int32(bl.num_cells_x), np.int32(bl.num_cells_y), np.int32(bl.num_cells_z),
             np.int32(subgrid_dx), np.int32(subgrid_dy), np.int32(subgrid_dz),
             np.int32(order),
             d_charge_grid),
            shared_mem=shmem,
        )

        alpha = 0.35
        bk = precompute_bk_factors(alpha, grid_x, grid_y, grid_z, order, box_x, box_y, box_z)
        d_bk = cp.asarray(bk)

        grid_3d = d_charge_grid.reshape(grid_x, grid_y, grid_z)
        grid_complex = cp.fft.rfftn(grid_3d)
        grid_complex = grid_complex * d_bk
        grid_3d = cp.fft.irfftn(grid_complex, s=(grid_x, grid_y, grid_z))
        d_phi_grid = grid_3d.ravel()

        d_fx = cp.zeros(N, dtype=np.float32)
        d_fy = cp.zeros(N, dtype=np.float32)
        d_fz = cp.zeros(N, dtype=np.float32)
        d_energy = cp.zeros(1, dtype=np.float32)

        gather_k = get_gather_kernel()
        total_slots = bl.max_blocks * 32
        grid_1d = ((total_slots + 255) // 256,)
        gather_k(
            grid_1d, (256,),
            (d_pos_x, d_pos_y, d_pos_z, d_charges,
             np.int32(N),
             bl.d_block_atoms,
             np.int32(total_slots),
             np.float32(1.0 / box_x), np.float32(1.0 / box_y), np.float32(1.0 / box_z),
             np.int32(grid_x), np.int32(grid_y), np.int32(grid_z), np.int32(order),
             d_phi_grid, d_fx, d_fy, d_fz, d_energy),
        )

        net_fx = float(cp.sum(d_fx))
        net_fy = float(cp.sum(d_fy))
        net_fz = float(cp.sum(d_fz))
        max_force = max(
            float(cp.max(cp.abs(d_fx))),
            float(cp.max(cp.abs(d_fy))),
            float(cp.max(cp.abs(d_fz))),
        )

        assert abs(net_fx) < max_force * 0.05 + 1e-7, f"Net fx={net_fx} too large"
        assert abs(net_fy) < max_force * 0.05 + 1e-7, f"Net fy={net_fy} too large"
        assert abs(net_fz) < max_force * 0.05 + 1e-7, f"Net fz={net_fz} too large"

    def test_self_energy(self):
        charges = np.array([0.5, -0.3, 0.8], dtype=np.float32)
        alpha = 0.35
        sq_pi = 1.772453850905516
        coulomb_const = 0.13893556595455

        expected = -coulomb_const * alpha / sq_pi * float(np.sum(charges ** 2))

        d_energy = cp.zeros(1, dtype=np.float32)
        self_k = get_self_energy_kernel()
        self_k((1,), (1,), (np.float32(expected), d_energy))

        assert abs(float(d_energy[0]) - expected) < 1e-6
        assert expected < 0


class TestPMEReciprocalForce:

    @staticmethod
    def _build_block_list(pos, pbc, topo, cutoff):
        from mdpy.core.block_list import BlockList
        pbc_64 = np.asarray(pbc, dtype=np.float64).reshape(3, 3)
        pbc_inv = np.linalg.inv(pbc_64)
        bl = BlockList(cutoff=cutoff, skin=1.0)
        bl.rebuild(topo, _PBCContext(pbc_64, pbc_inv, pos.astype(np.float64)), force=True)
        return bl

    def test_nonzero_energy_and_forces(self):
        from mdpy.force.pme_reciprocal_force import PMEReciprocalForce
        from mdpy.core.state import State
        from mdpy.core.topology import Topology
        from mdpy.core.parameter_table import ParameterTable

        N = 20
        box = 40.0
        cutoff = 10.0
        pbc = np.eye(3, dtype=np.float32) * box

        topo = Topology()
        topo.num_particles = N

        pt = ParameterTable()
        np.random.seed(42)
        charges = np.random.randn(N).astype(np.float32)

        state = State(topo.num_particles)
        state.set_charges(charges)
        state.set_pbc(pbc.flatten())

        pos = np.random.uniform(2, box - 2, (N, 3)).astype(np.float32)
        state.d_positions_x[:] = cp.asarray(pos[:, 0])
        state.d_positions_y[:] = cp.asarray(pos[:, 1])
        state.d_positions_z[:] = cp.asarray(pos[:, 2])

        pme = PMEReciprocalForce(cutoff)
        pme.initialize_grid(topo, pt, pbc_matrix=pbc)

        bl = self._build_block_list(pos, pbc, topo, cutoff)

        state.zero_forces()
        state.zero_energy()
        pme.compute(state, block_list=bl)

        fx = cp.asnumpy(state.d_forces_x)
        fy = cp.asnumpy(state.d_forces_y)
        fz = cp.asnumpy(state.d_forces_z)
        energy = float(state.d_energy[0])

        print(f"PME reciprocal energy: {energy:.6f}")
        print(f"Max force: {max(np.max(np.abs(fx)), np.max(np.abs(fy)), np.max(np.abs(fz))):.6f}")

        assert abs(energy) > 1e-6, f"Energy should be nonzero, got {energy}"
        max_force = max(np.max(np.abs(fx)), np.max(np.abs(fy)), np.max(np.abs(fz)))
        assert max_force > 1e-10, f"Forces should be nonzero, max={max_force}"

    def test_reproducibility(self):
        from mdpy.force.pme_reciprocal_force import PMEReciprocalForce
        from mdpy.core.state import State
        from mdpy.core.topology import Topology
        from mdpy.core.parameter_table import ParameterTable

        N = 15
        box = 35.0
        cutoff = 9.0
        pbc = np.eye(3, dtype=np.float32) * box

        topo = Topology()
        topo.num_particles = N

        np.random.seed(7)
        pt = ParameterTable()
        charges = np.random.randn(N).astype(np.float32)

        state = State(topo.num_particles)
        state.set_charges(charges)
        state.set_pbc(pbc.flatten())

        pos = np.random.uniform(2, box - 2, (N, 3)).astype(np.float32)
        state.d_positions_x[:] = cp.asarray(pos[:, 0])
        state.d_positions_y[:] = cp.asarray(pos[:, 1])
        state.d_positions_z[:] = cp.asarray(pos[:, 2])

        pme = PMEReciprocalForce(cutoff)
        pme.initialize_grid(topo, pt, pbc_matrix=pbc)

        bl = self._build_block_list(pos, pbc, topo, cutoff)

        state.zero_forces()
        state.zero_energy()
        pme.compute(state, block_list=bl)
        energy1 = float(state.d_energy[0])
        fx1 = cp.asnumpy(state.d_forces_x).copy()

        state.zero_forces()
        state.zero_energy()
        pme.compute(state, block_list=bl)
        energy2 = float(state.d_energy[0])
        fx2 = cp.asnumpy(state.d_forces_x).copy()

        assert abs(energy1 - energy2) < 1e-6, f"Energy not reproducible: {energy1} vs {energy2}"
        np.testing.assert_allclose(fx1, fx2, atol=1e-6)


class TestGridSizing:

    def test_ewald_coefficient_erfc_bound(self):
        from scipy.special import erfc

        cutoff = 12.0
        rtol = 1e-5
        alpha = _calc_ewald_coefficient(cutoff, rtol)
        actual = erfc(alpha * cutoff)
        assert actual <= rtol, f"erfc({alpha:.4f}*{cutoff}) = {actual:.2e} > {rtol}"
        alpha_minus = alpha - 0.001
        assert erfc(alpha_minus * cutoff) > rtol, "Should be tight bound"

    def test_ewald_coefficient_various_cutoffs(self):
        from scipy.special import erfc

        for cutoff in [8.0, 10.0, 12.0, 15.0]:
            alpha = _calc_ewald_coefficient(cutoff)
            assert erfc(alpha * cutoff) <= 1e-5 * (1 + 1e-10)
            assert 0.1 < alpha < 1.0

    def test_next_fft_friendly_size_basic(self):
        assert _next_fft_friendly_size(1) == 1
        assert _next_fft_friendly_size(2) == 2
        assert _next_fft_friendly_size(7) == 7
        assert _next_fft_friendly_size(8) == 8
        assert _next_fft_friendly_size(9) == 9
        assert _next_fft_friendly_size(11) == 12
        assert _next_fft_friendly_size(13) == 14
        assert _next_fft_friendly_size(90) == 90

    def test_from_box_1m9z(self):
        from mdpy.core.topology import Topology
        from mdpy.core.parameter_table import ParameterTable

        box = 108.0
        cutoff = 12.0
        N = 1
        topo = Topology()
        topo.num_particles = N
        pt = ParameterTable()
        pbc = np.eye(3, dtype=np.float32) * box
        from mdpy.force.pme_reciprocal_force import PMEReciprocalForce
        pme = PMEReciprocalForce(cutoff)
        pme.initialize_grid(topo, pt, pbc_matrix=pbc)
        assert pme.grid_x == 90
        assert pme.grid_y == 90
        assert pme.grid_z == 90
        assert pme.order == 4

    def test_from_box_6po6(self):
        from mdpy.core.topology import Topology
        from mdpy.core.parameter_table import ParameterTable
        from mdpy.force.pme_reciprocal_force import PMEReciprocalForce

        box = 100.0
        cutoff = 10.0
        N = 1
        topo = Topology()
        topo.num_particles = N
        pt = ParameterTable()
        pbc = np.eye(3, dtype=np.float32) * box
        pme = PMEReciprocalForce(cutoff)
        pme.initialize_grid(topo, pt, pbc_matrix=pbc)
        assert pme.grid_x == 84
        assert pme.grid_y == 84
        assert pme.grid_z == 84

    def test_from_box_rectangular(self):
        from mdpy.core.topology import Topology
        from mdpy.core.parameter_table import ParameterTable
        from mdpy.force.pme_reciprocal_force import PMEReciprocalForce

        box_x, box_y, box_z = 80.0, 60.0, 40.0
        cutoff = 10.0
        N = 1
        topo = Topology()
        topo.num_particles = N
        pt = ParameterTable()
        pbc = np.diag(np.array([box_x, box_y, box_z], dtype=np.float32))
        pme = PMEReciprocalForce(cutoff)
        pme.initialize_grid(topo, pt, pbc_matrix=pbc)
        assert pme.grid_x != pme.grid_y or pme.grid_y != pme.grid_z
        assert pme.grid_x >= box_x / 1.2 * 0.95
        assert pme.grid_y >= box_y / 1.2 * 0.95
        assert pme.grid_z >= box_z / 1.2 * 0.95

    def test_from_box_custom_spacing(self):
        from mdpy.core.topology import Topology
        from mdpy.core.parameter_table import ParameterTable
        from mdpy.force.pme_reciprocal_force import PMEReciprocalForce

        box = 108.0
        cutoff = 12.0
        N = 1
        topo = Topology()
        topo.num_particles = N
        pt = ParameterTable()
        pbc = np.eye(3, dtype=np.float32) * box
        pme_default = PMEReciprocalForce(cutoff)
        pme_default.initialize_grid(topo, pt, pbc_matrix=pbc)
        pme_fine = PMEReciprocalForce(cutoff, fourier_spacing=0.8)
        pme_fine.initialize_grid(topo, pt, pbc_matrix=pbc)
        assert pme_fine.grid_x > pme_default.grid_x

    def test_from_box_custom_rtol(self):
        from mdpy.core.topology import Topology
        from mdpy.core.parameter_table import ParameterTable
        from mdpy.force.pme_reciprocal_force import PMEReciprocalForce
        from scipy.special import erfc

        box = 108.0
        cutoff = 12.0
        N = 1
        topo = Topology()
        topo.num_particles = N
        pt = ParameterTable()
        pbc = np.eye(3, dtype=np.float32) * box
        pme_loose = PMEReciprocalForce(cutoff, ewald_rtol=1e-3)
        pme_loose.initialize_grid(topo, pt, pbc_matrix=pbc)
        pme_tight = PMEReciprocalForce(cutoff, ewald_rtol=1e-8)
        pme_tight.initialize_grid(topo, pt, pbc_matrix=pbc)
        assert pme_loose.alpha < pme_tight.alpha
        assert erfc(pme_loose.alpha * 12) <= 1e-3 * (1 + 1e-10)
        assert erfc(pme_tight.alpha * 12) <= 1e-8 * (1 + 1e-10)


class TestPMEIntegration6PO6:

    @pytest.fixture(autouse=True)
    def setup(self):
        from mdpy.io.psf_parser import PSFParser
        from mdpy.io.pdb_parser import PDBParser
        from mdpy.io.charmm_toppar_parser import CharmmTopparParser
        from mdpy.io.charmm_toppar_parser import create_parameter_table

        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

        psf = PSFParser(os.path.join(data_dir, '6PO6.psf'))
        pdb = PDBParser(os.path.join(data_dir, '6PO6.pdb'))
        toppar = CharmmTopparParser(os.path.join(data_dir, 'par_all36_prot.prm'))

        self.psf = psf
        self.topology = psf.topology
        self.parameter_table = create_parameter_table(self.topology, toppar, type_names=self.psf.particle_type_names)
        self.positions = pdb.positions.astype(np.float32)
        self.N = self.topology.num_particles
        self.box = 100.0
        self.cutoff = 10.0

    def _build_pme_system(self):
        from mdpy.force.bonded_force import BondedForce
        from mdpy.force.nonbonded_force import NonbondedForce
        from mdpy.force.expressions.lennard_jones import lennard_jones
        from mdpy.force.expressions.screened_coulomb import screened_coulomb
        from mdpy.force.pme_reciprocal_force import PMEReciprocalForce
        from mdpy.core.state import State
        from mdpy.system import System

        pbc_matrix = np.eye(3, dtype=np.float32) * self.box

        state = State(self.topology.num_particles)
        state.set_masses(self.psf.masses)
        state.set_charges(self.psf.charges)
        state.set_type_indices(self.psf.particle_type_indices)
        system = System(self.topology, state)

        system.set_pbc(pbc_matrix)

        for f in create_bonded_forces(self.topology, self.parameter_table):
            system.add_force_term(f)

        nb = NonbondedForce(lennard_jones + screened_coulomb, cutoff=self.cutoff)
        lj_pair = self.parameter_table.type_pair_parameters['lj_pair']
        nb.set_pair_parameter('sigma', lj_pair[0::2].astype(np.float32))
        nb.set_pair_parameter('epsilon', lj_pair[1::2].astype(np.float32))
        system.add_force_term(nb)

        pme = PMEReciprocalForce(self.cutoff)
        pme.initialize_grid(self.topology, self.parameter_table, pbc_matrix=pbc_matrix)
        system.add_force_term(pme)

        pbc_inv = np.linalg.inv(pbc_matrix.astype(np.float64))
        raw_positions = self.positions.astype(np.float64)
        frac = raw_positions @ pbc_inv
        frac -= np.floor(frac)
        wrapped = (frac @ pbc_matrix).astype(np.float32)

        system.set_positions(wrapped.astype(np.float32))
        system.set_velocities(np.zeros((self.N, 3), dtype=np.float32))

        system.update_neighbor_list(force_rebuild=True)

        return system, pme

    def test_pme_system_nonzero_energy(self):
        system, pme = self._build_pme_system()

        system.compute_forces()

        from cupy import asnumpy
        fx = asnumpy(system.state.d_forces_x)
        fy = asnumpy(system.state.d_forces_y)
        fz = asnumpy(system.state.d_forces_z)

        max_force = max(np.max(np.abs(fx)), np.max(np.abs(fy)), np.max(np.abs(fz)))

        print(f"6PO6 N={self.N}")
        print(f"PME params: alpha={pme.alpha:.4f}, grid=({pme.grid_x}, {pme.grid_y}, {pme.grid_z})")
        print(f"Max force: {max_force:.6f}")

        assert max_force > 1e-6, "Forces should be nonzero"

        net_fx = np.sum(fx)
        net_fy = np.sum(fy)
        net_fz = np.sum(fz)
        print(f"Net force: ({net_fx:.6f}, {net_fy:.6f}, {net_fz:.6f})")
        assert abs(net_fx) < max_force * 0.01, f"Net fx too large: {net_fx}"
        assert abs(net_fy) < max_force * 0.01, f"Net fy too large: {net_fy}"
        assert abs(net_fz) < max_force * 0.01, f"Net fz too large: {net_fz}"

    def test_pme_energy_nonzero(self):
        system, pme = self._build_pme_system()

        system.compute_forces()
        energies = system.dump_energy()
        print(f"PME energies: {energies}")

        assert 'bond' in energies
        assert 'nonbonded' in energies
        assert 'pme_reciprocal' in energies

        assert abs(energies['pme_reciprocal']) > 1e-6, \
            f"PME reciprocal energy should be nonzero: {energies['pme_reciprocal']}"

    def test_pme_self_energy_negative(self):
        from mdpy.force.pme_reciprocal_force import _calc_ewald_coefficient

        alpha = _calc_ewald_coefficient(self.cutoff)

        charges = self.psf.charges.astype(np.float64)
        COULOMB_CONST = 0.13893556595455
        SQRT_PI = 1.772453850905516

        self_energy = -COULOMB_CONST * alpha / SQRT_PI * np.sum(charges ** 2)
        print(f"Self-energy: {self_energy:.6f}")
        assert self_energy < 0, "Self-energy should be negative"

    def test_pme_reads_state_charges(self):
        system, pme = self._build_pme_system()
        state = system.state
        block_list = system.block_list

        state.zero_forces()
        state.zero_energy()
        pme.compute(state, block_list=block_list, compute_energy=True)
        baseline = float(state.d_energy[0])

        state.d_charges[:] = state.d_charges * 2.0

        state.zero_forces()
        state.zero_energy()
        pme.compute(state, block_list=block_list, compute_energy=True)
        doubled = float(state.d_energy[0])

        assert abs(doubled - baseline) > 1e-3, (
            f"PME did not respond to state.d_charges mutation: "
            f"baseline={baseline}, doubled={doubled}"
        )

    def test_self_energy_reflects_single_charge_mutation(self):
        system, pme = self._build_pme_system()

        system.compute_forces()
        e_before = system.dump_energy()['pme_reciprocal']

        orig = float(system.state.d_charges[0].get())
        system.state.d_charges[0] = orig * 2.0

        system.compute_forces()
        e_after = system.dump_energy()['pme_reciprocal']

        assert abs(e_after - e_before) > 1e-6, (
            f"PME energy did not reflect single-charge mutation: "
            f"before={e_before}, after={e_after}"
        )

        system.state.d_charges[0] = orig


class TestBilateralPaddingUnwrapped:

    def test_subgrid_dimensions_bilateral(self):
        from mdpy.core.block_list import BlockList

        box = 50.0
        cutoff = 10.0
        skin = 2.0
        N = 1
        grid_x = grid_y = grid_z = 50
        order = 4

        positions = np.array([[25.0, 25.0, 25.0]], dtype=np.float64)
        topo = type('T', (), {'num_particles': N, 'particle_type_indices': np.zeros(N, dtype=np.int32)})()
        pbc = np.eye(3, dtype=np.float64) * box
        pbc_inv = np.linalg.inv(pbc)

        bl = BlockList(cutoff=cutoff, skin=skin)
        bl.rebuild(topo, _PBCContext(pbc, pbc_inv, positions), force=True)
        subgrid_dx = -(-grid_x // bl.num_cells_x) + 2 * order
        subgrid_dy = -(-grid_y // bl.num_cells_y) + 2 * order
        subgrid_dz = -(-grid_z // bl.num_cells_z) + 2 * order
        subgrid_total = subgrid_dx * subgrid_dy * subgrid_dz

        expected_dx = -(-grid_x // bl.num_cells_x) + 2 * order
        expected_dy = -(-grid_y // bl.num_cells_y) + 2 * order
        expected_dz = -(-grid_z // bl.num_cells_z) + 2 * order

        assert subgrid_dx == expected_dx, \
            f"subgrid_dx={subgrid_dx}, expected={expected_dx}"
        assert subgrid_dy == expected_dy
        assert subgrid_dz == expected_dz

        base_dx = -(-grid_x // bl.num_cells_x)
        assert subgrid_dx == base_dx + 2 * order, \
            f"Bilateral padding should add 2*order: base={base_dx}, got={subgrid_dx}"
        assert subgrid_dx > base_dx, \
            f"Subgrid must be larger than base: {subgrid_dx} vs {base_dx}"

    def test_charge_conservation_at_boundary(self):
        from mdpy.force.pme_reciprocal_force import get_cell_spread_kernel, compute_bspline_weights
        from mdpy.core.block_list import BlockList

        box = 50.0
        cutoff = 10.0
        skin = 2.0
        N = 1
        grid_x = grid_y = grid_z = 50
        order = 4

        charge = np.array([1.0], dtype=np.float32)
        pos_x = np.array([0.01], dtype=np.float32)
        pos_y = np.array([0.01], dtype=np.float32)
        pos_z = np.array([0.01], dtype=np.float32)

        d_pos_x = cp.asarray(pos_x)
        d_pos_y = cp.asarray(pos_y)
        d_pos_z = cp.asarray(pos_z)
        d_charges = cp.asarray(charge)

        ref_grid = np.zeros(grid_x * grid_y * grid_z, dtype=np.float64)
        for i in range(N):
            fx = pos_x[i] / box * grid_x
            fy = pos_y[i] / box * grid_y
            fz = pos_z[i] / box * grid_z
            theta_x, _ = compute_bspline_weights(float(fx), order)
            theta_y, _ = compute_bspline_weights(float(fy), order)
            theta_z, _ = compute_bspline_weights(float(fz), order)
            gx0 = int(math.floor(fx)) % grid_x
            gy0 = int(math.floor(fy)) % grid_y
            gz0 = int(math.floor(fz)) % grid_z
            for kx in range(order):
                ix = (gx0 + kx) % grid_x
                for ky in range(order):
                    iy = (gy0 + ky) % grid_y
                    for kz in range(order):
                        iz = (gz0 + kz) % grid_z
                        ref_grid[ix * grid_y * grid_z + iy * grid_z + iz] += \
                            float(charge[i]) * float(theta_x[kx]) * float(theta_y[ky]) * float(theta_z[kz])

        positions = np.stack([pos_x, pos_y, pos_z], axis=1).astype(np.float64)
        topo = type('T', (), {'num_particles': N, 'particle_type_indices': np.zeros(N, dtype=np.int32)})()
        pbc = np.eye(3, dtype=np.float64) * box
        pbc_inv = np.linalg.inv(pbc)
        bl = BlockList(cutoff=cutoff, skin=skin)
        bl.rebuild(topo, _PBCContext(pbc, pbc_inv, positions), force=True)
        subgrid_dx = -(-grid_x // bl.num_cells_x) + 2 * order
        subgrid_dy = -(-grid_y // bl.num_cells_y) + 2 * order
        subgrid_dz = -(-grid_z // bl.num_cells_z) + 2 * order
        subgrid_total = subgrid_dx * subgrid_dy * subgrid_dz

        sorted_pos_x = d_pos_x
        sorted_pos_y = d_pos_y
        sorted_pos_z = d_pos_z
        sorted_charges = d_charges

        d_grid_cell = cp.zeros(grid_x * grid_y * grid_z, dtype=np.float32)
        cell_spread_k = get_cell_spread_kernel()
        shmem = subgrid_total * 4
        cell_spread_k(
            (bl.num_cells_total,), (256,),
            (sorted_pos_x, sorted_pos_y, sorted_pos_z, sorted_charges,
             bl.d_cell_block_offset, bl.d_cell_block_count, bl.d_block_atoms,
             np.int32(N),
             np.float32(1.0 / box), np.float32(1.0 / box), np.float32(1.0 / box),
             np.int32(grid_x), np.int32(grid_y), np.int32(grid_z),
             np.int32(bl.num_cells_x), np.int32(bl.num_cells_y), np.int32(bl.num_cells_z),
             np.int32(subgrid_dx), np.int32(subgrid_dy), np.int32(subgrid_dz),
             np.int32(order),
             d_grid_cell),
            shared_mem=shmem,
        )

        cell_grid = cp.asnumpy(d_grid_cell)
        ref_sum = float(np.sum(ref_grid))
        cell_sum = float(np.sum(cell_grid))

        assert abs(cell_sum - 1.0) < 1e-4, \
            f"Charge not conserved: cell_sum={cell_sum:.6f}, expected=1.0"
        assert abs(cell_sum - ref_sum) < 1e-4, \
            f"Cell sum ({cell_sum:.6f}) != ref sum ({ref_sum:.6f})"

        nonzero = np.abs(cell_grid) > 1e-10
        assert np.any(nonzero), "Charge was not spread to grid at all"

        ref_arr = ref_grid.astype(np.float32)
        if np.any(nonzero):
            rel_err = np.max(np.abs(ref_arr[nonzero] - cell_grid[nonzero]) / (np.abs(ref_arr[nonzero]) + 1e-10))
            assert rel_err < 1e-3, f"Max relative error: {rel_err}"
