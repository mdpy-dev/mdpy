from __future__ import annotations

import math

import cupy as cp
import numpy as np
import pytest

from mdpy.force.pme_bspline import (
    compute_bspline_weights,
    get_bspline_kernel,
    get_exclusion_kernel,
    get_gather_kernel,
    get_self_energy_kernel,
    get_spread_kernel,
    precompute_bk_factors,
)


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

    def test_gpu_matches_cpu(self):
        N = 100
        np.random.seed(42)
        box_x, box_y, box_z = 100.0, 100.0, 100.0
        grid_x, grid_y, grid_z = 64, 64, 64
        order = 4

        pos_x = np.random.uniform(0, box_x, N).astype(np.float32)
        pos_y = np.random.uniform(0, box_y, N).astype(np.float32)
        pos_z = np.random.uniform(0, box_z, N).astype(np.float32)

        d_pos_x = cp.asarray(pos_x)
        d_pos_y = cp.asarray(pos_y)
        d_pos_z = cp.asarray(pos_z)
        d_theta = cp.zeros(N * order * 3, dtype=np.float32)
        d_dtheta = cp.zeros(N * order * 3, dtype=np.float32)
        d_grid_idx = cp.zeros(N * 3, dtype=np.int32)

        kernel = get_bspline_kernel()
        kernel(
            (1,), (256,),
            (d_pos_x, d_pos_y, d_pos_z, np.int32(N),
             np.float32(1.0 / box_x), np.float32(1.0 / box_y), np.float32(1.0 / box_z),
             np.int32(grid_x), np.int32(grid_y), np.int32(grid_z), np.int32(order),
             d_theta, d_dtheta, d_grid_idx),
        )

        h_theta = cp.asnumpy(d_theta).reshape(N, order * 3)
        h_dtheta = cp.asnumpy(d_dtheta).reshape(N, order * 3)
        h_grid_idx = cp.asnumpy(d_grid_idx).reshape(N, 3)

        for i in range(N):
            fx = pos_x[i] / box_x * grid_x
            fy = pos_y[i] / box_y * grid_y
            fz = pos_z[i] / box_z * grid_z

            for dim, (frac, gd) in enumerate([(fx, grid_x), (fy, grid_y), (fz, grid_z)]):
                u = frac - math.floor(frac)
                theta_cpu, dtheta_cpu = compute_bspline_weights(u, order=order)

                expected_gidx = int(math.floor(frac)) % gd
                assert h_grid_idx[i, dim] == expected_gidx, \
                    f"atom {i} dim {dim}: grid_idx {h_grid_idx[i, dim]} vs {expected_gidx}"

                for k in range(order):
                    gpu_theta = h_theta[i, dim * order + k]
                    gpu_dtheta = h_dtheta[i, dim * order + k]
                    assert abs(gpu_theta - theta_cpu[k]) < 1e-5, \
                        f"atom {i} dim {dim} k {k}: theta {gpu_theta} vs {theta_cpu[k]}"
                    assert abs(gpu_dtheta - dtheta_cpu[k]) < 1e-4, \
                        f"atom {i} dim {dim} k {k}: dtheta {gpu_dtheta} vs {dtheta_cpu[k]}"


class TestChargeSpreading:

    def test_charge_conservation(self):
        N = 50
        np.random.seed(123)
        order = 4
        grid_x, grid_y, grid_z = 32, 32, 32
        box_x, box_y, box_z = 50.0, 50.0, 50.0

        charges = np.random.randn(N).astype(np.float32) * 0.5
        pos_x = np.random.uniform(0, box_x, N).astype(np.float32)
        pos_y = np.random.uniform(0, box_y, N).astype(np.float32)
        pos_z = np.random.uniform(0, box_z, N).astype(np.float32)

        d_charges = cp.asarray(charges)
        d_pos_x = cp.asarray(pos_x)
        d_pos_y = cp.asarray(pos_y)
        d_pos_z = cp.asarray(pos_z)

        d_theta = cp.zeros(N * order * 3, dtype=np.float32)
        d_dtheta = cp.zeros(N * order * 3, dtype=np.float32)
        d_grid_idx = cp.zeros(N * 3, dtype=np.int32)

        bspline_kernel = get_bspline_kernel()
        bspline_kernel(
            (1,), (256,),
            (d_pos_x, d_pos_y, d_pos_z, np.int32(N),
             np.float32(1.0 / box_x), np.float32(1.0 / box_y), np.float32(1.0 / box_z),
             np.int32(grid_x), np.int32(grid_y), np.int32(grid_z), np.int32(order),
             d_theta, d_dtheta, d_grid_idx),
        )

        d_charge_grid = cp.zeros(grid_x * grid_y * grid_z, dtype=np.float32)
        spread_kernel = get_spread_kernel()
        spread_kernel(
            (1,), (256,),
            (d_charges, d_grid_idx, d_theta,
             np.int32(N), np.int32(grid_x), np.int32(grid_y), np.int32(grid_z), np.int32(order),
             d_charge_grid),
        )

        grid_sum = float(cp.sum(d_charge_grid))
        charge_sum = float(np.sum(charges))
        assert abs(grid_sum - charge_sum) < abs(charge_sum) * 1e-5 + 1e-6, \
            f"grid_sum={grid_sum}, charge_sum={charge_sum}"

    def test_single_atom_known_position(self):
        order = 4
        grid_x, grid_y, grid_z = 32, 32, 32
        box_x, box_y, box_z = 32.0, 32.0, 32.0

        charges = np.array([1.0], dtype=np.float32)
        pos_x = np.array([5.0], dtype=np.float32)
        pos_y = np.array([5.0], dtype=np.float32)
        pos_z = np.array([5.0], dtype=np.float32)

        d_charges = cp.asarray(charges)
        d_pos_x = cp.asarray(pos_x)
        d_pos_y = cp.asarray(pos_y)
        d_pos_z = cp.asarray(pos_z)

        d_theta = cp.zeros(1 * order * 3, dtype=np.float32)
        d_dtheta = cp.zeros(1 * order * 3, dtype=np.float32)
        d_grid_idx = cp.zeros(1 * 3, dtype=np.int32)

        bspline_kernel = get_bspline_kernel()
        bspline_kernel(
            (1,), (256,),
            (d_pos_x, d_pos_y, d_pos_z, np.int32(1),
             np.float32(1.0 / box_x), np.float32(1.0 / box_y), np.float32(1.0 / box_z),
             np.int32(grid_x), np.int32(grid_y), np.int32(grid_z), np.int32(order),
             d_theta, d_dtheta, d_grid_idx),
        )

        d_charge_grid = cp.zeros(grid_x * grid_y * grid_z, dtype=np.float32)
        spread_kernel = get_spread_kernel()
        spread_kernel(
            (1,), (256,),
            (d_charges, d_grid_idx, d_theta,
             np.int32(1), np.int32(grid_x), np.int32(grid_y), np.int32(grid_z), np.int32(order),
             d_charge_grid),
        )

        h_grid = cp.asnumpy(d_charge_grid).reshape(grid_x, grid_y, grid_z)
        total = np.sum(h_grid)
        assert abs(total - 1.0) < 1e-5, f"Total charge={total}"
        assert np.sum(h_grid > 0) > 1, "Charge should spread to multiple grid points"


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

        d_theta = cp.zeros(N * order * 3, dtype=np.float32)
        d_dtheta = cp.zeros(N * order * 3, dtype=np.float32)
        d_grid_idx = cp.zeros(N * 3, dtype=np.int32)

        bspline_k = get_bspline_kernel()
        bspline_k(
            (1,), (256,),
            (d_pos_x, d_pos_y, d_pos_z, np.int32(N),
             np.float32(1.0 / box_x), np.float32(1.0 / box_y), np.float32(1.0 / box_z),
             np.int32(grid_x), np.int32(grid_y), np.int32(grid_z), np.int32(order),
             d_theta, d_dtheta, d_grid_idx),
        )

        d_charge_grid = cp.zeros(grid_x * grid_y * grid_z, dtype=np.float32)
        spread_k = get_spread_kernel()
        spread_k(
            (1,), (256,),
            (d_charges, d_grid_idx, d_theta,
             np.int32(N), np.int32(grid_x), np.int32(grid_y), np.int32(grid_z), np.int32(order),
             d_charge_grid),
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
        gather_k(
            (1,), (256,),
            (d_charges, d_grid_idx, d_theta, d_dtheta,
             np.int32(N), np.int32(grid_x), np.int32(grid_y), np.int32(grid_z), np.int32(order),
             np.float32(1.0 / box_x), np.float32(1.0 / box_y), np.float32(1.0 / box_z),
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


class TestExclusionCorrection:

    def test_produces_nonzero_energy(self):
        N = 4
        box_x, box_y, box_z = 20.0, 20.0, 20.0
        alpha = 0.35

        charges = np.array([0.5, -0.3, 0.8, -0.6], dtype=np.float32)
        pos_x = np.array([5.0, 6.0, 10.0, 15.0], dtype=np.float32)
        pos_y = np.array([5.0, 5.0, 10.0, 10.0], dtype=np.float32)
        pos_z = np.array([5.0, 5.0, 10.0, 10.0], dtype=np.float32)

        pair_i = np.array([0, 2], dtype=np.int32)
        pair_j = np.array([1, 3], dtype=np.int32)
        pair_scale = np.array([0.0, 0.8333333], dtype=np.float32)

        d_pos_x = cp.asarray(pos_x)
        d_pos_y = cp.asarray(pos_y)
        d_pos_z = cp.asarray(pos_z)
        d_charges = cp.asarray(charges)
        d_pair_i = cp.asarray(pair_i)
        d_pair_j = cp.asarray(pair_j)
        d_pair_scale = cp.asarray(pair_scale)

        d_fx = cp.zeros(N, dtype=np.float32)
        d_fy = cp.zeros(N, dtype=np.float32)
        d_fz = cp.zeros(N, dtype=np.float32)
        d_energy = cp.zeros(1, dtype=np.float32)

        excl_k = get_exclusion_kernel()
        excl_k((1,), (256,),
            (d_pos_x, d_pos_y, d_pos_z, d_charges,
             d_pair_i, d_pair_j, d_pair_scale,
             np.int32(2), np.float32(alpha),
             np.float32(box_x), np.float32(box_y), np.float32(box_z),
             d_fx, d_fy, d_fz, d_energy))

        h_fx = cp.asnumpy(d_fx)
        h_fy = cp.asnumpy(d_fy)
        h_fz = cp.asnumpy(d_fz)
        energy = float(d_energy[0])

        print(f"Exclusion energy: {energy:.6f}")
        print(f"Forces x: {h_fx}")
        print(f"Forces y: {h_fy}")
        print(f"Forces z: {h_fz}")

        assert abs(energy) > 1e-6, "Exclusion energy should be nonzero"

        assert np.max(np.abs(h_fx)) > 1e-6 or np.max(np.abs(h_fy)) > 1e-6 or np.max(np.abs(h_fz)) > 1e-6

        net_fx = float(np.sum(h_fx))
        net_fy = float(np.sum(h_fy))
        net_fz = float(np.sum(h_fz))
        max_force = max(float(np.max(np.abs(h_fx))), float(np.max(np.abs(h_fy))), float(np.max(np.abs(h_fz))))
        if max_force > 1e-10:
            assert abs(net_fx) < max_force * 0.01
            assert abs(net_fy) < max_force * 0.01
            assert abs(net_fz) < max_force * 0.01

    def test_newtons_third_law(self):
        N = 2
        box_x, box_y, box_z = 20.0, 20.0, 20.0
        alpha = 0.35

        charges = np.array([1.0, -1.0], dtype=np.float32)
        pos_x = np.array([5.0, 7.0], dtype=np.float32)
        pos_y = np.array([5.0, 5.0], dtype=np.float32)
        pos_z = np.array([5.0, 5.0], dtype=np.float32)

        pair_i = np.array([0], dtype=np.int32)
        pair_j = np.array([1], dtype=np.int32)
        pair_scale = np.array([0.0], dtype=np.float32)

        d_pos_x = cp.asarray(pos_x)
        d_pos_y = cp.asarray(pos_y)
        d_pos_z = cp.asarray(pos_z)
        d_charges = cp.asarray(charges)
        d_pair_i = cp.asarray(pair_i)
        d_pair_j = cp.asarray(pair_j)
        d_pair_scale = cp.asarray(pair_scale)

        d_fx = cp.zeros(N, dtype=np.float32)
        d_fy = cp.zeros(N, dtype=np.float32)
        d_fz = cp.zeros(N, dtype=np.float32)
        d_energy = cp.zeros(1, dtype=np.float32)

        excl_k = get_exclusion_kernel()
        excl_k((1,), (256,),
            (d_pos_x, d_pos_y, d_pos_z, d_charges,
             d_pair_i, d_pair_j, d_pair_scale,
             np.int32(1), np.float32(alpha),
             np.float32(box_x), np.float32(box_y), np.float32(box_z),
             d_fx, d_fy, d_fz, d_energy))

        h_fx = cp.asnumpy(d_fx)
        h_fy = cp.asnumpy(d_fy)
        h_fz = cp.asnumpy(d_fz)

        assert abs(h_fx[0] + h_fx[1]) < 1e-6, f"F_x not balanced: {h_fx}"
        assert abs(h_fy[0] + h_fy[1]) < 1e-6, f"F_y not balanced: {h_fy}"
        assert abs(h_fz[0] + h_fz[1]) < 1e-6, f"F_z not balanced: {h_fz}"
