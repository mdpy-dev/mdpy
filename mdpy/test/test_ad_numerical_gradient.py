"""Numerical gradient verification for AD-derived expression forces.

For each expression, creates a minimal atom system, runs the GPU kernel to get
analytical forces, then computes numerical gradients via central finite differences
and compares the two.
"""

import numpy as np
import cupy as cp
import pytest

from mdpy import env
from mdpy.force.bonded_force import BondedForce
from mdpy.force.bonded_transpiler import bonded_expression
from mdpy.force.markers import param, scalar as scalar_marker


class MockGPUContext:
    def __init__(self, positions, pbc_matrix):
        pos = positions.astype(np.float32)
        self.d_positions_x = cp.asarray(pos[:, 0])
        self.d_positions_y = cp.asarray(pos[:, 1])
        self.d_positions_z = cp.asarray(pos[:, 2])
        N = positions.shape[0]
        self.d_forces_x = cp.zeros(N, dtype=np.float32)
        self.d_forces_y = cp.zeros(N, dtype=np.float32)
        self.d_forces_z = cp.zeros(N, dtype=np.float32)
        self.d_energy = cp.zeros(1, dtype=np.float32)
        pbc_inv = np.linalg.inv(pbc_matrix)
        self.d_pbc_inv = cp.asarray(
            np.ascontiguousarray(pbc_inv, dtype=np.float32).ravel()
        )
        self.d_pbc_matrix = cp.asarray(
            np.ascontiguousarray(pbc_matrix, dtype=np.float32).ravel()
        )

    def get_forces(self):
        return np.stack([
            self.d_forces_x.get(),
            self.d_forces_y.get(),
            self.d_forces_z.get(),
        ], axis=1)


def _large_pbc():
    return np.eye(3, dtype=env.NUMPY_FLOAT) * 100.0


@bonded_expression(body=2)
def _harmonic_bond(p1, p2, k=0.0, r0=0.0):
    r = distance(p1, p2)
    dr = r - r0
    return k * dr * dr


@bonded_expression(body=3)
def _charmm_angle(p1, p2, p3, k=0.0, theta0=0.0, k_ub=0.0, r_ub=0.0):
    theta = angle(p1, p2, p3)
    dt = theta - theta0
    e_angle = k * dt * dt
    r13 = distance(p1, p3)
    dr13 = r13 - r_ub
    e_ub = k_ub * dr13 * dr13
    return e_angle + e_ub


@bonded_expression(body=4)
def _periodic_dihedral(p1, p2, p3, p4, k=0.0, n=0.0, delta=0.0):
    phi = dihedral(p1, p2, p3, p4)
    return k * (1.0 + cos(n * phi - delta))


@bonded_expression(body=4)
def _harmonic_improper(p1, p2, p3, p4, k=0.0, psi0=0.0):
    psi = dihedral(p1, p2, p3, p4)
    dp = psi - psi0
    return k * dp * dp


@bonded_expression(body=2)
def _nb14_lj_coulomb(pos1, pos2, charge1, charge2, sigma=param, epsilon=param, charge_scale=param):
    r = distance(pos1, pos2)
    e_coul = charge_scale * 0.13893556595455 * charge1 * charge2 / r
    sr = sigma / r
    sr6 = sr * sr * sr * sr * sr * sr
    e_lj = 4.0 * epsilon * (sr6 * sr6 - sr6)
    return e_coul + e_lj


def _compute_bonded_energy(expression, positions, params_list, per_particle_data=None, pbc=None):
    if pbc is None:
        pbc = _large_pbc()
    force = BondedForce(expression)
    if per_particle_data:
        for name, arr in per_particle_data.items():
            force.set_parameter(name, arr)
    for indices, params in params_list:
        force.add(indices, **params)
    ctx = MockGPUContext(positions, pbc)
    force.compute(ctx)
    return float(ctx.d_energy[0])


def _numerical_gradient_bonded(expression, positions, params_list, per_particle_data=None, h=1e-4):
    N = positions.shape[0]
    num_grad = np.zeros_like(positions)
    base_energy = _compute_bonded_energy(expression, positions, params_list, per_particle_data)

    for i in range(N):
        for k in range(3):
            pos_plus = positions.copy()
            pos_plus[i, k] += h
            e_plus = _compute_bonded_energy(expression, pos_plus, params_list, per_particle_data)

            pos_minus = positions.copy()
            pos_minus[i, k] -= h
            e_minus = _compute_bonded_energy(expression, pos_minus, params_list, per_particle_data)

            num_grad[i, k] = -(e_plus - e_minus) / (2.0 * h)

    return num_grad, base_energy


def _compare_forces(expression, positions, params_list, per_particle_data=None, tol=1e-2):
    pbc = _large_pbc()
    force = BondedForce(expression)
    if per_particle_data:
        for name, arr in per_particle_data.items():
            force.set_parameter(name, arr)
    for indices, params in params_list:
        force.add(indices, **params)
    ctx = MockGPUContext(positions, pbc)
    force.compute(ctx)
    anal_forces = ctx.get_forces()

    num_grad, energy = _numerical_gradient_bonded(expression, positions, params_list, per_particle_data)

    mask = np.abs(num_grad) > 1e-6
    if mask.any():
        rel_err = np.abs(anal_forces[mask] - num_grad[mask]) / (np.abs(num_grad[mask]) + 1e-10)
        max_rel = rel_err.max()
    else:
        max_rel = 0.0
    return anal_forces, num_grad, max_rel, energy


class TestBondedExpressions:

    def test_harmonic_bond_gradient(self):
        positions = np.array([
            [10.0, 10.0, 10.0],
            [12.0, 10.5, 10.0],
        ], dtype=np.float32)

        anal, num, max_err, energy = _compare_forces(
            _harmonic_bond, positions,
            [([0, 1], {'k': 200.0, 'r0': 1.5})],
        )
        assert max_err < 0.01, (
            f"harmonic_bond: max rel err = {max_err:.6e}\n"
            f"  analytical: {anal}\n  numerical: {num}"
        )

    def test_charmm_angle_gradient(self):
        positions = np.array([
            [10.0, 12.0, 10.0],
            [10.0, 10.0, 10.0],
            [12.0, 10.0, 10.5],
        ], dtype=np.float32)

        anal, num, max_err, energy = _compare_forces(
            _charmm_angle, positions,
            [([0, 1, 2], {'k': 40.0, 'theta0': 1.8, 'k_ub': 0.0, 'r_ub': 0.0})],
        )
        assert max_err < 0.01, (
            f"charmm_angle: max rel err = {max_err:.6e}\n"
            f"  analytical: {anal}\n  numerical: {num}"
        )

    def test_charmm_angle_with_ub_gradient(self):
        positions = np.array([
            [10.0, 12.0, 10.0],
            [10.0, 10.0, 10.0],
            [12.0, 10.0, 10.5],
        ], dtype=np.float32)

        anal, num, max_err, energy = _compare_forces(
            _charmm_angle, positions,
            [([0, 1, 2], {'k': 40.0, 'theta0': 1.8, 'k_ub': 5.0, 'r_ub': 2.5})],
        )
        assert max_err < 0.01, (
            f"charmm_angle (w/UB): max rel err = {max_err:.6e}\n"
            f"  analytical: {anal}\n  numerical: {num}"
        )

    def test_periodic_dihedral_gradient(self):
        positions = np.array([
            [10.0, 12.0, 10.0],
            [10.0, 10.0, 10.0],
            [12.0, 10.0, 10.0],
            [12.0, 10.0, 12.0],
        ], dtype=np.float32)

        anal, num, max_err, energy = _compare_forces(
            _periodic_dihedral, positions,
            [([0, 1, 2, 3], {'k': 1.0, 'n': 3.0, 'delta': 3.14159265})],
        )
        assert max_err < 0.01, (
            f"periodic_dihedral: max rel err = {max_err:.6e}\n"
            f"  analytical: {anal}\n  numerical: {num}"
        )

    def test_harmonic_improper_gradient(self):
        positions = np.array([
            [10.0, 12.0, 10.0],
            [10.0, 10.0, 10.0],
            [12.0, 10.0, 10.0],
            [12.0, 10.0, 12.0],
        ], dtype=np.float32)

        anal, num, max_err, energy = _compare_forces(
            _harmonic_improper, positions,
            [([0, 1, 2, 3], {'k': 5.0, 'psi0': 0.0})],
        )
        assert max_err < 0.01, (
            f"harmonic_improper: max rel err = {max_err:.6e}\n"
            f"  analytical: {anal}\n  numerical: {num}"
        )

    def test_nb14_lj_coulomb_gradient(self):
        positions = np.array([
            [10.0, 10.0, 10.0],
            [11.5, 10.5, 10.2],
        ], dtype=np.float32)

        per_particle = {
            'charge': np.array([0.3, -0.2], dtype=np.float32),
        }

        anal, num, max_err, energy = _compare_forces(
            _nb14_lj_coulomb, positions,
            [([0, 1], {'sigma': 3.5, 'epsilon': 0.05, 'charge_scale': 0.8333})],
            per_particle_data=per_particle,
        )
        assert max_err < 0.01, (
            f"nb14_lj_coulomb: max rel err = {max_err:.6e}\n"
            f"  analytical: {anal}\n  numerical: {num}"
        )


class TestNonbondedExpressions:
    pass
