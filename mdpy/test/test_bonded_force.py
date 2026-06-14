import numpy as np
import pytest

from mdpy import env
from mdpy.force.bonded_force import BondedForce
from mdpy.force.bonded_transpiler import bonded_expression


def _make_large_pbc():
    return np.eye(3, dtype=env.NUMPY_FLOAT) * 100.0


class MockGPUContext:
    def __init__(self, positions, pbc_matrix):
        import cupy as cp
        pos = positions.astype(np.float32)
        self.d_positions_x = cp.asarray(pos[:, 0])
        self.d_positions_y = cp.asarray(pos[:, 1])
        self.d_positions_z = cp.asarray(pos[:, 2])
        N = positions.shape[0]
        self.d_forces_x = cp.zeros(N, dtype=np.float32)
        self.d_forces_y = cp.zeros(N, dtype=np.float32)
        self.d_forces_z = cp.zeros(N, dtype=np.float32)
        self.d_energy = cp.zeros(1, dtype=np.float32)
        self.d_charges = None
        pbc_inv = np.linalg.inv(pbc_matrix)
        self.d_pbc_inv = cp.asarray(
            np.ascontiguousarray(pbc_inv, dtype=np.float32).ravel()
        )
        self.d_pbc_matrix = cp.asarray(
            np.ascontiguousarray(pbc_matrix, dtype=np.float32).ravel()
        )

    @property
    def d_forces(self):
        import cupy as cp
        return cp.stack([self.d_forces_x, self.d_forces_y, self.d_forces_z], axis=1).ravel()


@bonded_expression(body=2)
def harmonic_bond(p1, p2, k=0.0, r0=0.0):
    r = distance(p1, p2)
    dr = r - r0
    return k * dr * dr


@bonded_expression(body=3)
def harmonic_angle(p1, p2, p3, k=0.0, theta0=0.0):
    theta = angle(p1, p2, p3)
    dt = theta - theta0
    return k * dt * dt


@bonded_expression(body=4)
def periodic_dihedral(p1, p2, p3, p4, k=0.0, n=0.0, delta=0.0):
    phi = dihedral(p1, p2, p3, p4)
    return k * (1.0 + cos(n * phi - delta))


@bonded_expression(body=4)
def harmonic_improper(p1, p2, p3, p4, k=0.0, psi0=0.0):
    psi = dihedral(p1, p2, p3, p4)
    dp = psi - psi0
    return k * dp * dp


def test_harmonic_bond_energy():
    force = BondedForce(harmonic_bond)
    k = 100.0
    r0 = 1.5
    force.add([0, 1], k=k, r0=r0)

    positions = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=env.NUMPY_FLOAT)
    pbc_matrix = _make_large_pbc()

    context = MockGPUContext(positions, pbc_matrix)
    force.compute(context)

    energy = float(context.d_energy[0])
    expected_energy = k * (2.0 - r0) ** 2
    assert abs(energy - expected_energy) < 1e-2, f"Energy {energy} != {expected_energy}"


def test_harmonic_bond_forces():
    force = BondedForce(harmonic_bond)
    k = 100.0
    r0 = 1.5
    force.add([0, 1], k=k, r0=r0)

    positions = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=env.NUMPY_FLOAT)
    pbc_matrix = _make_large_pbc()

    context = MockGPUContext(positions, pbc_matrix)
    force.compute(context)

    gpu_forces = context.d_forces.get().reshape(-1, 3)
    expected_force_magnitude = 2.0 * k * (2.0 - r0)
    assert abs(gpu_forces[0, 0] - expected_force_magnitude) < 0.1, \
        f"Force on atom 0: {gpu_forces[0, 0]} != {expected_force_magnitude}"
    assert abs(gpu_forces[1, 0] + expected_force_magnitude) < 0.1, \
        f"Force on atom 1: {gpu_forces[1, 0]} != {-expected_force_magnitude}"


def test_harmonic_bond_newtons_third_law():
    force = BondedForce(harmonic_bond)
    k = 200.0
    r0 = 1.0
    force.add([0, 1], k=k, r0=r0)

    positions = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ], dtype=env.NUMPY_FLOAT)
    pbc_matrix = _make_large_pbc()

    context = MockGPUContext(positions, pbc_matrix)
    force.compute(context)

    gpu_forces = context.d_forces.get().reshape(-1, 3)
    total_force = np.sum(gpu_forces, axis=0)
    assert np.linalg.norm(total_force) < 1e-3, \
        f"Forces not balanced: sum={total_force}"


def test_multiple_bonds():
    force = BondedForce(harmonic_bond)
    k = 100.0
    r0 = 1.5
    force.add([0, 1], k=k, r0=r0)
    force.add([1, 2], k=k, r0=r0)

    positions = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ], dtype=env.NUMPY_FLOAT)
    pbc_matrix = _make_large_pbc()

    context = MockGPUContext(positions, pbc_matrix)
    force.compute(context)

    energy = float(context.d_energy[0])
    expected_energy = 2.0 * k * (2.0 - r0) ** 2
    assert abs(energy - expected_energy) < 1e-1, f"Energy {energy} != {expected_energy}"

    gpu_forces = context.d_forces.get().reshape(-1, 3)
    total_force = np.sum(gpu_forces, axis=0)
    assert np.linalg.norm(total_force) < 0.5, \
        f"Forces not balanced: sum={total_force}"


def test_harmonic_angle_energy():
    force = BondedForce(harmonic_angle)
    k = 50.0
    theta0 = np.pi / 3
    force.add([0, 1, 2], k=k, theta0=theta0)

    positions = np.array([
        [0.0, 1.5, 0.0],
        [0.0, 0.0, 0.0],
        [1.5 * np.cos(np.pi / 6), 1.5 * np.sin(np.pi / 6), 0.0],
    ], dtype=env.NUMPY_FLOAT)
    pbc_matrix = _make_large_pbc()

    context = MockGPUContext(positions, pbc_matrix)
    force.compute(context)

    energy = float(context.d_energy[0])
    assert np.isfinite(energy)
    assert energy != 0.0


def test_harmonic_angle_forces_balanced():
    force = BondedForce(harmonic_angle)
    k = 50.0
    theta0 = np.pi / 3
    force.add([0, 1, 2], k=k, theta0=theta0)

    positions = np.array([
        [0.0, 1.5, 0.0],
        [0.0, 0.0, 0.0],
        [1.5 * np.cos(np.pi / 6), 1.5 * np.sin(np.pi / 6), 0.0],
    ], dtype=env.NUMPY_FLOAT)
    pbc_matrix = _make_large_pbc()

    context = MockGPUContext(positions, pbc_matrix)
    force.compute(context)

    gpu_forces = context.d_forces.get().reshape(-1, 3)
    total_force = np.sum(gpu_forces, axis=0)
    assert np.linalg.norm(total_force) < 0.5, \
        f"Angle forces not balanced: sum={total_force}"


def test_periodic_dihedral_energy():
    force = BondedForce(periodic_dihedral)
    k = 20.0
    n = 1.0
    delta = np.pi
    force.add([0, 1, 2, 3], k=k, n=n, delta=delta)

    positions = np.array([
        [0.0, 1.5, 0.0],
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [1.5, 0.0, 1.5],
    ], dtype=env.NUMPY_FLOAT)
    pbc_matrix = _make_large_pbc()

    context = MockGPUContext(positions, pbc_matrix)
    force.compute(context)

    energy = float(context.d_energy[0])
    assert energy != 0.0


def test_periodic_dihedral_forces_balanced():
    force = BondedForce(periodic_dihedral)
    k = 20.0
    n = 1.0
    delta = np.pi
    force.add([0, 1, 2, 3], k=k, n=n, delta=delta)

    positions = np.array([
        [0.0, 1.5, 0.0],
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [1.5, 0.0, 1.5],
    ], dtype=env.NUMPY_FLOAT)
    pbc_matrix = _make_large_pbc()

    context = MockGPUContext(positions, pbc_matrix)
    force.compute(context)

    gpu_forces = context.d_forces.get().reshape(-1, 3)
    total_force = np.sum(gpu_forces, axis=0)
    assert np.linalg.norm(total_force) < 0.5, \
        f"Dihedral forces not balanced: sum={total_force}"


def test_improper_forces_balanced():
    force = BondedForce(harmonic_improper)
    k = 30.0
    psi0 = 0.0
    force.add([0, 1, 2, 3], k=k, psi0=psi0)

    positions = np.array([
        [0.0, 1.5, 0.0],
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [1.5, 0.0, 1.5],
    ], dtype=env.NUMPY_FLOAT)
    pbc_matrix = _make_large_pbc()

    context = MockGPUContext(positions, pbc_matrix)
    force.compute(context)

    gpu_forces = context.d_forces.get().reshape(-1, 3)
    total_force = np.sum(gpu_forces, axis=0)
    assert np.linalg.norm(total_force) < 0.5, \
        f"Improper forces not balanced: sum={total_force}"


def test_bond_energy_at_equilibrium():
    force = BondedForce(harmonic_bond)
    k = 100.0
    r0 = 1.5
    force.add([0, 1], k=k, r0=r0)

    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
    ], dtype=env.NUMPY_FLOAT)
    pbc_matrix = _make_large_pbc()

    context = MockGPUContext(positions, pbc_matrix)
    force.compute(context)

    energy = float(context.d_energy[0])
    assert abs(energy) < 1e-3, f"Energy at equilibrium should be ~0, got {energy}"


def test_empty_terms():
    force = BondedForce(harmonic_bond)

    positions = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=env.NUMPY_FLOAT)
    pbc_matrix = _make_large_pbc()

    context = MockGPUContext(positions, pbc_matrix)
    force.compute(context)

    energy = float(context.d_energy[0])
    assert abs(energy) < 1e-6, f"Empty force energy should be 0, got {energy}"


def test_incremental_add_and_sync():
    force = BondedForce(harmonic_bond)
    k = 100.0
    r0 = 1.5

    force.add([0, 1], k=k, r0=r0)
    force.add([1, 2], k=k, r0=r0)

    positions = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.5, 0.0, 0.0],
    ], dtype=env.NUMPY_FLOAT)
    pbc_matrix = _make_large_pbc()

    context = MockGPUContext(positions, pbc_matrix)
    force.compute(context)

    energy_before = float(context.d_energy[0])
    assert energy_before > 0.0

    force.add([0, 2], k=k, r0=r0)
    context2 = MockGPUContext(positions, pbc_matrix)
    force.sync()
    force.compute(context2)

    energy_after = float(context2.d_energy[0])
    assert energy_after > energy_before


@bonded_expression(body=2)
def simple_coulomb(pos1, pos2, charge1, charge2):
    r = distance(pos1, pos2)
    return 0.13893556595455 * charge1 * charge2 / r


def test_per_particle_set_parameter():
    force = BondedForce(simple_coulomb)
    charges = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
    force.set_parameter("charge", charges)

    assert "charge" in force._per_particle_gpu
    assert force._per_particle_gpu["charge"].shape[0] == 4

    import cupy as cp
    gpu_charges = cp.asnumpy(force._per_particle_gpu["charge"])
    np.testing.assert_allclose(gpu_charges, charges, atol=1e-6)


def test_per_particle_energy():
    force = BondedForce(simple_coulomb)

    q1, q2 = 1.0, -0.5
    charges = np.array([q1, q2], dtype=np.float32)
    force.set_parameter("charge", charges)
    force.add([0, 1])

    positions = np.array([
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ], dtype=env.NUMPY_FLOAT)
    pbc_matrix = _make_large_pbc()

    context = MockGPUContext(positions, pbc_matrix)
    force.compute(context)

    energy = float(context.d_energy[0])
    r = 3.0
    expected = 0.13893556595455 * q1 * q2 / r
    assert abs(energy - expected) < 1e-3, f"Energy {energy} != {expected}"


def test_per_particle_forces_balanced():
    force = BondedForce(simple_coulomb)

    charges = np.array([1.0, -1.0], dtype=np.float32)
    force.set_parameter("charge", charges)
    force.add([0, 1])

    positions = np.array([
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ], dtype=env.NUMPY_FLOAT)
    pbc_matrix = _make_large_pbc()

    context = MockGPUContext(positions, pbc_matrix)
    force.compute(context)

    gpu_forces = context.d_forces.get().reshape(-1, 3)
    total_force = np.sum(gpu_forces, axis=0)
    assert np.linalg.norm(total_force) < 0.5, \
        f"Forces not balanced: sum={total_force}"


def test_per_particle_multiple_pairs():
    force = BondedForce(simple_coulomb)

    charges = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
    force.set_parameter("charge", charges)
    force.add([0, 1])
    force.add([2, 3])

    positions = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [7.0, 0.0, 0.0],
    ], dtype=env.NUMPY_FLOAT)
    pbc_matrix = _make_large_pbc()

    context = MockGPUContext(positions, pbc_matrix)
    force.compute(context)

    energy = float(context.d_energy[0])
    expected_e01 = 0.13893556595455 * 1.0 * (-1.0) / 2.0
    expected_e23 = 0.13893556595455 * 0.5 * (-0.5) / 2.0
    expected = expected_e01 + expected_e23
    assert abs(energy - expected) < 1e-2, f"Energy {energy} != {expected}"


def test_per_particle_no_per_particle_props():
    force = BondedForce(harmonic_bond)
    k = 100.0
    r0 = 1.5
    force.add([0, 1], k=k, r0=r0)

    positions = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=env.NUMPY_FLOAT)
    pbc_matrix = _make_large_pbc()

    context = MockGPUContext(positions, pbc_matrix)
    force.compute(context)

    energy = float(context.d_energy[0])
    expected_energy = k * (2.0 - r0) ** 2
    assert abs(energy - expected_energy) < 1e-2


COULOMB_CONSTANT = 0.13893556595455


def _analytical_nb14_energy(r, q1, q2, sigma, epsilon):
    e_coul = COULOMB_CONSTANT * q1 * q2 / r
    sr = sigma / r
    sr6 = sr ** 6
    e_lj = 4.0 * epsilon * (sr6 * sr6 - sr6)
    return e_coul + e_lj


from mdpy.force.expressions.nb14 import nb14_lj_coulomb


def test_nb14_energy_analytical():
    force = BondedForce(nb14_lj_coulomb)

    q1, q2 = 1.0, -0.5
    sigma = 1.0
    epsilon = 0.1
    r = 3.0

    charges = np.array([q1, q2], dtype=np.float32)
    force.set_parameter("charge", charges)
    force.add([0, 1], sigma=sigma, epsilon=epsilon)

    positions = np.array([
        [0.0, 0.0, 0.0],
        [r, 0.0, 0.0],
    ], dtype=env.NUMPY_FLOAT)
    pbc_matrix = _make_large_pbc()

    context = MockGPUContext(positions, pbc_matrix)
    force.compute(context)

    energy = float(context.d_energy[0])
    expected = _analytical_nb14_energy(r, q1, q2, sigma, epsilon)
    assert abs(energy - expected) < 1e-3, f"Energy {energy} != {expected}"


def test_nb14_forces_balanced():
    force = BondedForce(nb14_lj_coulomb)

    charges = np.array([1.0, -0.5], dtype=np.float32)
    force.set_parameter("charge", charges)
    force.add([0, 1], sigma=1.0, epsilon=0.1)

    positions = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ], dtype=env.NUMPY_FLOAT)
    pbc_matrix = _make_large_pbc()

    context = MockGPUContext(positions, pbc_matrix)
    force.compute(context)

    gpu_forces = context.d_forces.get().reshape(-1, 3)
    total_force = np.sum(gpu_forces, axis=0)
    assert np.linalg.norm(total_force) < 0.5, \
        f"nb14 forces not balanced: sum={total_force}"


def test_nb14_multiple_pairs():
    force = BondedForce(nb14_lj_coulomb)

    charges = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
    force.set_parameter("charge", charges)

    sigma1, eps1 = 1.0, 0.1
    sigma2, eps2 = 2.0, 0.2
    force.add([0, 1], sigma=sigma1, epsilon=eps1)
    force.add([2, 3], sigma=sigma2, epsilon=eps2)

    positions = np.array([
        [0.0, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [7.5, 0.0, 0.0],
    ], dtype=env.NUMPY_FLOAT)
    pbc_matrix = _make_large_pbc()

    context = MockGPUContext(positions, pbc_matrix)
    force.compute(context)

    energy = float(context.d_energy[0])
    e01 = _analytical_nb14_energy(2.5, 1.0, -1.0, sigma1, eps1)
    e23 = _analytical_nb14_energy(2.5, 0.5, -0.5, sigma2, eps2)
    expected = e01 + e23
    assert abs(energy - expected) < 1e-2, f"Energy {energy} != {expected}"


def test_nb14_force_direction():
    force = BondedForce(nb14_lj_coulomb)

    q1, q2 = 1.0, -1.0
    sigma = 3.5
    epsilon = 0.1

    charges = np.array([q1, q2], dtype=np.float32)
    force.set_parameter("charge", charges)
    force.add([0, 1], sigma=sigma, epsilon=epsilon)

    positions = np.array([
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
    ], dtype=env.NUMPY_FLOAT)
    pbc_matrix = _make_large_pbc()

    context = MockGPUContext(positions, pbc_matrix)
    force.compute(context)

    gpu_forces = context.d_forces.get().reshape(-1, 3)
    f0x = gpu_forces[0, 0]
    f1x = gpu_forces[1, 0]
    assert abs(f0x + f1x) < 0.5, \
        f"Forces not equal and opposite: f0x={f0x}, f1x={f1x}"
