import numpy as np
import pytest

from mdpy import env
from mdpy.core.topology import Builder
from mdpy.forcefield.parameters import ParameterTable
from mdpy.force.bonded_force import BondedForce


def _make_large_pbc():
    return np.eye(3, dtype=env.NUMPY_FLOAT) * 100.0


def _make_parameter_table(term_params):
    pt = ParameterTable()
    for name, values in term_params.items():
        pt.add_per_term(name, values)
    return pt


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
        bx = float(pbc_matrix[0, 0])
        by = float(pbc_matrix[1, 1])
        bz = float(pbc_matrix[2, 2])
        self.d_box_dims = cp.array([
            bx, by, bz, 1.0/bx, 1.0/by, 1.0/bz
        ], dtype=np.float32)

    @property
    def d_forces(self):
        import cupy as cp
        return cp.stack([self.d_forces_x, self.d_forces_y, self.d_forces_z], axis=1).ravel()


def _build_two_particle():
    builder = Builder()
    builder.set_particles(
        masses=np.array([12.0, 12.0], dtype=env.NUMPY_FLOAT),
        charges=np.zeros(2, dtype=env.NUMPY_FLOAT),
        particle_types=np.zeros(2, dtype=env.NUMPY_INT),
    )
    builder.add_bond(0, 1, 100.0, 1.5)
    return builder.build()


def _build_three_particle():
    builder = Builder()
    builder.set_particles(
        masses=np.array([12.0, 12.0, 12.0], dtype=env.NUMPY_FLOAT),
        charges=np.zeros(3, dtype=env.NUMPY_FLOAT),
        particle_types=np.zeros(3, dtype=env.NUMPY_INT),
    )
    builder.add_bond(0, 1, 100.0, 1.5)
    builder.add_bond(1, 2, 100.0, 1.5)
    builder.add_angle(0, 1, 2, 50.0, np.pi / 3, 10.0, 2.5)
    return builder.build()


def _build_four_particle():
    builder = Builder()
    builder.set_particles(
        masses=np.array([12.0, 12.0, 12.0, 12.0], dtype=env.NUMPY_FLOAT),
        charges=np.zeros(4, dtype=env.NUMPY_FLOAT),
        particle_types=np.zeros(4, dtype=env.NUMPY_INT),
    )
    builder.add_bond(0, 1, 100.0, 1.5)
    builder.add_bond(1, 2, 100.0, 1.5)
    builder.add_bond(2, 3, 100.0, 1.5)
    builder.add_angle(0, 1, 2, 50.0, np.pi / 3, 0.0, 0.0)
    builder.add_angle(1, 2, 3, 50.0, np.pi / 3, 0.0, 0.0)
    builder.add_dihedral(0, 1, 2, 3, 20.0, 1.0, np.pi)
    builder.add_improper(0, 1, 2, 3, 30.0, 0.0)
    return builder.build()


def _build_empty():
    builder = Builder()
    builder.set_particles(
        masses=np.array([12.0, 12.0], dtype=env.NUMPY_FLOAT),
        charges=np.zeros(2, dtype=env.NUMPY_FLOAT),
        particle_types=np.zeros(2, dtype=env.NUMPY_INT),
    )
    return builder.build()


def test_bond_force_gpu():
    topology, term_params = _build_two_particle()
    parameter_table = _make_parameter_table(term_params)
    positions = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=env.NUMPY_FLOAT)
    pbc_matrix = _make_large_pbc()

    force = BondedForce.charmm(topology, parameter_table)
    context = MockGPUContext(positions, pbc_matrix)
    force.compute(context)
    energy = float(context.d_energy[0])

    expected_energy = 100.0 * (2.0 - 1.5) ** 2
    assert abs(energy - expected_energy) < 1e-2, f"GPU energy {energy} != {expected_energy}"

    gpu_forces = context.d_forces.get().reshape(-1, 3)
    expected_force_magnitude = 2.0 * 100.0 * (2.0 - 1.5)
    assert abs(gpu_forces[0, 0] - expected_force_magnitude) < 0.1
    assert abs(gpu_forces[1, 0] + expected_force_magnitude) < 0.1


def test_angle_force_gpu():
    topology, term_params = _build_three_particle()
    parameter_table = _make_parameter_table(term_params)
    positions = np.array([
        [0.0, 1.5, 0.0],
        [0.0, 0.0, 0.0],
        [1.5 * np.cos(np.pi / 6), 1.5 * np.sin(np.pi / 6), 0.0],
    ], dtype=env.NUMPY_FLOAT)
    pbc_matrix = _make_large_pbc()

    force = BondedForce.charmm(topology, parameter_table)
    context = MockGPUContext(positions, pbc_matrix)
    force.compute(context)
    energy = float(context.d_energy[0])

    assert np.isfinite(energy)
    assert energy != 0.0

    gpu_forces = context.d_forces.get().reshape(-1, 3)
    total_force = np.sum(gpu_forces, axis=0)
    assert np.linalg.norm(total_force) < 0.5


def test_dihedral_force_gpu():
    topology, term_params = _build_four_particle()
    parameter_table = _make_parameter_table(term_params)
    positions = np.array([
        [0.0, 1.5, 0.0],
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [1.5, 0.0, 1.5],
    ], dtype=env.NUMPY_FLOAT)
    pbc_matrix = _make_large_pbc()

    force = BondedForce.charmm(topology, parameter_table)
    context = MockGPUContext(positions, pbc_matrix)
    force.compute(context)
    energy = float(context.d_energy[0])

    assert energy != 0.0

    gpu_forces = context.d_forces.get().reshape(-1, 3)
    total_force = np.sum(gpu_forces, axis=0)
    assert np.linalg.norm(total_force) < 0.5


def test_improper_force_gpu():
    topology, term_params = _build_four_particle()
    parameter_table = _make_parameter_table(term_params)
    positions = np.array([
        [0.0, 1.5, 0.0],
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [1.5, 0.0, 1.5],
    ], dtype=env.NUMPY_FLOAT)
    pbc_matrix = _make_large_pbc()

    force = BondedForce.charmm(topology, parameter_table)
    context = MockGPUContext(positions, pbc_matrix)
    force.compute(context)

    gpu_forces = context.d_forces.get().reshape(-1, 3)
    total_force = np.sum(gpu_forces, axis=0)
    assert np.linalg.norm(total_force) < 0.5


def test_empty_terms_gpu():
    topology, term_params = _build_empty()
    parameter_table = _make_parameter_table(term_params)
    positions = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=env.NUMPY_FLOAT)
    pbc_matrix = _make_large_pbc()

    force = BondedForce.charmm(topology, parameter_table)
    context = MockGPUContext(positions, pbc_matrix)
    force.compute(context)
    energy = float(context.d_energy[0])

    assert abs(energy) < 1e-6, f"Empty topology GPU energy should be 0, got {energy}"


def test_cpu_gpu_consistency():
    topology, term_params = _build_four_particle()
    parameter_table = _make_parameter_table(term_params)
    positions = np.array([
        [0.0, 1.5, 0.0],
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [1.5, 0.0, 1.5],
    ], dtype=env.NUMPY_FLOAT)
    pbc_matrix = _make_large_pbc()

    force = BondedForce.charmm(topology, parameter_table)

    context = MockGPUContext(positions, pbc_matrix)
    force.compute(context)
    gpu_forces = context.d_forces.get().reshape(-1, 3)

    gpu_total = np.sum(gpu_forces, axis=0)
    assert np.linalg.norm(gpu_total) < 0.5, \
        f"GPU forces not balanced: sum={gpu_total}"
