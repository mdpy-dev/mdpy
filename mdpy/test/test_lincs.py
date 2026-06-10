import numpy as np
import cupy as cp
import pytest
from mdpy.core.topology import Builder
from mdpy.core.gpu_context import GPUContext
from mdpy.constraint.lincs import LincsConstraint


def _make_ethane_system():
    masses = np.array([12.0, 1.0, 1.0, 1.0, 12.0, 1.0, 1.0, 1.0], dtype=np.float32)
    positions = np.array([
        [5.0, 5.0, 5.0],
        [5.0, 5.0, 6.09],
        [5.0, 6.09, 5.0],
        [6.09, 5.0, 5.0],
        [6.54, 5.0, 5.0],
        [6.54, 5.0, 6.09],
        [6.54, 6.09, 5.0],
        [7.63, 5.0, 5.0],
    ], dtype=np.float32)
    constraint_pairs = [
        (0, 1), (0, 2), (0, 3),
        (4, 5), (4, 6), (4, 7),
        (0, 4),
    ]
    target_lengths = [1.09, 1.09, 1.09, 1.09, 1.09, 1.09, 1.54]
    mol_ids = np.zeros(8, dtype=np.int32)
    pbc_matrix = np.diag([20.0, 20.0, 20.0]).astype(np.float32)
    return constraint_pairs, target_lengths, masses, positions, mol_ids, pbc_matrix


def test_lincs_preserves_bond_lengths():
    np.random.seed(42)
    pairs, lengths, masses, positions, mol_ids, pbc = _make_ethane_system()
    lincs = LincsConstraint(pairs, lengths, masses, expansion_order=4, num_iterations=1)
    gpu = GPUContext()
    topology = Builder().set_particles(
        masses,
        np.zeros(len(masses), dtype=np.float32),
        np.zeros(len(masses), dtype=np.int32),
        mol_ids,
    ).build()[0]
    gpu.initialize(topology, pbc.flatten())
    gpu.upload_positions(positions)
    gpu.upload_prev_positions(positions.copy())
    perturbed = positions + np.random.randn(*positions.shape).astype(np.float32) * 0.005
    gpu.d_positions_x[:] = cp.asarray(perturbed[:, 0])
    gpu.d_positions_y[:] = cp.asarray(perturbed[:, 1])
    gpu.d_positions_z[:] = cp.asarray(perturbed[:, 2])
    identity_map = cp.arange(len(masses), dtype=np.int32)
    lincs.apply(gpu, 0.002, d_pdb_to_sorted=identity_map)
    corrected = gpu.download_positions()
    for c, (i, j) in enumerate(pairs):
        d = np.linalg.norm(corrected[i] - corrected[j])
        assert abs(d - lengths[c]) < 0.02, f"Bond {c} ({i}-{j}): {d} != {lengths[c]}"


def test_lincs_no_change_if_already_correct():
    np.random.seed(42)
    pairs, lengths, masses, positions, mol_ids, pbc = _make_ethane_system()
    lincs = LincsConstraint(pairs, lengths, masses, expansion_order=4, num_iterations=1)
    gpu = GPUContext()
    topology = Builder().set_particles(
        masses,
        np.zeros(len(masses), dtype=np.float32),
        np.zeros(len(masses), dtype=np.int32),
        mol_ids,
    ).build()[0]
    gpu.initialize(topology, pbc.flatten())
    gpu.upload_positions(positions)
    gpu.upload_prev_positions(positions.copy())
    identity_map = cp.arange(len(masses), dtype=np.int32)
    lincs.apply(gpu, 0.002, d_pdb_to_sorted=identity_map)
    corrected = gpu.download_positions()
    np.testing.assert_allclose(corrected, positions, atol=1e-3)


def _make_rebuild_test_system():
    from mdpy.core.topology import Builder
    from mdpy.core.parameter_table import ParameterTable
    from mdpy.force.bonded_force import BondedForce
    from mdpy.system import System
    from mdpy.integrator.verlet import VerletIntegrator
    from mdpy.constraint.constraint_scheme import create_constraints

    masses = np.array([12.0, 1.0, 1.0, 1.0, 12.0, 1.0, 1.0, 1.0], dtype=np.float32)
    charges = np.zeros(8, dtype=np.float32)
    ptypes = np.zeros(8, dtype=np.int32)
    mol_ids = np.zeros(8, dtype=np.int32)

    positions = np.array([
        [5.0, 5.0, 5.0],
        [5.0, 5.0, 6.09],
        [5.0, 6.09, 5.0],
        [6.09, 5.0, 5.0],
        [6.54, 5.0, 5.0],
        [6.54, 5.0, 6.09],
        [6.54, 6.09, 5.0],
        [7.63, 5.0, 5.0],
    ], dtype=np.float32)

    builder = Builder()
    builder.add_bond(0, 1, 450.0, 1.09)
    builder.add_bond(0, 2, 450.0, 1.09)
    builder.add_bond(0, 3, 450.0, 1.09)
    builder.add_bond(4, 5, 450.0, 1.09)
    builder.add_bond(4, 6, 450.0, 1.09)
    builder.add_bond(4, 7, 450.0, 1.09)
    builder.add_bond(0, 4, 450.0, 1.54)
    builder.set_particles(masses, charges, ptypes, mol_ids)
    topology, term_params = builder.build()

    pt = ParameterTable()
    for name, values in term_params.items():
        pt.add_term_parameter(name, values)

    pbc_matrix = np.diag([20.0, 20.0, 20.0]).astype(np.float32)
    return topology, pbc_matrix, pt, positions


def test_lincs_multiple_rebuilds():
    topology, pbc_matrix, parameter_table, positions = _make_rebuild_test_system()

    from mdpy.force.bonded_force import BondedForce
    from mdpy.system import System
    from mdpy.integrator.verlet import VerletIntegrator
    from mdpy.constraint.lincs import LincsConstraint

    system = System(topology, pbc_matrix, cutoff=12.0)
    bonded = BondedForce.charmm(topology, parameter_table)
    system.add_force_term(bonded)

    constraint_pairs = [
        (0, 1), (0, 2), (0, 3),
        (4, 5), (4, 6), (4, 7),
        (0, 4),
    ]
    target_lengths = [1.09, 1.09, 1.09, 1.09, 1.09, 1.09, 1.54]
    lincs = LincsConstraint(constraint_pairs, target_lengths, topology.masses)
    system.add_constraint(lincs)

    system.upload_positions(positions)
    system.upload_velocities(np.random.RandomState(42).randn(*positions.shape).astype(np.float32) * 0.001)

    integrator = VerletIntegrator(0.002)

    for step in range(20):
        system.update_neighbor_list(sync_interval=1)
        system.compute_forces()
        integrator.step(system)
        system.apply_constraints(0.002)

    pos, vel = system.dump_state()
    assert not np.any(np.isnan(pos)), "Positions became NaN after multiple rebuilds"

    for c, (i, j) in enumerate(constraint_pairs):
        d = np.linalg.norm(pos[i] - pos[j])
        assert abs(d - target_lengths[c]) < 0.05, (
            f"Bond {c} ({i}-{j}): {d:.4f} != {target_lengths[c]}, "
            f"diff={abs(d - target_lengths[c]):.4f}"
        )
