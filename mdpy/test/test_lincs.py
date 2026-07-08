import numpy as np
import cupy as cp
import pytest
from mdpy.core.topology import Builder
from mdpy.core.state import State
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
    topology = Builder().set_particles(len(masses)).build()[0]
    state = State(topology.num_particles)
    state.set_pbc(pbc.flatten())
    state.set_positions(positions)
    state.set_prev_positions(positions.copy())
    perturbed = positions + np.random.randn(*positions.shape).astype(np.float32) * 0.005
    state.d_positions_x[:] = cp.asarray(perturbed[:, 0])
    state.d_positions_y[:] = cp.asarray(perturbed[:, 1])
    state.d_positions_z[:] = cp.asarray(perturbed[:, 2])
    identity_map = cp.arange(len(masses), dtype=np.int32)
    lincs.apply(state, 0.002, d_pdb_to_sorted=identity_map)
    corrected = state.download_positions()
    for c, (i, j) in enumerate(pairs):
        d = np.linalg.norm(corrected[i] - corrected[j])
        assert abs(d - lengths[c]) < 0.02, f"Bond {c} ({i}-{j}): {d} != {lengths[c]}"


def test_lincs_no_change_if_already_correct():
    np.random.seed(42)
    pairs, lengths, masses, positions, mol_ids, pbc = _make_ethane_system()
    lincs = LincsConstraint(pairs, lengths, masses, expansion_order=4, num_iterations=1)
    topology = Builder().set_particles(len(masses)).build()[0]
    state = State(topology.num_particles)
    state.set_pbc(pbc.flatten())
    state.set_positions(positions)
    state.set_prev_positions(positions.copy())
    identity_map = cp.arange(len(masses), dtype=np.int32)
    lincs.apply(state, 0.002, d_pdb_to_sorted=identity_map)
    corrected = state.download_positions()
    np.testing.assert_allclose(corrected, positions, atol=1e-3)


def _make_rebuild_test_system():
    from mdpy.core.topology import Builder
    from mdpy.core.parameter_table import ParameterTable
    from mdpy.force.bonded_force import BondedForce
    from mdpy.force.factories.charmm import create_bonded_forces
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
    builder.set_particles(8)
    topology, term_params = builder.build()

    pt = ParameterTable()
    for name, values in term_params.items():
        pt.add_term_parameter(name, values)

    pbc_matrix = np.diag([20.0, 20.0, 20.0]).astype(np.float32)
    return topology, pbc_matrix, pt, positions, masses


def test_lincs_multiple_rebuilds():
    topology, pbc_matrix, parameter_table, positions, masses = _make_rebuild_test_system()

    from mdpy.force.bonded_force import BondedForce
    from mdpy.force.factories.charmm import create_bonded_forces
    from mdpy.system import System
    from mdpy.integrator.verlet import VerletIntegrator
    from mdpy.constraint.lincs import LincsConstraint

    state = State(8)
    state.set_masses(masses)
    state.set_charges(np.zeros(8, dtype=np.float32))
    state.set_type_indices(np.zeros(8, dtype=np.int32))
    system = System(topology, state)

    system.set_pbc(pbc_matrix)

    system._cutoff = 12.0
    bonded = create_bonded_forces(topology, parameter_table)
    for f in bonded:
        system.add_force_term(f)

    constraint_pairs = [
        (0, 1), (0, 2), (0, 3),
        (4, 5), (4, 6), (4, 7),
        (0, 4),
    ]
    target_lengths = [1.09, 1.09, 1.09, 1.09, 1.09, 1.09, 1.54]
    lincs = LincsConstraint(constraint_pairs, target_lengths, masses)
    system.add_constraint(lincs)

    system.set_positions(positions)
    system.set_velocities(np.random.RandomState(42).randn(*positions.shape).astype(np.float32) * 0.001)

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


def test_lincs_many_groups_multi_block():
    np.random.seed(42)
    n_groups = 50
    n_atoms = n_groups * 2
    masses = np.ones(n_atoms, dtype=np.float32)
    masses[0::2] = 12.0
    constraint_pairs = [(i * 2, i * 2 + 1) for i in range(n_groups)]
    target_lengths = [1.09] * n_groups
    positions = np.random.RandomState(42).rand(n_atoms, 3).astype(np.float32) * 10.0 + 50.0
    mol_ids = np.arange(n_atoms, dtype=np.int32)
    pbc_matrix = np.diag([100.0, 100.0, 100.0]).astype(np.float32)

    lincs = LincsConstraint(constraint_pairs, target_lengths, masses, expansion_order=4, num_iterations=1)

    assert lincs.num_constraints == n_groups
    assert lincs.num_constraint_threads > n_groups, (
        f"num_ct={lincs.num_constraint_threads} should be much larger than "
        f"num_constraints={n_groups} due to per-group block alignment"
    )

    topology = Builder().set_particles(n_atoms).build()[0]
    state = State(topology.num_particles)
    state.set_pbc(pbc_matrix.flatten())
    state.set_positions(positions)
    state.set_prev_positions(positions.copy())

    perturbed = positions + np.random.randn(*positions.shape).astype(np.float32) * 0.005
    state.d_positions_x[:] = cp.asarray(perturbed[:, 0])
    state.d_positions_y[:] = cp.asarray(perturbed[:, 1])
    state.d_positions_z[:] = cp.asarray(perturbed[:, 2])

    identity_map = cp.arange(n_atoms, dtype=np.int32)
    lincs.apply(state, 0.002, d_pdb_to_sorted=identity_map)

    cp.cuda.Stream.null.synchronize()

    corrected = state.download_positions()
    assert not np.any(np.isnan(corrected)), "Positions became NaN — likely illegal memory access in LINCS kernel"

    for c, (i, j) in enumerate(constraint_pairs):
        d = np.linalg.norm(corrected[i] - corrected[j])
        assert abs(d - target_lengths[c]) < 0.05, (
            f"Bond {c} ({i}-{j}): {d:.4f} != {target_lengths[c]}"
        )


def test_lincs_packing_efficiency():
    n_groups = 50
    n_atoms = n_groups * 2
    masses = np.ones(n_atoms, dtype=np.float32)
    masses[0::2] = 12.0
    constraint_pairs = [(i * 2, i * 2 + 1) for i in range(n_groups)]
    target_lengths = [1.09] * n_groups

    lincs = LincsConstraint(constraint_pairs, target_lengths, masses)

    n_blocks = lincs.num_constraint_threads // 256
    theoretical_min_blocks = (n_groups + 255) // 256
    assert n_blocks <= theoretical_min_blocks + 1, (
        f"Packing too loose: {n_blocks} blocks for {n_groups} single-constraint groups, "
        f"expected ~{theoretical_min_blocks}"
    )


def test_lincs_shared_atom_atomicAdd():
    np.random.seed(42)
    masses = np.array([12.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    positions = np.array([
        [5.0, 5.0, 5.0],
        [5.0, 5.0, 6.09],
        [5.0, 6.09, 5.0],
        [6.09, 5.0, 5.0],
        [5.0, 5.0, 3.91],
    ], dtype=np.float32)
    constraint_pairs = [(0, 1), (0, 2), (0, 3), (0, 4)]
    target_lengths = [1.09, 1.09, 1.09, 1.09]
    mol_ids = np.zeros(5, dtype=np.int32)
    pbc_matrix = np.diag([20.0, 20.0, 20.0]).astype(np.float32)

    lincs = LincsConstraint(constraint_pairs, target_lengths, masses, expansion_order=4, num_iterations=1)
    topology = Builder().set_particles(5).build()[0]
    state = State(topology.num_particles)
    state.set_pbc(pbc_matrix.flatten())
    state.set_positions(positions)
    state.set_prev_positions(positions.copy())
    perturbed = positions + np.random.randn(*positions.shape).astype(np.float32) * 0.01
    state.d_positions_x[:] = cp.asarray(perturbed[:, 0])
    state.d_positions_y[:] = cp.asarray(perturbed[:, 1])
    state.d_positions_z[:] = cp.asarray(perturbed[:, 2])
    identity_map = cp.arange(5, dtype=np.int32)
    lincs.apply(state, 0.002, d_pdb_to_sorted=identity_map)
    corrected = state.download_positions()
    for c, (i, j) in enumerate(constraint_pairs):
        d = np.linalg.norm(corrected[i] - corrected[j])
        assert abs(d - target_lengths[c]) < 0.02, f"Bond {c} ({i}-{j}): {d:.4f} != {target_lengths[c]}"


def test_lincs_ring_topology():
    np.random.seed(42)
    n_atoms = 6
    masses = np.full(n_atoms, 1.0, dtype=np.float32)
    spacing = 1.54
    positions = np.zeros((n_atoms, 3), dtype=np.float32)
    for i in range(n_atoms):
        angle = 2.0 * np.pi * i / n_atoms
        positions[i] = [5.0 + spacing * np.cos(angle), 5.0 + spacing * np.sin(angle), 5.0]
    constraint_pairs = [(i, (i + 1) % n_atoms) for i in range(n_atoms)]
    target_lengths = [spacing] * n_atoms
    mol_ids = np.zeros(n_atoms, dtype=np.int32)
    pbc_matrix = np.diag([20.0, 20.0, 20.0]).astype(np.float32)

    lincs = LincsConstraint(constraint_pairs, target_lengths, masses, expansion_order=4, num_iterations=1)
    topology = Builder().set_particles(n_atoms).build()[0]
    state = State(topology.num_particles)
    state.set_pbc(pbc_matrix.flatten())
    state.set_positions(positions)
    state.set_prev_positions(positions.copy())
    perturbed = positions + np.random.randn(*positions.shape).astype(np.float32) * 0.01
    state.d_positions_x[:] = cp.asarray(perturbed[:, 0])
    state.d_positions_y[:] = cp.asarray(perturbed[:, 1])
    state.d_positions_z[:] = cp.asarray(perturbed[:, 2])
    identity_map = cp.arange(n_atoms, dtype=np.int32)
    lincs.apply(state, 0.002, d_pdb_to_sorted=identity_map)
    corrected = state.download_positions()
    for c, (i, j) in enumerate(constraint_pairs):
        d = np.linalg.norm(corrected[i] - corrected[j])
        assert abs(d - target_lengths[c]) < 0.03, f"Ring bond {c} ({i}-{j}): {d:.4f} != {target_lengths[c]}"


def test_lincs_md_loop_bond_length_statistics():
    topology, pbc_matrix, parameter_table, positions, masses = _make_rebuild_test_system()

    from mdpy.force.bonded_force import BondedForce
    from mdpy.force.factories.charmm import create_bonded_forces
    from mdpy.system import System
    from mdpy.integrator.verlet import VerletIntegrator
    from mdpy.constraint.lincs import LincsConstraint

    state = State(8)
    state.set_masses(masses)
    state.set_charges(np.zeros(8, dtype=np.float32))
    state.set_type_indices(np.zeros(8, dtype=np.int32))
    system = System(topology, state)

    system.set_pbc(pbc_matrix)

    system._cutoff = 12.0
    bonded = create_bonded_forces(topology, parameter_table)
    for f in bonded:
        system.add_force_term(f)

    constraint_pairs = [
        (0, 1), (0, 2), (0, 3),
        (4, 5), (4, 6), (4, 7),
        (0, 4),
    ]
    target_lengths = [1.09, 1.09, 1.09, 1.09, 1.09, 1.09, 1.54]
    lincs = LincsConstraint(constraint_pairs, target_lengths, masses)
    system.add_constraint(lincs)

    system.set_positions(positions)
    system.set_velocities(np.random.RandomState(42).randn(*positions.shape).astype(np.float32) * 0.001)

    integrator = VerletIntegrator(0.002)

    max_deviations = []
    for step in range(100):
        system.update_neighbor_list(sync_interval=1)
        system.compute_forces()
        integrator.step(system)
        system.apply_constraints(0.002)
        pos, _ = system.dump_state()
        assert not np.any(np.isnan(pos)), f"NaN at step {step}"
        step_max = 0.0
        for c, (i, j) in enumerate(constraint_pairs):
            d = np.linalg.norm(pos[i] - pos[j])
            step_max = max(step_max, abs(d - target_lengths[c]))
        max_deviations.append(step_max)

    overall_max = max(max_deviations)
    overall_mean = np.mean(max_deviations)
    assert overall_max < 0.02, f"Max bond deviation {overall_max:.4f} > 0.02"
    assert overall_mean < 0.005, f"Mean bond deviation {overall_mean:.4f} > 0.005"
