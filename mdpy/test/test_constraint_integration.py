import numpy as np
import pytest
from mdpy.core.topology import Topology
from mdpy.core.state import State
from mdpy.core.parameter_set import ParameterSet
from mdpy.force.bonded_force import BondedForce
from mdpy.force.factories.charmm import create_bonded_forces
from mdpy.system import System
from mdpy.integrator.verlet import VerletIntegrator
from mdpy.constraint.constraint_scheme import create_constraints


def _build_test_system():
    n_waters = 3
    n_atoms = n_waters * 3
    masses = np.zeros(n_atoms, dtype=np.float32)
    charges = np.zeros(n_atoms, dtype=np.float32)
    ptypes = np.zeros(n_atoms, dtype=np.int32)
    mol_ids = np.zeros(n_atoms, dtype=np.int32)
    molecule_types = [''] * n_atoms
    positions = np.zeros((n_atoms, 3), dtype=np.float32)

    dOH = 1.0
    dHH = 1.63298
    half_hh = dHH / 2.0
    height = np.sqrt(dOH**2 - half_hh**2)

    bond_params_list = []
    idx = 0
    for w in range(n_waters):
        ow = idx
        hw1 = idx + 1
        hw2 = idx + 2
        cx = 5.0 + w * 4.0
        cy = 5.0
        cz = 5.0
        positions[ow] = [cx, cy, cz]
        positions[hw1] = [cx + half_hh, cy + height, cz]
        positions[hw2] = [cx - half_hh, cy + height, cz]
        masses[ow] = 15.999
        masses[hw1] = 1.008
        masses[hw2] = 1.008
        mol_ids[ow] = w
        mol_ids[hw1] = w
        mol_ids[hw2] = w
        bond_params_list.append((ow, hw1, 450.0, dOH))
        bond_params_list.append((hw1, hw2, 450.0, dHH))
        idx += 3

    topology = Topology()
    topology.num_particles = n_atoms
    bond_term_params = []
    for i, j, k_val, r0_val in bond_params_list:
        topology.add_bond(i, j)
        bond_term_params.append([k_val, r0_val])
    pbc_matrix = np.eye(3, dtype=np.float32) * 30.0

    term_params = {'bond': np.array(bond_term_params, dtype=np.float32)}
    pt = ParameterSet()
    for name, values in term_params.items():
        pt.add_term_parameter(name, values)

    return topology, pbc_matrix, pt, positions, masses, mol_ids, molecule_types


def test_constraint_loop():
    topology, pbc_matrix, parameter_table, positions, masses, mol_ids, molecule_types = _build_test_system()

    n = topology.num_particles
    state = State(n)
    state.set_particle_masses(masses)
    state.set_particle_charges(np.zeros(n, dtype=np.float32))
    state.set_particle_type_indices(np.zeros(n, dtype=np.int32))
    system = System(topology, state)

    system.set_pbc(pbc_matrix)

    system._cutoff = 12.0

    bonded = create_bonded_forces(topology, parameter_table)
    for f in bonded:
        system.add_force_term(f)

    constraints = create_constraints(topology, parameter_table, scheme='h-bonds', particle_masses=masses, particle_molecule_ids=mol_ids, particle_molecule_types=molecule_types)
    for c in constraints:
        system.add_constraint(c)

    system.set_positions(positions)

    time_step = 0.002
    velocities = np.random.RandomState(42).randn(*positions.shape).astype(np.float32) * 0.001
    system.set_velocities(velocities)

    integrator = VerletIntegrator(time_step)

    for step in range(5):
        system.update_neighbor_list(sync_interval=10)
        system.compute_forces()
        integrator.step(system)
        system.apply_constraints(time_step)

    pos, vel = system.dump_state()
    assert pos.shape == positions.shape
    assert not np.any(np.isnan(pos))


def test_constraint_loop_multiple_rebuilds():
    topology, pbc_matrix, parameter_table, positions, masses, mol_ids, molecule_types = _build_test_system()

    n = topology.num_particles
    state = State(n)
    state.set_particle_masses(masses)
    state.set_particle_charges(np.zeros(n, dtype=np.float32))
    state.set_particle_type_indices(np.zeros(n, dtype=np.int32))
    system = System(topology, state)

    system.set_pbc(pbc_matrix)

    system._cutoff = 12.0

    bonded = create_bonded_forces(topology, parameter_table)
    for f in bonded:
        system.add_force_term(f)

    constraints = create_constraints(topology, parameter_table, scheme='h-bonds', particle_masses=masses, particle_molecule_ids=mol_ids, particle_molecule_types=molecule_types)
    for c in constraints:
        system.add_constraint(c)

    system.set_positions(positions)
    velocities = np.random.RandomState(123).randn(*positions.shape).astype(np.float32) * 0.001
    system.set_velocities(velocities)

    integrator = VerletIntegrator(0.002)

    for step in range(50):
        system.update_neighbor_list(sync_interval=1)
        system.compute_forces()
        integrator.step(system)
        system.apply_constraints(0.002)

    pos, vel = system.dump_state()
    assert pos.shape == positions.shape
    assert not np.any(np.isnan(pos)), "Positions became NaN"
    assert not np.any(np.isnan(vel)), "Velocities became NaN"


def _build_mixed_system():
    n_waters = 10
    n_ethane_atoms = 8
    n_atoms = n_waters * 3 + n_ethane_atoms

    masses = np.zeros(n_atoms, dtype=np.float32)
    charges = np.zeros(n_atoms, dtype=np.float32)
    ptypes = np.zeros(n_atoms, dtype=np.int32)
    mol_ids = np.zeros(n_atoms, dtype=np.int32)
    molecule_types = [''] * n_atoms
    positions = np.zeros((n_atoms, 3), dtype=np.float32)

    dOH = 1.0
    dHH = 1.63298
    half_hh = dHH / 2.0
    height = np.sqrt(dOH**2 - half_hh**2)

    bond_params_list = []
    idx = 0
    for w in range(n_waters):
        ow = idx
        hw1 = idx + 1
        hw2 = idx + 2
        cx = 5.0 + w * 2.5
        cy = 5.0
        cz = 5.0
        positions[ow] = [cx, cy, cz]
        positions[hw1] = [cx + half_hh, cy + height, cz]
        positions[hw2] = [cx - half_hh, cy + height, cz]
        masses[ow] = 15.999
        masses[hw1] = 1.008
        masses[hw2] = 1.008
        mol_ids[ow] = w
        mol_ids[hw1] = w
        mol_ids[hw2] = w
        bond_params_list.append((ow, hw1, 450.0, dOH))
        bond_params_list.append((hw1, hw2, 450.0, dHH))
        idx += 3

    ebase = idx
    ethane_positions = np.array([
        [5.0, 15.0, 5.0],
        [5.0, 15.0, 6.09],
        [5.0, 16.09, 5.0],
        [6.09, 15.0, 5.0],
        [6.54, 15.0, 5.0],
        [6.54, 15.0, 6.09],
        [6.54, 16.09, 5.0],
        [7.63, 15.0, 5.0],
    ], dtype=np.float32)
    ethane_masses = [12.0, 1.0, 1.0, 1.0, 12.0, 1.0, 1.0, 1.0]
    ethane_bonds = [(0, 1), (0, 2), (0, 3), (4, 5), (4, 6), (4, 7), (0, 4)]
    for i in range(8):
        positions[ebase + i] = ethane_positions[i]
        masses[ebase + i] = ethane_masses[i]
        mol_ids[ebase + i] = n_waters
    for bi, bj in ethane_bonds:
        bond_params_list.append((ebase + bi, ebase + bj, 450.0, 1.09 if bj != 4 else 1.54))

    topology = Topology()
    topology.num_particles = n_atoms
    bond_term_params = []
    for i, j, k_val, r0_val in bond_params_list:
        topology.add_bond(i, j)
        bond_term_params.append([k_val, r0_val])
    pbc_matrix = np.eye(3, dtype=np.float32) * 40.0

    term_params = {'bond': np.array(bond_term_params, dtype=np.float32)}
    pt = ParameterSet()
    for name, values in term_params.items():
        pt.add_term_parameter(name, values)

    return topology, pbc_matrix, pt, positions, masses, mol_ids, molecule_types


def test_settle_lincs_coexistence_bond_lengths():
    topology, pbc_matrix, parameter_table, positions, masses, mol_ids, molecule_types = _build_mixed_system()

    n = topology.num_particles
    state = State(n)
    state.set_particle_masses(masses)
    state.set_particle_charges(np.zeros(n, dtype=np.float32))
    state.set_particle_type_indices(np.zeros(n, dtype=np.int32))
    system = System(topology, state)

    system.set_pbc(pbc_matrix)

    system._cutoff = 12.0
    bonded = create_bonded_forces(topology, parameter_table)
    for f in bonded:
        system.add_force_term(f)

    constraints = create_constraints(topology, parameter_table, scheme='h-bonds', particle_masses=masses, particle_molecule_ids=mol_ids, particle_molecule_types=molecule_types)
    for c in constraints:
        system.add_constraint(c)

    system.set_positions(positions)
    system.set_velocities(np.random.RandomState(42).randn(*positions.shape).astype(np.float32) * 0.001)

    integrator = VerletIntegrator(0.002)
    for step in range(100):
        system.update_neighbor_list(sync_interval=1)
        system.compute_forces()
        integrator.step(system)
        system.apply_constraints(0.002)

    pos, vel = system.dump_state()
    assert not np.any(np.isnan(pos))
    assert not np.any(np.isnan(vel))

    dOH = 1.0
    dHH = 1.63298
    for w in range(10):
        ow = w * 3
        hw1 = w * 3 + 1
        hw2 = w * 3 + 2
        d_oh1 = np.linalg.norm(pos[ow] - pos[hw1])
        d_oh2 = np.linalg.norm(pos[ow] - pos[hw2])
        d_hh = np.linalg.norm(pos[hw1] - pos[hw2])
        assert abs(d_oh1 - dOH) < 1e-3, f"Water {w} O-H1: {d_oh1:.4f}"
        assert abs(d_oh2 - dOH) < 1e-3, f"Water {w} O-H2: {d_oh2:.4f}"
        assert abs(d_hh - dHH) < 1e-3, f"Water {w} H-H: {d_hh:.4f}"

    ebase = 30
    ethane_bonds = [(0, 1), (0, 2), (0, 3), (4, 5), (4, 6), (4, 7), (0, 4)]
    target_lengths = [1.09, 1.09, 1.09, 1.09, 1.09, 1.09, 1.54]
    for c, (bi, bj) in enumerate(ethane_bonds):
        d = np.linalg.norm(pos[ebase + bi] - pos[ebase + bj])
        assert abs(d - target_lengths[c]) < 0.05, f"Ethane bond {c} ({bi}-{bj}): {d:.4f}"


def test_settle_md_loop_rebuilds_bond_lengths():
    topology, pbc_matrix, parameter_table, positions, masses, mol_ids, molecule_types = _build_test_system()

    n = topology.num_particles
    state = State(n)
    state.set_particle_masses(masses)
    state.set_particle_charges(np.zeros(n, dtype=np.float32))
    state.set_particle_type_indices(np.zeros(n, dtype=np.int32))
    system = System(topology, state)

    system.set_pbc(pbc_matrix)

    system._cutoff = 4.0
    bonded = create_bonded_forces(topology, parameter_table)
    for f in bonded:
        system.add_force_term(f)

    constraints = create_constraints(topology, parameter_table, scheme='h-bonds', particle_masses=masses, particle_molecule_ids=mol_ids, particle_molecule_types=molecule_types)
    for c in constraints:
        system.add_constraint(c)

    system.set_positions(positions)
    system.set_velocities(np.random.RandomState(99).randn(*positions.shape).astype(np.float32) * 0.01)

    integrator = VerletIntegrator(0.002)
    dOH = 1.0
    dHH = 1.63298
    for step in range(100):
        system.update_neighbor_list(sync_interval=1)
        system.compute_forces()
        integrator.step(system)
        system.apply_constraints(0.002)

    pos, vel = system.dump_state()
    assert not np.any(np.isnan(pos))
    for w in range(3):
        ow = w * 3
        hw1 = w * 3 + 1
        hw2 = w * 3 + 2
        d_oh1 = np.linalg.norm(pos[ow] - pos[hw1])
        d_oh2 = np.linalg.norm(pos[ow] - pos[hw2])
        d_hh = np.linalg.norm(pos[hw1] - pos[hw2])
        assert abs(d_oh1 - dOH) < 5e-3, f"Water {w} O-H1: {d_oh1:.4f}"
        assert abs(d_oh2 - dOH) < 5e-3, f"Water {w} O-H2: {d_oh2:.4f}"
        assert abs(d_hh - dHH) < 5e-3, f"Water {w} H-H: {d_hh:.4f}"


def test_lincs_md_loop_rebuilds_bond_lengths():
    from mdpy.constraint.lincs import LincsConstraint

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

    bond_specs = [
        (0, 1, 450.0, 1.09),
        (0, 2, 450.0, 1.09),
        (0, 3, 450.0, 1.09),
        (4, 5, 450.0, 1.09),
        (4, 6, 450.0, 1.09),
        (4, 7, 450.0, 1.09),
        (0, 4, 450.0, 1.54),
    ]

    topology = Topology()
    topology.num_particles = len(masses)
    bond_term_params = []
    for i, j, k_val, r0_val in bond_specs:
        topology.add_bond(i, j)
        bond_term_params.append([k_val, r0_val])

    term_params = {'bond': np.array(bond_term_params, dtype=np.float32)}
    pt = ParameterSet()
    for name, values in term_params.items():
        pt.add_term_parameter(name, values)

    pbc_matrix = np.diag([15.0, 15.0, 15.0]).astype(np.float32)
    n = len(masses)
    state = State(n)
    state.set_particle_masses(masses)
    state.set_particle_charges(np.zeros(n, dtype=np.float32))
    state.set_particle_type_indices(np.zeros(n, dtype=np.int32))
    system = System(topology, state)
    system.set_pbc(pbc_matrix)
    system._cutoff = 4.0
    bonded = create_bonded_forces(topology, pt)
    for f in bonded:
        system.add_force_term(f)

    constraint_pairs = [(0, 1), (0, 2), (0, 3), (4, 5), (4, 6), (4, 7), (0, 4)]
    target_lengths = [1.09, 1.09, 1.09, 1.09, 1.09, 1.09, 1.54]
    lincs = LincsConstraint(constraint_pairs, target_lengths, masses)
    system.add_constraint(lincs)

    system.set_positions(positions)
    system.set_velocities(np.random.RandomState(77).randn(*positions.shape).astype(np.float32) * 0.01)

    integrator = VerletIntegrator(0.002)
    for step in range(100):
        system.update_neighbor_list(sync_interval=1)
        system.compute_forces()
        integrator.step(system)
        system.apply_constraints(0.002)

    pos, vel = system.dump_state()
    assert not np.any(np.isnan(pos))
    for c, (i, j) in enumerate(constraint_pairs):
        d = np.linalg.norm(pos[i] - pos[j])
        assert abs(d - target_lengths[c]) < 0.05, (
            f"Bond {c} ({i}-{j}): {d:.4f} != {target_lengths[c]}"
        )
