import numpy as np
import pytest
from mdpy.core.topology import Builder
from mdpy.core.parameter_table import ParameterTable
from mdpy.force.bonded_force import BondedForce
from mdpy.system import System
from mdpy.integrator.verlet import VerletIntegrator
from mdpy.constraint.constraint_scheme import create_constraints


def _build_test_system():
    builder = Builder()
    n_waters = 3
    n_atoms = n_waters * 3
    masses = np.zeros(n_atoms, dtype=np.float32)
    charges = np.zeros(n_atoms, dtype=np.float32)
    ptypes = np.zeros(n_atoms, dtype=np.int32)
    mol_ids = np.zeros(n_atoms, dtype=np.int32)
    positions = np.zeros((n_atoms, 3), dtype=np.float32)

    dOH = 1.0
    dHH = 1.63298
    half_hh = dHH / 2.0
    height = np.sqrt(dOH**2 - half_hh**2)

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
        builder.add_bond(ow, hw1, 450.0, dOH)
        builder.add_bond(hw1, hw2, 450.0, dHH)
        idx += 3

    builder.set_particles(masses, charges, ptypes, mol_ids)
    topology, term_params = builder.build()
    pbc_matrix = np.eye(3, dtype=np.float32) * 30.0

    pt = ParameterTable()
    for name, values in term_params.items():
        pt.add_term_parameter(name, values)

    return topology, pbc_matrix, pt, positions


def test_constraint_loop():
    topology, pbc_matrix, parameter_table, positions = _build_test_system()

    system = System(topology, pbc_matrix, cutoff=12.0)

    bonded = BondedForce.charmm(topology, parameter_table)
    system.add_force_term(bonded)

    constraints = create_constraints(topology, parameter_table, scheme='h-bonds')
    for c in constraints:
        system.add_constraint(c)

    system.upload_positions(positions)

    dt = 0.002
    velocities = np.random.RandomState(42).randn(*positions.shape).astype(np.float32) * 0.001
    system.upload_velocities(velocities)

    integrator = VerletIntegrator(dt)

    for step in range(5):
        system.update_neighbor_list(sync_interval=10)
        system.compute_forces()
        integrator.step(system)
        system.apply_constraints(dt)

    pos, vel = system.dump_state()
    assert pos.shape == positions.shape
    assert not np.any(np.isnan(pos))


def test_constraint_loop_multiple_rebuilds():
    topology, pbc_matrix, parameter_table, positions = _build_test_system()

    system = System(topology, pbc_matrix, cutoff=12.0)

    bonded = BondedForce.charmm(topology, parameter_table)
    system.add_force_term(bonded)

    constraints = create_constraints(topology, parameter_table, scheme='h-bonds')
    for c in constraints:
        system.add_constraint(c)

    system.upload_positions(positions)
    velocities = np.random.RandomState(123).randn(*positions.shape).astype(np.float32) * 0.001
    system.upload_velocities(velocities)

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
