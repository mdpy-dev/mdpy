import numpy as np
import pytest

from mdpy import precision
from mdpy.core.topology import Topology
from mdpy.core.state import State
from mdpy.system import System
from mdpy.integrator.verlet import VerletIntegrator


DT = 0.5


def _make_system(n_atoms, box_size, cutoff=None):
    if cutoff is None:
        cutoff = box_size / 2.0
    topology = Topology()
    topology.num_particles = n_atoms
    pbc_matrix = np.eye(3, dtype=precision.FLOAT) * box_size
    state = State(n_atoms)
    state.set_masses(np.full(n_atoms, 12.0, dtype=precision.FLOAT))
    state.set_charges(np.zeros(n_atoms, dtype=precision.FLOAT))
    state.set_type_indices(np.zeros(n_atoms, dtype=precision.INT))
    system = System(topology, state)
    system.set_pbc(pbc_matrix)
    system._cutoff = cutoff
    positions = np.zeros((n_atoms, 3), dtype=precision.FLOAT)
    velocities = np.zeros((n_atoms, 3), dtype=precision.FLOAT)
    system.set_positions(positions)
    system.set_velocities(velocities)
    system.update_neighbor_list(force_rebuild=True)
    return system


def test_atom_near_box_boundary_stays_bounded():
    box = 10.0
    system = _make_system(2, box)

    positions = np.array(
        [[9.95, 5.0, 5.0], [5.0, 5.0, 5.0]], dtype=precision.FLOAT
    )
    system.set_positions(positions)

    integrator = VerletIntegrator(DT)
    integrator.step(system)

    pos, _ = system.dump_state()
    assert np.all(np.isfinite(pos))
    assert np.all(pos >= -1e-6)
    assert np.all(pos <= box + 1e-6)


def test_velocity_near_boundary_correct():
    box = 10.0
    system = _make_system(2, box)

    positions = np.array(
        [[0.01, 5.0, 5.0], [5.0, 5.0, 5.0]], dtype=precision.FLOAT
    )
    velocities = np.array(
        [[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=precision.FLOAT
    )
    system.set_positions(positions)
    system.set_velocities(velocities)

    integrator = VerletIntegrator(DT)
    integrator.step(system)

    pos, vel = system.dump_state()
    assert np.all(np.isfinite(pos))
    assert np.all(np.isfinite(vel))
    assert abs(pos[0, 0] - 0.06) < 0.1


def test_multi_step_no_explosion():
    box = 50.0
    rng = np.random.RandomState(42)
    n = 32
    positions = rng.uniform(1.0, 49.0, size=(n, 3)).astype(precision.FLOAT)
    velocities = np.zeros((n, 3), dtype=precision.FLOAT)

    system = _make_system(n, box)
    system.set_positions(positions)
    system.set_velocities(velocities)

    integrator = VerletIntegrator(DT)
    for _ in range(100):
        system.update_neighbor_list(sync_interval=200)
        system.compute_forces()
        integrator.step(system)

    pos, _ = system.dump_state()
    assert np.all(np.isfinite(pos))
    assert np.all(pos >= -0.1)
    assert np.all(pos <= box + 0.1)
