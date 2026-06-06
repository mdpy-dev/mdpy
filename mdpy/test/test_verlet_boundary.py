import numpy as np
import pytest

from mdpy import env
from mdpy.core.topology import Builder
from mdpy.system import System
from mdpy.integrator.verlet import VerletIntegrator


DT = 0.5


def _make_system(n_atoms, box_size, cutoff=None):
    if cutoff is None:
        cutoff = box_size / 2.0
    builder = Builder()
    builder.set_particles(
        masses=np.full(n_atoms, 12.0, dtype=env.NUMPY_FLOAT),
        charges=np.zeros(n_atoms, dtype=env.NUMPY_FLOAT),
        particle_types=np.zeros(n_atoms, dtype=env.NUMPY_INT),
    )
    topology, _ = builder.build()
    pbc_matrix = np.eye(3, dtype=env.NUMPY_FLOAT) * box_size
    system = System(topology, pbc_matrix, cutoff=cutoff)
    positions = np.zeros((n_atoms, 3), dtype=env.NUMPY_FLOAT)
    velocities = np.zeros((n_atoms, 3), dtype=env.NUMPY_FLOAT)
    system.upload_positions(positions)
    system.upload_velocities(velocities)
    positions_soa = (
        system.gpu.d_positions_x,
        system.gpu.d_positions_y,
        system.gpu.d_positions_z,
    )
    system.block_list.rebuild(
        positions_soa, topology, system.pbc_matrix, system.pbc_inv
    )
    system._permute_all_arrays()
    system.block_list.build_block_pairs(topology, system.pbc_matrix)
    return system


def test_atom_near_box_boundary_stays_bounded():
    box = 10.0
    system = _make_system(2, box)

    positions = np.array(
        [[9.95, 5.0, 5.0], [5.0, 5.0, 5.0]], dtype=env.NUMPY_FLOAT
    )
    system.upload_positions(positions)

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
        [[0.01, 5.0, 5.0], [5.0, 5.0, 5.0]], dtype=env.NUMPY_FLOAT
    )
    velocities = np.array(
        [[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=env.NUMPY_FLOAT
    )
    system.upload_positions(positions)
    system.upload_velocities(velocities)

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
    positions = rng.uniform(1.0, 49.0, size=(n, 3)).astype(env.NUMPY_FLOAT)
    velocities = np.zeros((n, 3), dtype=env.NUMPY_FLOAT)

    system = _make_system(n, box)
    system.upload_positions(positions)
    system.upload_velocities(velocities)

    integrator = VerletIntegrator(DT)
    for _ in range(100):
        system.update_neighbor_list(sync_interval=200)
        system.compute_forces()
        integrator.step(system)

    pos, _ = system.dump_state()
    assert np.all(np.isfinite(pos))
    assert np.all(pos >= -0.1)
    assert np.all(pos <= box + 0.1)
