from __future__ import annotations

import numpy as np
import cupy as cp
import pytest

from mdpy.core.state import State
from mdpy.core.topology import Topology
from mdpy.system import System
from mdpy.minimizer import SteepestDescentMinimizer


def _make_system(num_particles, positions, forces, masses=None):
    topology = Topology()
    topology.num_particles = num_particles
    state = State(num_particles)
    state.set_positions(np.asarray(positions, dtype=np.float32))
    state.set_velocities(np.zeros((num_particles, 3), dtype=np.float32))
    state.set_charges(np.zeros(num_particles, dtype=np.float32))
    state.set_masses(np.full(num_particles, 1.0, dtype=np.float32)
                     if masses is None
                     else np.asarray(masses, dtype=np.float32))
    state.set_type_indices(np.zeros(num_particles, dtype=np.int32))
    state.set_pbc(np.eye(3, dtype=np.float32) * 100.0)
    state.d_forces_x[:] = cp.asarray(np.asarray(forces[:, 0], dtype=np.float32))
    state.d_forces_y[:] = cp.asarray(np.asarray(forces[:, 1], dtype=np.float32))
    state.d_forces_z[:] = cp.asarray(np.asarray(forces[:, 2], dtype=np.float32))
    return System(topology, state)


class TestSteepestDescent:
    def test_step_moves_along_negative_gradient(self):
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
        forces = np.array([[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
        system = _make_system(2, positions, forces)
        minimizer = SteepestDescentMinimizer(step_size=0.1)
        minimizer.step(system)
        new_pos = system.state.download_positions()
        assert new_pos[0, 0] > 0.0
        assert new_pos[1, 0] < 2.0

    def test_step_size_zero_does_not_move(self):
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
        forces = np.array([[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
        system = _make_system(2, positions, forces)
        minimizer = SteepestDescentMinimizer(step_size=0.0)
        minimizer.step(system)
        new_pos = system.state.download_positions()
        np.testing.assert_array_equal(new_pos, positions)

    def test_zero_force_does_not_move(self):
        positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        forces = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        system = _make_system(1, positions, forces)
        minimizer = SteepestDescentMinimizer(step_size=0.1)
        minimizer.step(system)
        new_pos = system.state.download_positions()
        np.testing.assert_allclose(new_pos, positions)

    def test_heavier_particle_moves_less(self):
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
        forces = np.array([[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
        system = _make_system(2, positions, forces, masses=np.array([1.0, 10.0]))
        minimizer = SteepestDescentMinimizer(step_size=0.1)
        minimizer.step(system)
        new_pos = system.state.download_positions()
        displacement_0 = abs(new_pos[0, 0] - positions[0, 0])
        displacement_1 = abs(new_pos[1, 0] - positions[1, 0])
        assert displacement_0 > displacement_1 * 5


class TestForceReduction:
    def test_compute_max_force_single_component(self):
        positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        forces = np.array([[3.0, 4.0, 0.0]], dtype=np.float32)
        system = _make_system(1, positions, forces)
        minimizer = SteepestDescentMinimizer()
        max_f = minimizer.compute_max_force(system)
        assert max_f == pytest.approx(4.0)

    def test_compute_max_force_multiple_particles(self):
        positions = np.zeros((3, 3), dtype=np.float32)
        forces = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 3.0],
        ], dtype=np.float32)
        system = _make_system(3, positions, forces)
        minimizer = SteepestDescentMinimizer()
        max_f = minimizer.compute_max_force(system)
        assert max_f == pytest.approx(5.0)

    def test_compute_max_force_mass_scaled(self):
        positions = np.zeros((2, 3), dtype=np.float32)
        forces = np.array([
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ], dtype=np.float32)
        system = _make_system(2, positions, forces, masses=np.array([0.5, 2.0]))
        minimizer = SteepestDescentMinimizer()
        max_f = minimizer.compute_max_force(system)
        assert max_f == pytest.approx(2.0)

    def test_compute_rms_force(self):
        positions = np.zeros((2, 3), dtype=np.float32)
        forces = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ], dtype=np.float32)
        system = _make_system(2, positions, forces)
        minimizer = SteepestDescentMinimizer()
        rms_f = minimizer.compute_rms_force(system)
        expected_rms = np.sqrt((1.0**2 + 2.0**2) / 2)
        assert rms_f == pytest.approx(expected_rms)
