from __future__ import annotations

import numpy as np
import os

import cupy as cp
import pytest

from mdpy.core.state import State
from mdpy.core.topology import Topology
from mdpy.system import System
from mdpy.minimizer import SteepestDescentMinimizer, ConjugateGradientMinimizer, LBFGSMinimizer, FIREMinimizer


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
    state.d_forces_x[:] = cp.asarray(forces[:, 0])
    state.d_forces_y[:] = cp.asarray(forces[:, 1])
    state.d_forces_z[:] = cp.asarray(forces[:, 2])
    return System(topology, state)


class TestSteepestDescent:
    def test_step_moves_along_negative_gradient(self):
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
        forces = np.array([[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
        system = _make_system(2, positions, forces)
        minimizer = SteepestDescentMinimizer(step_size=0.1)
        minimizer.step(system)
        new_pos = system.state.download_positions()
        assert new_pos[0, 0] < 0.0
        assert new_pos[1, 0] > 2.0

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

    def test_compute_max_force_multi_block(self):
        num_particles = 300
        positions = np.zeros((num_particles, 3), dtype=np.float32)
        forces = np.zeros((num_particles, 3), dtype=np.float32)
        forces[200, 1] = 7.0
        system = _make_system(num_particles, positions, forces)
        minimizer = SteepestDescentMinimizer()
        max_f = minimizer.compute_max_force(system)
        assert max_f == pytest.approx(7.0)

    def test_compute_rms_force_multi_block(self):
        num_particles = 300
        positions = np.zeros((num_particles, 3), dtype=np.float32)
        forces = np.ones((num_particles, 3), dtype=np.float32)
        system = _make_system(num_particles, positions, forces)
        minimizer = SteepestDescentMinimizer()
        rms_f = minimizer.compute_rms_force(system)
        expected_rms = np.sqrt(3.0)
        assert rms_f == pytest.approx(expected_rms)


class TestConjugateGradient:
    def test_step_moves_toward_minimum(self):
        positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        forces = np.array([[-2.0, 0.0, 0.0]], dtype=np.float32)
        system = _make_system(1, positions, forces)
        minimizer = ConjugateGradientMinimizer(step_size=0.1)
        minimizer.step(system)
        new_pos = system.state.download_positions()
        assert new_pos[0, 0] < 0.0

    def test_step_size_zero_does_not_move(self):
        positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        forces = np.array([[-2.0, 0.0, 0.0]], dtype=np.float32)
        system = _make_system(1, positions, forces)
        minimizer = ConjugateGradientMinimizer(step_size=0.0)
        minimizer.step(system)
        new_pos = system.state.download_positions()
        np.testing.assert_array_equal(new_pos, positions)


class TestLBFGS:
    def test_step_moves_toward_minimum(self):
        positions = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        forces = np.array([[-2.0, 0.0, 0.0]], dtype=np.float32)
        system = _make_system(1, positions, forces)
        minimizer = LBFGSMinimizer(history_size=3, step_size=0.1)
        minimizer.step(system)
        new_pos = system.state.download_positions()
        assert new_pos[0, 0] < 1.0

    def test_step_size_zero_does_not_move(self):
        positions = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        forces = np.array([[-2.0, 0.0, 0.0]], dtype=np.float32)
        system = _make_system(1, positions, forces)
        minimizer = LBFGSMinimizer(history_size=3, step_size=0.0)
        minimizer.step(system)
        new_pos = system.state.download_positions()
        np.testing.assert_allclose(new_pos, positions)

    def test_two_step_reduces_force(self):
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
        system = _make_system(2, positions, np.zeros((2, 3), dtype=np.float32))
        minimizer = LBFGSMinimizer(history_size=3, step_size=0.1)

        for step in range(3):
            x0 = float(system.state.d_positions_x[0])
            x1 = float(system.state.d_positions_x[1])
            force_x = x1 - x0 - 1.0
            system.state.d_forces_x[0] = force_x
            system.state.d_forces_x[1] = -force_x
            minimizer.step(system)

        final_x0 = float(system.state.d_positions_x[0])
        final_x1 = float(system.state.d_positions_x[1])
        final_dist = abs(final_x1 - final_x0)
        assert final_dist < 2.0


class TestFIRE:
    def test_step_moves_toward_minimum(self):
        positions = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        forces = np.array([[-2.0, 0.0, 0.0]], dtype=np.float32)
        system = _make_system(1, positions, forces)
        minimizer = FIREMinimizer(time_step=0.5)
        minimizer.step(system)
        new_pos = system.state.download_positions()
        assert new_pos[0, 0] < 1.0

    def test_time_step_zero_does_not_move(self):
        positions = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        forces = np.array([[-2.0, 0.0, 0.0]], dtype=np.float32)
        system = _make_system(1, positions, forces)
        minimizer = FIREMinimizer(time_step=0.0)
        minimizer.step(system)
        new_pos = system.state.download_positions()
        np.testing.assert_allclose(new_pos, positions)


class TestMinimizerIntegration:

    @pytest.fixture(scope="class")
    def sixpo6_system(self):
        from mdpy.io.psf_parser import PSFParser
        from mdpy.io.pdb_parser import PDBParser
        from mdpy.io.charmm_toppar_parser import CharmmTopparParser
        from mdpy.force.factories.charmm import create_charmm_forces

        DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
        psf = PSFParser(os.path.join(DATA_DIR, "6PO6.psf"))
        pdb = PDBParser(os.path.join(DATA_DIR, "6PO6.pdb"))
        toppar = CharmmTopparParser(
            os.path.join(DATA_DIR, "par_all36_prot.prm"),
            os.path.join(DATA_DIR, "toppar_water_ions.str"),
        )
        topology = psf.topology
        parameter_set = toppar.resolve_parameter_set(topology, psf.particle_type_names)
        pbc_matrix = np.eye(3, dtype=np.float64) * 50.0
        state = State(topology.num_particles)
        state.set_pbc(pbc_matrix)
        state.set_positions(pdb.positions)
        state.set_charges(psf.particle_charges)
        state.set_masses(psf.particle_masses)
        state.set_type_indices(parameter_set.particle_type_indices)
        state.set_velocities(np.zeros((topology.num_particles, 3), dtype=np.float64))
        forces = create_charmm_forces(
            topology, parameter_set, pbc_matrix, cutoff=12.0,
        )
        system = System(topology, state)
        for f in forces["bonded"]:
            system.add_force_term(f)
        system.add_force_term(forces["nonbonded"])
        return system

    def test_sd_reduces_max_force(self, sixpo6_system):
        system = sixpo6_system
        system.update_neighbor_list()
        minimizer = SteepestDescentMinimizer(step_size=0.01)
        system.compute_forces()
        initial_max_f = minimizer.compute_max_force(system)
        for _ in range(100):
            system.compute_forces()
            minimizer.step(system)
        system.compute_forces()
        final_max_f = minimizer.compute_max_force(system)
        assert final_max_f < initial_max_f
