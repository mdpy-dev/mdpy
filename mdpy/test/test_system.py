import numpy as np
import pytest

from mdpy import env
from mdpy.core.topology import Builder
from mdpy.core.gpu_context import GPUContext
from mdpy.core.particle_table import ParticleTable
from mdpy.core.tile_list import TileList
from mdpy.forcefield.parameters import ParameterTable
from mdpy.force.bonded_force import BondedForce
from mdpy.system import System
from mdpy.integrator.verlet import VerletIntegrator
from mdpy.integrator.langevin import LangevinBAOABIntegrator


def _make_large_pbc():
    return np.eye(3, dtype=env.NUMPY_FLOAT) * 100.0


def _make_parameter_table(term_params):
    pt = ParameterTable()
    for name, values in term_params.items():
        pt.add_per_term(name, values)
    return pt


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
    builder.add_angle(0, 1, 2, 50.0, np.pi * 170 / 180, 0.0, 0.0)
    builder.add_angle(1, 2, 3, 50.0, np.pi * 170 / 180, 0.0, 0.0)
    builder.add_dihedral(0, 1, 2, 3, 20.0, 2.0, np.pi)
    return builder.build()


def _build_simple_bond():
    builder = Builder()
    builder.set_particles(
        masses=np.array([12.0, 12.0], dtype=env.NUMPY_FLOAT),
        charges=np.zeros(2, dtype=env.NUMPY_FLOAT),
        particle_types=np.zeros(2, dtype=env.NUMPY_INT),
    )
    builder.add_bond(0, 1, 200.0, 1.5)
    return builder.build()


def _four_particle_positions():
    return np.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.5, 0.0, 0.0],
    ], dtype=env.NUMPY_FLOAT)


class TestGPUContext:

    def test_initialize_cpu(self):
        topology, _ = _build_four_particle()
        pbc_matrix = _make_large_pbc()
        ctx = GPUContext()
        ctx.initialize(topology, pbc_matrix.flatten())

        assert ctx.number_particles == 4
        assert ctx.d_positions_x.shape == (4,)
        assert ctx.d_positions_y.shape == (4,)
        assert ctx.d_positions_z.shape == (4,)
        assert ctx.d_velocities_x.shape == (4,)
        assert ctx.d_forces_x.shape == (4,)
        assert ctx.d_prev_positions_x.shape == (4,)
        assert ctx.d_masses.shape == (4,)
        assert ctx.d_energy.shape == (1,)
        assert ctx.d_pbc_matrix.shape == (9,)
        assert ctx.d_pbc_inv.shape == (9,)

    def test_upload_download_round_trip(self):
        topology, _ = _build_four_particle()
        pbc_matrix = _make_large_pbc()
        ctx = GPUContext()
        ctx.initialize(topology, pbc_matrix.flatten())

        table = ParticleTable(4)
        table.positions[:] = _four_particle_positions()
        table.velocities[:] = np.random.randn(4, 3).astype(env.NUMPY_FLOAT)

        original_positions = table.positions.copy()
        original_velocities = table.velocities.copy()

        ctx.upload_positions(table)
        ctx.upload_velocities(table)

        table.positions[:] = 999.0
        table.velocities[:] = 999.0

        ctx.download_positions(table)
        ctx.download_velocities(table)

        np.testing.assert_allclose(table.positions, original_positions, atol=1e-6)
        np.testing.assert_allclose(table.velocities, original_velocities, atol=1e-6)

    def test_zero_forces_energy(self):
        topology, _ = _build_four_particle()
        pbc_matrix = _make_large_pbc()
        ctx = GPUContext()
        ctx.initialize(topology, pbc_matrix.flatten())

        ctx.d_forces_x[:] = 1.0
        ctx.d_forces_y[:] = 1.0
        ctx.d_forces_z[:] = 1.0
        ctx.d_energy[:] = 42.0

        ctx.zero_forces()
        ctx.zero_energy()

        assert np.all(ctx.d_forces_x.get() == 0)
        assert np.all(ctx.d_forces_y.get() == 0)
        assert np.all(ctx.d_forces_z.get() == 0)
        assert ctx.d_energy[0] == 0


class TestSystem:

    def test_system_creation(self):
        topology, _ = _build_four_particle()
        pbc_matrix = _make_large_pbc()
        system = System(topology, pbc_matrix, cutoff=12.0)

        assert system.topology is topology
        assert system.particles.num_particles == 4
        assert isinstance(system.gpu, GPUContext)
        assert isinstance(system.tile_list, TileList)
        assert system.cutoff == 12.0
        assert system.dump_energy() == {}
        assert system.step_count == 0

    def test_system_add_force_term(self):
        topology, term_params = _build_four_particle()
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = _make_large_pbc()
        system = System(topology, pbc_matrix)

        bonded = BondedForce.charmm(topology, parameter_table)
        system.add_force_term(bonded)
        assert len(system.force_terms) == 1

    def test_system_compute_forces(self):
        topology, term_params = _build_four_particle()
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = _make_large_pbc()
        system = System(topology, pbc_matrix)

        bonded = BondedForce.charmm(topology, parameter_table)
        system.add_force_term(bonded)

        system.particles.positions[:] = np.array([
            [0.0, 0.0, 0.0],
            [1.6, 0.0, 0.0],
            [2.8, 0.5, 0.0],
            [4.5, 0.0, 1.0],
        ], dtype=env.NUMPY_FLOAT)
        system.gpu.upload_positions(system.particles)

        system.compute_forces()

        assert sum(system.dump_energy().values()) != 0.0
        assert all(np.isfinite(v) for v in system.dump_energy().values())
        assert 'bonded' in system.dump_energy()

        system.gpu.download_forces(system.particles)
        assert not np.all(system.particles.forces == 0)

    def test_system_single_step_verlet(self):
        topology, term_params = _build_simple_bond()
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = _make_large_pbc()
        system = System(topology, pbc_matrix)

        bonded = BondedForce.charmm(topology, parameter_table)
        system.add_force_term(bonded)

        system.particles.positions[:] = np.array([
            [0.0, 0.0, 0.0],
            [1.6, 0.0, 0.0],
        ], dtype=env.NUMPY_FLOAT)
        system.particles.velocities[:] = 0.0

        initial_positions = system.particles.positions.copy()

        integrator = VerletIntegrator(time_step=0.5)
        system.step(integrator, number_steps=1)
        system.dump_state()

        assert not np.allclose(system.particles.positions, initial_positions)
        assert np.all(np.isfinite(system.particles.positions))
        assert system.step_count == 1

    def test_system_hundred_steps_verlet(self):
        topology, term_params = _build_simple_bond()
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = _make_large_pbc()
        system = System(topology, pbc_matrix)

        bonded = BondedForce.charmm(topology, parameter_table)
        system.add_force_term(bonded)

        system.particles.positions[:] = np.array([
            [0.0, 0.0, 0.0],
            [1.6, 0.0, 0.0],
        ], dtype=env.NUMPY_FLOAT)
        system.particles.velocities[:] = 0.0

        integrator = VerletIntegrator(time_step=0.1)
        energies = []
        for _ in range(100):
            system.step(integrator, number_steps=1)
            energies.append(sum(system.dump_energy().values()))

        assert all(np.isfinite(energy) for energy in energies)
        assert system.step_count == 100

    def test_system_equilibrium_bond(self):
        topology, term_params = _build_simple_bond()
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = _make_large_pbc()
        system = System(topology, pbc_matrix)

        bonded = BondedForce.charmm(topology, parameter_table)
        system.add_force_term(bonded)

        system.particles.positions[:] = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
        ], dtype=env.NUMPY_FLOAT)
        system.particles.velocities[:] = 0.0

        integrator = VerletIntegrator(time_step=0.1)
        system.step(integrator, number_steps=1)

        total_energy = sum(system.dump_energy().values())
        assert abs(total_energy) < 1e-3, \
            f"At equilibrium, energy should be ~0, got {total_energy}"


class TestVerletIntegrator:

    def test_free_particle_no_drift(self):
        topology, _ = _build_simple_bond()
        pbc_matrix = _make_large_pbc()
        ctx = GPUContext()
        ctx.initialize(topology, pbc_matrix.flatten())

        import cupy as cp
        ctx.d_positions_x[:] = cp.asarray(np.array([10.0, 11.5], dtype=np.float32))
        ctx.d_positions_y[:] = cp.asarray(np.array([10.0, 10.0], dtype=np.float32))
        ctx.d_positions_z[:] = cp.asarray(np.array([10.0, 10.0], dtype=np.float32))
        ctx.d_velocities_x[:] = cp.asarray(np.array([0.01, -0.01], dtype=np.float32))
        ctx.d_velocities_y[:] = cp.asarray(np.array([0.0, 0.0], dtype=np.float32))
        ctx.d_velocities_z[:] = cp.asarray(np.array([0.0, 0.0], dtype=np.float32))
        ctx.d_forces_x[:] = 0.0
        ctx.d_forces_y[:] = 0.0
        ctx.d_forces_z[:] = 0.0

        integrator = VerletIntegrator(time_step=1.0)
        integrator.step(ctx)

        pos_x = ctx.d_positions_x.get()
        pos_y = ctx.d_positions_y.get()
        pos_z = ctx.d_positions_z.get()
        positions = np.stack([pos_x, pos_y, pos_z], axis=1)
        assert np.all(np.isfinite(positions))

        vel_x = ctx.d_velocities_x.get()
        vel_y = ctx.d_velocities_y.get()
        vel_z = ctx.d_velocities_z.get()
        velocities = np.stack([vel_x, vel_y, vel_z], axis=1)
        assert np.allclose(velocities[0], [0.01, 0.0, 0.0], atol=1e-4)
        assert np.allclose(velocities[1], [-0.01, 0.0, 0.0], atol=1e-4)

    def test_harmonic_oscillation(self):
        topology, term_params = _build_simple_bond()
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = _make_large_pbc()
        system = System(topology, pbc_matrix)
        bonded = BondedForce.charmm(topology, parameter_table)
        system.add_force_term(bonded)

        system.particles.positions[:] = np.array([
            [0.0, 0.0, 0.0],
            [1.6, 0.0, 0.0],
        ], dtype=env.NUMPY_FLOAT)
        system.particles.velocities[:] = 0.0

        integrator = VerletIntegrator(time_step=0.05)
        distances = []
        for _ in range(200):
            system.step(integrator, number_steps=1)
            system.dump_state()
            distance = np.linalg.norm(
                system.particles.positions[1] - system.particles.positions[0]
            )
            distances.append(distance)

        assert all(np.isfinite(distance) for distance in distances)
        min_distance = min(distances)
        max_distance = max(distances)
        assert min_distance < 1.6, f"Expected oscillation below 1.6, min={min_distance}"
        assert max_distance > 1.5, f"Expected oscillation around equilibrium 1.5, max={max_distance}"


class TestLangevinIntegrator:

    def test_langevin_step(self):
        topology, term_params = _build_simple_bond()
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = _make_large_pbc()
        system = System(topology, pbc_matrix)
        bonded = BondedForce.charmm(topology, parameter_table)
        system.add_force_term(bonded)

        system.particles.positions[:] = np.array([
            [0.0, 0.0, 0.0],
            [1.6, 0.0, 0.0],
        ], dtype=env.NUMPY_FLOAT)
        system.particles.velocities[:] = 0.0

        integrator = LangevinBAOABIntegrator(
            time_step=0.1, temperature=300.0, friction=0.1
        )
        system.step(integrator, number_steps=1)
        system.dump_state()

        assert np.all(np.isfinite(system.particles.positions))
        assert system.step_count == 1

    def test_langevin_temperature_relaxation(self):
        topology, term_params = _build_simple_bond()
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = _make_large_pbc()
        system = System(topology, pbc_matrix)
        bonded = BondedForce.charmm(topology, parameter_table)
        system.add_force_term(bonded)

        system.particles.positions[:] = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
        ], dtype=env.NUMPY_FLOAT)
        system.particles.velocities[:] = 0.0

        target_temperature = 300.0
        integrator = LangevinBAOABIntegrator(
            time_step=0.05, temperature=target_temperature, friction=1.0
        )

        for _ in range(500):
            system.step(integrator, number_steps=1)

        system.dump_state()
        positions = system.particles.positions
        assert np.all(np.isfinite(positions))

        kinetic_energy = 0.5 * 12.0 * np.sum(
            ((positions[1] - [1.5, 0.0, 0.0]) ** 2) / (0.05 ** 2)
        )
        measured_temperature = kinetic_energy / (1.5 * 8.314462618e-7)

        assert measured_temperature > 10.0, \
            f"Temperature too low: {measured_temperature}, thermostat should heat up"
        assert measured_temperature < 10000.0, \
            f"Temperature too high: {measured_temperature}, thermostat should regulate"

    def test_gpu_system_step(self):
        topology, term_params = _build_simple_bond()
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = _make_large_pbc()
        system = System(topology, pbc_matrix)
        bonded = BondedForce.charmm(topology, parameter_table)
        system.add_force_term(bonded)

        system.particles.positions[:] = np.array([
            [0.0, 0.0, 0.0],
            [1.6, 0.0, 0.0],
        ], dtype=env.NUMPY_FLOAT)
        system.particles.velocities[:] = 0.0

        integrator = VerletIntegrator(time_step=0.1)
        system.step(integrator, number_steps=10)
        assert np.all(np.isfinite(system.particles.positions))


class TestRebuildSortCorrectness:

    def test_positions_preserved_after_rebuild(self):
        topology, term_params = _build_four_particle()
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = _make_large_pbc()
        system = System(topology, pbc_matrix, cutoff=12.0, skin=1.0)
        bonded = BondedForce.charmm(topology, parameter_table)
        system.add_force_term(bonded)

        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.5, 0.0, 0.0],
        ], dtype=env.NUMPY_FLOAT)
        system.particles.positions[:] = positions
        system.particles.velocities[:] = 0.0

        integrator = VerletIntegrator(time_step=0.1)

        system.step(integrator, number_steps=5)
        system.dump_state()
        pos_after_first = system.particles.positions.copy()

        assert np.all(np.isfinite(pos_after_first))
        assert pos_after_first.shape == (4, 3)

        assert system._pdb_to_current_sorted is not None
        p2s = system._pdb_to_current_sorted
        assert len(np.unique(p2s)) == len(p2s), "pdb_to_sorted is not a permutation"

        s2p = system._sorted_to_pdb_np()
        assert len(np.unique(s2p)) == len(s2p), "sorted_to_pdb is not a permutation"

        identity = np.arange(4, dtype=env.NUMPY_INT)
        np.testing.assert_array_equal(s2p[p2s], identity,
            err_msg="sorted_to_pdb[pdb_to_sorted] != identity")

        for i in range(30):
            system.step(integrator, number_steps=1)

        system.dump_state()
        pos_after_30 = system.particles.positions.copy()
        assert np.all(np.isfinite(pos_after_30))

        p2s_2 = system._pdb_to_current_sorted
        assert len(np.unique(p2s_2)) == len(p2s_2), "pdb_to_sorted not permutation after more steps"
        s2p_2 = system._sorted_to_pdb_np()
        assert len(np.unique(s2p_2)) == len(s2p_2), "sorted_to_pdb not permutation after more steps"
        np.testing.assert_array_equal(s2p_2[p2s_2], identity,
            err_msg="permutation invariant broken after rebuild")

    def test_atom_identity_preserved_across_rebuilds(self):
        topology, term_params = _build_four_particle()
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = _make_large_pbc()
        system = System(topology, pbc_matrix, cutoff=12.0, skin=1.0)
        bonded = BondedForce.charmm(topology, parameter_table)
        system.add_force_term(bonded)

        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.5, 0.0, 0.0],
        ], dtype=env.NUMPY_FLOAT)
        velocities = np.array([
            [0.01, 0.0, 0.0],
            [-0.01, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [-0.01, 0.0, 0.0],
        ], dtype=env.NUMPY_FLOAT)
        system.particles.positions[:] = positions
        system.particles.velocities[:] = velocities

        integrator = VerletIntegrator(time_step=0.1)
        system.step(integrator, number_steps=1)
        system.dump_state()
        prev_pos = system.particles.positions.copy()

        for i in range(50):
            system.step(integrator, number_steps=1)

            pos, vel = system.dump_state()

            assert pos.shape == (4, 3), f"step {i}: wrong shape"
            assert vel.shape == (4, 3), f"step {i}: wrong velocity shape"
            assert np.all(np.isfinite(pos)), f"step {i}: NaN/Inf in positions"
            assert np.all(np.isfinite(vel)), f"step {i}: NaN/Inf in velocities"

            p2s = system._pdb_to_current_sorted
            assert len(np.unique(p2s)) == len(p2s), \
                f"step {i}: pdb_to_sorted is not a valid permutation"

            dists = np.linalg.norm(pos - prev_pos, axis=1)
            assert np.all(dists < 5.0), \
                f"step {i}: atom jumped too far (max={dists.max():.2f}), likely wrong PDB mapping"

            prev_pos = pos.copy()

    def test_rebuild_with_large_displacement(self):
        n = 50
        builder = Builder()
        builder.set_particles(
            masses=np.full(n, 12.0, dtype=env.NUMPY_FLOAT),
            charges=np.zeros(n, dtype=env.NUMPY_FLOAT),
            particle_types=np.zeros(n, dtype=env.NUMPY_INT),
        )
        for i in range(n - 1):
            builder.add_bond(i, i + 1, k=300.0, r0=1.5)
        builder.build_exclusion_map()
        topology, term_params = builder.build()
        parameter_table = _make_parameter_table(term_params)

        box = 80.0
        pbc_matrix = np.eye(3, dtype=env.NUMPY_FLOAT) * box
        system = System(topology, pbc_matrix, cutoff=10.0, skin=2.0)
        bonded = BondedForce.charmm(topology, parameter_table)
        system.add_force_term(bonded)

        rng = np.random.RandomState(42)
        positions = rng.uniform(10, 70, (n, 3)).astype(env.NUMPY_FLOAT)
        velocities = rng.randn(n, 3).astype(env.NUMPY_FLOAT) * 0.01
        system.particles.positions[:] = positions.copy()
        system.particles.velocities[:] = velocities.copy()

        integrator = VerletIntegrator(time_step=0.5)

        system.step(integrator, number_steps=1)
        system.dump_state()
        snapshot_before = system.particles.positions.copy()

        for step_i in range(100):
            system.step(integrator, number_steps=1)

        pos_final, vel_final = system.dump_state()

        assert pos_final.shape == (n, 3)
        assert np.all(np.isfinite(pos_final))
        assert np.all(np.isfinite(vel_final))

        p2s = system._pdb_to_current_sorted
        assert len(np.unique(p2s)) == len(p2s), "final pdb_to_sorted not a permutation"
        s2p = system._sorted_to_pdb_np()
        identity = np.arange(n, dtype=env.NUMPY_INT)
        np.testing.assert_array_equal(s2p[p2s], identity,
            err_msg="permutation invariant broken after 100 steps")

        max_disp_per_step = np.max(np.abs(pos_final - snapshot_before)) / 100.0
        assert max_disp_per_step < 1.0, \
            f"atoms moved {max_disp_per_step:.3f} per step on average, likely wrong PDB mapping"

    def test_second_rebuild_positions_consistent(self):
        n = 100
        builder = Builder()
        builder.set_particles(
            masses=np.full(n, 12.0, dtype=env.NUMPY_FLOAT),
            charges=np.zeros(n, dtype=env.NUMPY_FLOAT),
            particle_types=np.zeros(n, dtype=env.NUMPY_INT),
        )
        for i in range(n - 1):
            builder.add_bond(i, i + 1, k=300.0, r0=1.5)
        builder.build_exclusion_map()
        topology, term_params = builder.build()
        parameter_table = _make_parameter_table(term_params)

        box = 80.0
        pbc_matrix = np.eye(3, dtype=env.NUMPY_FLOAT) * box
        system = System(topology, pbc_matrix, cutoff=10.0, skin=1.0)
        bonded = BondedForce.charmm(topology, parameter_table)
        system.add_force_term(bonded)

        rng = np.random.RandomState(42)
        positions = rng.uniform(10, 70, (n, 3)).astype(env.NUMPY_FLOAT)
        velocities = rng.randn(n, 3).astype(env.NUMPY_FLOAT) * 0.001
        system.particles.positions[:] = positions.copy()
        system.particles.velocities[:] = velocities.copy()

        integrator = VerletIntegrator(time_step=0.5)

        integrator._initialized = False

        system.step(integrator, number_steps=1)
        pos_after_first = system.dump_state()[0]
        assert np.all(np.isfinite(pos_after_first))

        p2s_first = system._pdb_to_current_sorted.copy()
        s2p_first = system._sorted_to_pdb_np()
        identity = np.arange(n, dtype=env.NUMPY_INT)
        np.testing.assert_array_equal(s2p_first[p2s_first], identity,
            err_msg="First rebuild: permutation invariant broken")

        for step_i in range(200):
            system.step(integrator, number_steps=1)

        pos_after_many, vel_after_many = system.dump_state()
        assert np.all(np.isfinite(pos_after_many)), \
            "NaN/Inf in positions after 200 steps — permutation bug likely scrambled arrays"
        assert np.all(np.isfinite(vel_after_many)), \
            "NaN/Inf in velocities after 200 steps"

        p2s_final = system._pdb_to_current_sorted
        s2p_final = system._sorted_to_pdb_np()
        assert len(np.unique(p2s_final)) == n, "pdb_to_sorted not a permutation after 200 steps"
        assert len(np.unique(s2p_final)) == n, "sorted_to_pdb not a permutation after 200 steps"
        np.testing.assert_array_equal(s2p_final[p2s_final], identity,
            err_msg="Permutation invariant broken after multiple rebuilds")

        gpu_pos_x = system.gpu.d_positions_x.get()
        gpu_pos_y = system.gpu.d_positions_y.get()
        gpu_pos_z = system.gpu.d_positions_z.get()
        gpu_pos = np.stack([gpu_pos_x, gpu_pos_y, gpu_pos_z], axis=1)

        s2p_gpu = system._sorted_to_pdb_np()
        pdb_gpu_pos = gpu_pos[s2p_gpu]

        np.testing.assert_allclose(pdb_gpu_pos, pos_after_many, atol=1e-5,
            err_msg="GPU sorted positions don't match dump_state output — "
                     "sorted→PDB mapping is wrong")
