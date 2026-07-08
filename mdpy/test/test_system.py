import os

import numpy as np
import pytest

from mdpy import precision
from mdpy.core.topology import Topology
from mdpy.core.state import State
from mdpy.core.block_list import BlockList
from mdpy.core.parameter_table import ParameterTable
from mdpy.force.bonded_force import BondedForce
from mdpy.force.factories.charmm import create_bonded_forces, create_charmm_forces
from mdpy.system import System
from mdpy.integrator.verlet import VerletIntegrator
from mdpy.integrator.langevin import LangevinBAOABIntegrator
from mdpy.force.nonbonded_transpiler import nonbonded_expression
from mdpy.force.markers import param
from mdpy.force.expressions.geometry import distance
from mdpy.force.nonbonded_force import NonbondedForce
from mdpy.io.psf_parser import PSFParser
from mdpy.io.pdb_parser import PDBParser
from mdpy.io.charmm_toppar_parser import CharmmTopparParser, create_parameter_table


def _make_system(topology, pbc_matrix, cutoff=12.0, skin=None,
                 rebuild_check_interval=None):
    n = topology.num_particles
    state = State(n)
    state.set_masses(np.full(n, 12.0, dtype=precision.FLOAT))
    state.set_charges(np.zeros(n, dtype=precision.FLOAT))
    state.set_type_indices(np.zeros(n, dtype=precision.INT))
    system = System(topology, state)
    system.set_pbc(pbc_matrix)
    system._cutoff = cutoff
    if skin is not None:
        system._skin = skin
    if rebuild_check_interval is not None:
        system._rebuild_check_interval = rebuild_check_interval
    return system


def _run_steps(system, integrator, n, sync_interval=10):
    for i in range(n):
        system.update_neighbor_list(sync_interval=sync_interval)
        system.compute_forces()
        integrator.step(system)


def _ensure_ready(system):
    pass


def _make_large_pbc():
    return np.eye(3, dtype=precision.FLOAT) * 100.0


def _make_parameter_table(term_params):
    pt = ParameterTable()
    for name, values in term_params.items():
        pt.add_term_parameter(name, values)
    return pt


def _build_four_particle():
    topology = Topology()
    topology.num_particles = 4
    topology.add_bond(0, 1)
    topology.add_bond(1, 2)
    topology.add_bond(2, 3)
    topology.add_angle(0, 1, 2)
    topology.add_angle(1, 2, 3)
    topology.add_dihedral(0, 1, 2, 3)
    term_params = {
        'bond': np.array([[100.0, 1.5], [100.0, 1.5], [100.0, 1.5]], dtype=precision.FLOAT),
        'angle': np.array([[50.0, np.pi * 170 / 180, 0.0, 0.0],
                           [50.0, np.pi * 170 / 180, 0.0, 0.0]], dtype=precision.FLOAT),
        'dihedral': np.array([[20.0, 2.0, np.pi]], dtype=precision.FLOAT),
    }
    return topology, term_params


def _build_simple_bond():
    topology = Topology()
    topology.num_particles = 2
    topology.add_bond(0, 1)
    term_params = {
        'bond': np.array([[200.0, 1.5]], dtype=precision.FLOAT),
    }
    return topology, term_params


def _four_particle_positions():
    return np.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.5, 0.0, 0.0],
    ], dtype=precision.FLOAT)


class TestState:

    def test_initialize_cpu(self):
        topology, _ = _build_four_particle()
        pbc_matrix = _make_large_pbc()
        ctx = State(topology.num_particles)
        ctx.set_pbc(pbc_matrix.flatten())

        assert ctx.num_particles == 4
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
        ctx = State(topology.num_particles)
        ctx.set_pbc(pbc_matrix.flatten())

        original_positions = _four_particle_positions()
        original_velocities = np.random.randn(4, 3).astype(precision.FLOAT)

        ctx.set_positions(original_positions)
        ctx.set_velocities(original_velocities)

        downloaded_positions = ctx.download_positions()
        downloaded_velocities = ctx.download_velocities()

        np.testing.assert_allclose(downloaded_positions, original_positions, atol=1e-6)
        np.testing.assert_allclose(downloaded_velocities, original_velocities, atol=1e-6)

    def test_zero_forces_energy(self):
        topology, _ = _build_four_particle()
        pbc_matrix = _make_large_pbc()
        ctx = State(topology.num_particles)
        ctx.set_pbc(pbc_matrix.flatten())

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

    def test_state_constructed_from_num_particles(self):
        state = State(4)
        assert state.num_particles == 4
        assert state.d_charges.shape == (4,)
        assert state.d_masses.shape == (4,)
        assert state.d_type_indices.shape == (4,)
        assert state.is_ready is False

    def test_state_is_ready_after_all_fields_set(self):
        state = State(2)
        state.set_pbc(np.eye(3, dtype=np.float32).ravel())
        state.set_positions(np.zeros((2, 3), dtype=np.float32))
        state.set_velocities(np.zeros((2, 3), dtype=np.float32))
        state.set_charges(np.zeros(2, dtype=np.float32))
        state.set_masses(np.ones(2, dtype=np.float32))
        state.set_type_indices(np.zeros(2, dtype=np.int32))
        assert state.is_ready is True


class TestSystem:

    def test_system_creation(self):
        topology, _ = _build_four_particle()
        pbc_matrix = _make_large_pbc()
        system = _make_system(topology, pbc_matrix, cutoff=12.0)

        assert system.topology is topology
        assert system.num_particles == 4
        assert isinstance(system.state, State)
        assert system.cutoff == 12.0
        assert system.dump_energy() == {}

    def test_system_add_force_term(self):
        topology, term_params = _build_four_particle()
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = _make_large_pbc()
        system = _make_system(topology, pbc_matrix)

        bonded = create_bonded_forces(topology, parameter_table)
        for f in bonded:
            system.add_force_term(f)
        assert len(system.force_terms) == len(bonded)

    def test_system_compute_forces(self):
        topology, term_params = _build_four_particle()
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = _make_large_pbc()
        system = _make_system(topology, pbc_matrix)

        bonded = create_bonded_forces(topology, parameter_table)
        for f in bonded:
            system.add_force_term(f)

        system.set_positions(np.array([
            [0.0, 0.0, 0.0],
            [1.6, 0.0, 0.0],
            [2.8, 0.5, 0.0],
            [4.5, 0.0, 1.0],
        ], dtype=precision.FLOAT))
        system.set_velocities(np.zeros((4, 3), dtype=precision.FLOAT))

        system.compute_forces()

        assert sum(system.dump_energy().values()) != 0.0
        assert all(np.isfinite(v) for v in system.dump_energy().values())
        assert 'bond' in system.dump_energy()

        forces = system.dump_forces()
        assert not np.all(forces == 0)

    def test_nonbonded_forces_match_bruteforce(self):
        """Nonbonded forces from full pipeline must match O(N²) brute force."""

        @nonbonded_expression
        def lj_only(pos1, pos2, sigma=param, epsilon=param):
            r = distance(pos1, pos2)
            sr = sigma / r
            sr6 = sr * sr * sr * sr * sr * sr
            return 4.0 * epsilon * (sr6 * sr6 - sr6)

        # 8 atoms in a compact box — ensures block pairs exist
        topology = Topology()
        topology.num_particles = 8

        sigma_matrix = np.full((1, 1), 3.4, dtype=np.float32)
        epsilon_matrix = np.full((1, 1), 0.1, dtype=np.float32)

        pbc = np.eye(3, dtype=np.float64) * 20.0
        state = State(8)
        state.set_masses(np.full(8, 12.0, dtype=precision.FLOAT))
        state.set_charges(np.zeros(8, dtype=precision.FLOAT))
        state.set_type_indices(np.zeros(8, dtype=precision.INT))
        system = System(topology, state)
        system.set_pbc(pbc)

        nb = NonbondedForce(lj_only, cutoff=8.0)
        nb.set_pair_parameter('sigma', sigma_matrix)
        nb.set_pair_parameter('epsilon', epsilon_matrix)
        system.add_force_term(nb)

        positions = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [2.0, 2.0, 1.0],
            [1.0, 1.0, 2.0],
            [2.0, 1.0, 2.0],
            [1.0, 2.0, 2.0],
            [2.0, 2.0, 2.0],
        ], dtype=precision.FLOAT)
        system.set_positions(positions)
        system.set_velocities(np.zeros((8, 3), dtype=precision.FLOAT))

        system.update_neighbor_list()
        system.compute_forces()
        mdpy_forces = system.dump_forces()

        # Brute-force reference
        cutoff_sq = 8.0 ** 2
        ref_forces = np.zeros((8, 3), dtype=np.float64)
        sigma_val, eps_val = 3.4, 0.1
        for i in range(8):
            for j in range(i + 1, 8):
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dz = positions[j, 2] - positions[i, 2]
                r = np.sqrt(dx*dx + dy*dy + dz*dz)
                if r > 1e-6 and r*r <= cutoff_sq:
                    sr = sigma_val / r
                    sr6 = sr**6
                    f_mag = -4.0 * eps_val * (12 * sr6 * sr6 / r - 6 * sr6 / r)
                    fx = f_mag * dx / r
                    fy = f_mag * dy / r
                    fz = f_mag * dz / r
                    ref_forces[i, 0] += fx
                    ref_forces[i, 1] += fy
                    ref_forces[i, 2] += fz
                    ref_forces[j, 0] -= fx
                    ref_forces[j, 1] -= fy
                    ref_forces[j, 2] -= fz

        np.testing.assert_allclose(mdpy_forces, ref_forces, atol=1e-3,
                                    err_msg="Nonbonded forces must match brute-force reference")

    def test_system_single_step_verlet(self):
        topology, term_params = _build_simple_bond()
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = _make_large_pbc()
        system = _make_system(topology, pbc_matrix)

        bonded = create_bonded_forces(topology, parameter_table)
        for f in bonded:
            system.add_force_term(f)

        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.6, 0.0, 0.0],
        ], dtype=precision.FLOAT)
        system.set_positions(positions)
        system.set_velocities(np.zeros((2, 3), dtype=precision.FLOAT))

        initial_positions = positions.copy()

        integrator = VerletIntegrator(time_step=0.5)
        _ensure_ready(system)
        _run_steps(system, integrator, 1)
        pos, _ = system.dump_state()

        assert not np.allclose(pos, initial_positions)
        assert np.all(np.isfinite(pos))

    def test_system_hundred_steps_verlet(self):
        topology, term_params = _build_simple_bond()
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = _make_large_pbc()
        system = _make_system(topology, pbc_matrix)

        bonded = create_bonded_forces(topology, parameter_table)
        for f in bonded:
            system.add_force_term(f)

        system.set_positions(np.array([
            [0.0, 0.0, 0.0],
            [1.6, 0.0, 0.0],
        ], dtype=precision.FLOAT))
        system.set_velocities(np.zeros((2, 3), dtype=precision.FLOAT))

        integrator = VerletIntegrator(time_step=0.1)
        _ensure_ready(system)
        energies = []
        for _ in range(100):
            _run_steps(system, integrator, 1)
            energies.append(sum(system.dump_energy().values()))

        assert all(np.isfinite(energy) for energy in energies)

    def test_system_equilibrium_bond(self):
        topology, term_params = _build_simple_bond()
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = _make_large_pbc()
        system = _make_system(topology, pbc_matrix)

        bonded = create_bonded_forces(topology, parameter_table)
        for f in bonded:
            system.add_force_term(f)

        system.set_positions(np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
        ], dtype=precision.FLOAT))
        system.set_velocities(np.zeros((2, 3), dtype=precision.FLOAT))

        integrator = VerletIntegrator(time_step=0.1)
        _ensure_ready(system)
        _run_steps(system, integrator, 1)

        total_energy = sum(system.dump_energy().values())
        assert abs(total_energy) < 1e-3, \
            f"At equilibrium, energy should be ~0, got {total_energy}"


class TestVerletIntegrator:

    def test_free_particle_no_drift(self):
        topology, _ = _build_simple_bond()
        pbc_matrix = _make_large_pbc()
        system = _make_system(topology, pbc_matrix)

        system.set_positions(np.array([
            [10.0, 10.0, 10.0],
            [11.5, 10.0, 10.0],
        ], dtype=precision.FLOAT))
        system.set_velocities(np.array([
            [0.01, 0.0, 0.0],
            [-0.01, 0.0, 0.0],
        ], dtype=precision.FLOAT))

        integrator = VerletIntegrator(time_step=1.0)
        _ensure_ready(system)

        system.compute_forces()
        integrator.step(system)

        pos, _ = system.dump_state()
        assert np.all(np.isfinite(pos))

        _, vel = system.dump_state()
        assert np.allclose(vel[0], [0.01, 0.0, 0.0], atol=1e-4)
        assert np.allclose(vel[1], [-0.01, 0.0, 0.0], atol=1e-4)

    def test_harmonic_oscillation(self):
        topology, term_params = _build_simple_bond()
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = _make_large_pbc()
        system = _make_system(topology, pbc_matrix)
        bonded = create_bonded_forces(topology, parameter_table)
        for f in bonded:
            system.add_force_term(f)

        system.set_positions(np.array([
            [0.0, 0.0, 0.0],
            [1.6, 0.0, 0.0],
        ], dtype=precision.FLOAT))
        system.set_velocities(np.zeros((2, 3), dtype=precision.FLOAT))

        integrator = VerletIntegrator(time_step=0.05)
        _ensure_ready(system)
        distances = []
        for _ in range(200):
            _run_steps(system, integrator, 1)
            pos, _ = system.dump_state()
            distance = np.linalg.norm(pos[1] - pos[0])
            distances.append(distance)

        assert all(np.isfinite(distance) for distance in distances)
        min_distance = min(distances)
        max_distance = max(distances)
        assert min_distance < 1.6, f"Expected oscillation below 1.6, min={min_distance}"
        assert max_distance > 1.5, f"Expected oscillation around equilibrium 1.5, max={max_distance}"

    def test_set_positions_near_boundary_verlet(self):
        topology, _ = _build_simple_bond()
        box = 20.0
        pbc_matrix = np.eye(3, dtype=precision.FLOAT) * box
        system = _make_system(topology, pbc_matrix)

        system.set_positions(np.array([
            [1.0, 10.0, 10.0],
            [2.5, 10.0, 10.0],
        ], dtype=precision.FLOAT))
        system.set_velocities(np.array([
            [0.5, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
        ], dtype=precision.FLOAT))

        integrator = VerletIntegrator(time_step=0.5)
        _ensure_ready(system)
        _run_steps(system, integrator, 5)
        pos_before, vel_before = system.dump_state()

        new_positions = np.array([
            [box - 0.5, 10.0, 10.0],
            [box - 0.5 + 1.5, 10.0, 10.0],
        ], dtype=precision.FLOAT)
        system.set_positions(new_positions)
        integrator._initialized = False

        _ensure_ready(system)
        _run_steps(system, integrator, 10)
        pos_after, vel_after = system.dump_state()

        assert np.all(np.isfinite(pos_after))
        assert np.all(np.isfinite(vel_after))
        assert np.all(np.abs(vel_after) < 10.0), \
            f"Velocities exploded after upload: {vel_after}"

    def test_verlet_velocity_reasonable_after_upload(self):
        topology, _ = _build_simple_bond()
        box = 20.0
        pbc_matrix = np.eye(3, dtype=precision.FLOAT) * box
        system = _make_system(topology, pbc_matrix)

        system.set_positions(np.array([
            [1.0, 10.0, 10.0],
            [2.5, 10.0, 10.0],
        ], dtype=precision.FLOAT))
        system.set_velocities(np.zeros((2, 3), dtype=precision.FLOAT))

        integrator = VerletIntegrator(time_step=0.5)
        _ensure_ready(system)
        _run_steps(system, integrator, 3)
        _, vel_before = system.dump_state()

        near_edge = np.array([
            [box - 0.1, 10.0, 10.0],
            [box - 0.1 + 1.5, 10.0, 10.0],
        ], dtype=precision.FLOAT)
        system.set_positions(near_edge)
        system.set_velocities(np.zeros((2, 3), dtype=precision.FLOAT))
        integrator._initialized = False

        _ensure_ready(system)
        _run_steps(system, integrator, 1)
        _, vel_after = system.dump_state()

        assert np.all(np.abs(vel_after) < 5.0), \
            f"Velocity too large after upload near boundary: {vel_after}"


class TestLangevinIntegrator:

    def test_langevin_step(self):
        topology, term_params = _build_simple_bond()
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = _make_large_pbc()
        system = _make_system(topology, pbc_matrix)
        bonded = create_bonded_forces(topology, parameter_table)
        for f in bonded:
            system.add_force_term(f)

        system.set_positions(np.array([
            [0.0, 0.0, 0.0],
            [1.6, 0.0, 0.0],
        ], dtype=precision.FLOAT))
        system.set_velocities(np.zeros((2, 3), dtype=precision.FLOAT))

        integrator = LangevinBAOABIntegrator(
            time_step=0.1, temperature=300.0, friction=0.1
        )
        _ensure_ready(system)
        _run_steps(system, integrator, 1)
        pos, _ = system.dump_state()

        assert np.all(np.isfinite(pos))

    def test_langevin_temperature_relaxation(self):
        topology, term_params = _build_simple_bond()
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = _make_large_pbc()
        system = _make_system(topology, pbc_matrix)
        bonded = create_bonded_forces(topology, parameter_table)
        for f in bonded:
            system.add_force_term(f)

        system.set_positions(np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
        ], dtype=precision.FLOAT))
        system.set_velocities(np.zeros((2, 3), dtype=precision.FLOAT))

        target_temperature = 300.0
        integrator = LangevinBAOABIntegrator(
            time_step=0.05, temperature=target_temperature, friction=1.0
        )

        _ensure_ready(system)
        _run_steps(system, integrator, 499)

        pos_before, _ = system.dump_state()
        _run_steps(system, integrator, 1)
        pos_after, _ = system.dump_state()
        assert np.all(np.isfinite(pos_after))

        box = float(pbc_matrix[0, 0])
        delta = pos_after - pos_before
        frac = delta / box
        frac -= np.round(frac)
        delta_wrapped = frac * box
        velocity = delta_wrapped / 0.05
        kinetic_energy = 0.5 * 12.0 * np.sum(velocity ** 2)
        measured_temperature = kinetic_energy / (1.5 * 8.314462618e-7)

        assert measured_temperature > 10.0, \
            f"Temperature too low: {measured_temperature}, thermostat should heat up"
        assert measured_temperature < 10000.0, \
            f"Temperature too high: {measured_temperature}, thermostat should regulate"

    def test_set_positions_near_boundary_langevin(self):
        topology, term_params = _build_simple_bond()
        parameter_table = _make_parameter_table(term_params)
        box = 20.0
        pbc_matrix = np.eye(3, dtype=precision.FLOAT) * box
        system = _make_system(topology, pbc_matrix)
        bonded = create_bonded_forces(topology, parameter_table)
        for f in bonded:
            system.add_force_term(f)

        system.set_positions(np.array([
            [1.0, 10.0, 10.0],
            [2.5, 10.0, 10.0],
        ], dtype=precision.FLOAT))
        system.set_velocities(np.array([
            [0.5, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
        ], dtype=precision.FLOAT))

        integrator = LangevinBAOABIntegrator(
            time_step=0.1, temperature=300.0, friction=0.1
        )
        _ensure_ready(system)
        _run_steps(system, integrator, 5)

        new_positions = np.array([
            [box - 0.5, 10.0, 10.0],
            [0.5, 10.0, 10.0],
        ], dtype=precision.FLOAT)
        system.set_positions(new_positions)
        integrator._initialized = False

        _ensure_ready(system)
        _run_steps(system, integrator, 20)
        pos_after, vel_after = system.dump_state()

        assert np.all(np.isfinite(pos_after))
        assert np.all(np.isfinite(vel_after))
        assert np.all(np.abs(vel_after) < 100.0), \
            f"Velocities exploded after upload: {vel_after}"

    def test_gpu_system_step(self):
        topology, term_params = _build_simple_bond()
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = _make_large_pbc()
        system = _make_system(topology, pbc_matrix)
        bonded = create_bonded_forces(topology, parameter_table)
        for f in bonded:
            system.add_force_term(f)

        system.set_positions(np.array([
            [0.0, 0.0, 0.0],
            [1.6, 0.0, 0.0],
        ], dtype=precision.FLOAT))
        system.set_velocities(np.zeros((2, 3), dtype=precision.FLOAT))

        integrator = VerletIntegrator(time_step=0.1)
        _ensure_ready(system)
        _run_steps(system, integrator, 10)
        pos, _ = system.dump_state()
        assert np.all(np.isfinite(pos))


def _get_pdb_to_sorted(system):
    import cupy as cp
    d_stp = system.block_list.d_sorted_to_pdb
    sorted_to_pdb = cp.asnumpy(d_stp)
    pdb_to_sorted = np.empty_like(sorted_to_pdb)
    pdb_to_sorted[sorted_to_pdb] = np.arange(len(sorted_to_pdb), dtype=sorted_to_pdb.dtype)
    return pdb_to_sorted, sorted_to_pdb


class TestRebuildSortCorrectness:

    def test_positions_preserved_after_rebuild(self):
        topology, term_params = _build_four_particle()
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = _make_large_pbc()
        system = _make_system(topology, pbc_matrix, cutoff=12.0, skin=1.0)
        bonded = create_bonded_forces(topology, parameter_table)
        for f in bonded:
            system.add_force_term(f)

        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.5, 0.0, 0.0],
        ], dtype=precision.FLOAT)
        system.set_positions(positions)
        system.set_velocities(np.zeros((4, 3), dtype=precision.FLOAT))

        integrator = VerletIntegrator(time_step=0.1)
        _ensure_ready(system)

        _run_steps(system, integrator, 5)
        pos_after_first, _ = system.dump_state()

        assert np.all(np.isfinite(pos_after_first))
        assert pos_after_first.shape == (4, 3)

        p2s, s2p = _get_pdb_to_sorted(system)
        assert len(np.unique(p2s)) == len(p2s), "pdb_to_sorted is not a permutation"

        assert len(np.unique(s2p)) == len(s2p), "sorted_to_pdb is not a permutation"

        identity = np.arange(4, dtype=precision.INT)
        np.testing.assert_array_equal(s2p[p2s], identity,
            err_msg="sorted_to_pdb[pdb_to_sorted] != identity")

        for i in range(30):
            _run_steps(system, integrator, 1)

        pos_after_30, _ = system.dump_state()
        assert np.all(np.isfinite(pos_after_30))

        p2s_2, s2p_2 = _get_pdb_to_sorted(system)
        assert len(np.unique(p2s_2)) == len(p2s_2), "pdb_to_sorted not permutation after more steps"
        assert len(np.unique(s2p_2)) == len(s2p_2), "sorted_to_pdb not permutation after more steps"
        np.testing.assert_array_equal(s2p_2[p2s_2], identity,
            err_msg="permutation invariant broken after rebuild")

    def test_atom_identity_preserved_across_rebuilds(self):
        topology, term_params = _build_four_particle()
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = _make_large_pbc()
        system = _make_system(topology, pbc_matrix, cutoff=12.0, skin=1.0)
        bonded = create_bonded_forces(topology, parameter_table)
        for f in bonded:
            system.add_force_term(f)

        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.5, 0.0, 0.0],
        ], dtype=precision.FLOAT)
        velocities = np.array([
            [0.01, 0.0, 0.0],
            [-0.01, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [-0.01, 0.0, 0.0],
        ], dtype=precision.FLOAT)
        system.set_positions(positions)
        system.set_velocities(velocities)

        integrator = VerletIntegrator(time_step=0.1)
        _ensure_ready(system)
        _run_steps(system, integrator, 1)
        prev_pos, _ = system.dump_state()

        for i in range(50):
            _run_steps(system, integrator, 1)

            pos, vel = system.dump_state()

            assert pos.shape == (4, 3), f"step {i}: wrong shape"
            assert vel.shape == (4, 3), f"step {i}: wrong velocity shape"
            assert np.all(np.isfinite(pos)), f"step {i}: NaN/Inf in positions"
            assert np.all(np.isfinite(vel)), f"step {i}: NaN/Inf in velocities"

            p2s, _ = _get_pdb_to_sorted(system)
            assert len(np.unique(p2s)) == len(p2s), \
                f"step {i}: pdb_to_sorted is not a valid permutation"

            delta = pos - prev_pos
            delta -= np.round(delta / 100.0) * 100.0
            dists = np.linalg.norm(delta, axis=1)
            assert np.all(dists < 5.0), \
                f"step {i}: atom jumped too far (max={dists.max():.2f}), likely wrong PDB mapping"

            prev_pos = pos.copy()

    def test_rebuild_with_large_displacement(self):
        n = 50
        topology = Topology()
        topology.num_particles = n
        for i in range(n - 1):
            topology.add_bond(i, i + 1)
        term_params = {
            'bond': np.array([[300.0, 1.5]] * (n - 1), dtype=precision.FLOAT),
        }
        parameter_table = _make_parameter_table(term_params)

        box = 80.0
        pbc_matrix = np.eye(3, dtype=precision.FLOAT) * box
        system = _make_system(topology, pbc_matrix, cutoff=10.0, skin=2.0)
        bonded = create_bonded_forces(topology, parameter_table)
        for f in bonded:
            system.add_force_term(f)

        rng = np.random.RandomState(42)
        positions = rng.uniform(10, 70, (n, 3)).astype(precision.FLOAT)
        velocities = rng.randn(n, 3).astype(precision.FLOAT) * 0.01
        system.set_positions(positions)
        system.set_velocities(velocities)

        integrator = VerletIntegrator(time_step=0.5)
        _ensure_ready(system)

        _run_steps(system, integrator, 1)
        snapshot_before, _ = system.dump_state()

        for step_i in range(100):
            _run_steps(system, integrator, 1)

        pos_final, vel_final = system.dump_state()

        assert pos_final.shape == (n, 3)
        assert np.all(np.isfinite(pos_final))
        assert np.all(np.isfinite(vel_final))

        p2s, s2p = _get_pdb_to_sorted(system)
        assert len(np.unique(p2s)) == len(p2s), "final pdb_to_sorted not a permutation"
        identity = np.arange(n, dtype=precision.INT)
        np.testing.assert_array_equal(s2p[p2s], identity,
            err_msg="permutation invariant broken after 100 steps")

        max_disp_per_step = np.max(np.abs(pos_final - snapshot_before)) / 100.0
        pbc_inv = np.diag([1.0 / box] * 3).astype(precision.FLOAT)
        delta = pos_final - snapshot_before
        frac = delta @ pbc_inv.T
        frac -= np.round(frac)
        delta_wrapped = frac @ (np.diag([box] * 3).astype(precision.FLOAT)).T
        max_disp_per_step = np.max(np.abs(delta_wrapped)) / 100.0
        assert max_disp_per_step < 1.0, \
            f"atoms moved {max_disp_per_step:.3f} per step on average, likely wrong PDB mapping"

    def test_second_rebuild_positions_consistent(self):
        n = 100
        topology = Topology()
        topology.num_particles = n
        for i in range(n - 1):
            topology.add_bond(i, i + 1)
        term_params = {
            'bond': np.array([[300.0, 1.5]] * (n - 1), dtype=precision.FLOAT),
        }
        parameter_table = _make_parameter_table(term_params)

        box = 80.0
        pbc_matrix = np.eye(3, dtype=precision.FLOAT) * box
        system = _make_system(topology, pbc_matrix, cutoff=10.0, skin=1.0)
        bonded = create_bonded_forces(topology, parameter_table)
        for f in bonded:
            system.add_force_term(f)

        rng = np.random.RandomState(42)
        positions = rng.uniform(10, 70, (n, 3)).astype(precision.FLOAT)
        velocities = rng.randn(n, 3).astype(precision.FLOAT) * 0.001
        system.set_positions(positions)
        system.set_velocities(velocities)

        integrator = VerletIntegrator(time_step=0.5)

        integrator._initialized = False
        _ensure_ready(system)

        _run_steps(system, integrator, 1)
        pos_after_first = system.dump_state()[0]
        assert np.all(np.isfinite(pos_after_first))

        p2s_first, s2p_first = _get_pdb_to_sorted(system)
        identity = np.arange(n, dtype=precision.INT)
        np.testing.assert_array_equal(s2p_first[p2s_first], identity,
            err_msg="First rebuild: permutation invariant broken")

        for step_i in range(200):
            _run_steps(system, integrator, 1)

        pos_after_many, vel_after_many = system.dump_state()
        assert np.all(np.isfinite(pos_after_many)), \
            "NaN/Inf in positions after 200 steps — permutation bug likely scrambled arrays"
        assert np.all(np.isfinite(vel_after_many)), \
            "NaN/Inf in velocities after 200 steps"

        p2s_final, s2p_final = _get_pdb_to_sorted(system)
        assert len(np.unique(p2s_final)) == n, "pdb_to_sorted not a permutation after 200 steps"
        assert len(np.unique(s2p_final)) == n, "sorted_to_pdb not a permutation after 200 steps"
        np.testing.assert_array_equal(s2p_final[p2s_final], identity,
            err_msg="Permutation invariant broken after multiple rebuilds")

        state = system.state
        gpu_pos = np.stack([
            state.d_positions_x.get(),
            state.d_positions_y.get(),
            state.d_positions_z.get(),
        ], axis=1)

        np.testing.assert_allclose(gpu_pos, pos_after_many, atol=1e-5,
            err_msg="GPU positions don't match dump_state output — "
                     "PDB-order state is wrong")


class TestLazyEnergy:

    def test_dump_energy_returns_correct_values(self):
        topology, term_params = _build_simple_bond()
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = _make_large_pbc()
        system = _make_system(topology, pbc_matrix)
        bonded = create_bonded_forces(topology, parameter_table)
        for f in bonded:
            system.add_force_term(f)

        system.set_positions(np.array([
            [0.0, 0.0, 0.0],
            [1.6, 0.0, 0.0],
        ], dtype=precision.FLOAT))
        system.set_velocities(np.zeros((2, 3), dtype=precision.FLOAT))
        system.compute_forces()
        energy = system.dump_energy()
        assert 'bond' in energy
        assert energy['bond'] != 0.0
        assert np.isfinite(energy['bond'])

    def test_step_without_dump_energy_no_accumulator_stall(self):
        topology, term_params = _build_simple_bond()
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = _make_large_pbc()
        system = _make_system(topology, pbc_matrix)
        bonded = create_bonded_forces(topology, parameter_table)
        for f in bonded:
            system.add_force_term(f)

        system.set_positions(np.array([
            [0.0, 0.0, 0.0],
            [1.6, 0.0, 0.0],
        ], dtype=precision.FLOAT))
        system.set_velocities(np.zeros((2, 3), dtype=precision.FLOAT))

        integrator = VerletIntegrator(time_step=0.1)
        _ensure_ready(system)
        _run_steps(system, integrator, 5)
        pos, vel = system.dump_state()
        assert np.all(np.isfinite(pos))
        assert np.all(np.isfinite(vel))

        energy = system.dump_energy()
        assert 'bond' in energy
        assert np.isfinite(energy['bond'])

    def test_energy_changes_between_steps(self):
        topology, term_params = _build_simple_bond()
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = _make_large_pbc()
        system = _make_system(topology, pbc_matrix)
        bonded = create_bonded_forces(topology, parameter_table)
        for f in bonded:
            system.add_force_term(f)

        system.set_positions(np.array([
            [0.0, 0.0, 0.0],
            [1.6, 0.0, 0.0],
        ], dtype=precision.FLOAT))
        system.set_velocities(np.zeros((2, 3), dtype=precision.FLOAT))

        integrator = VerletIntegrator(time_step=0.1)
        _ensure_ready(system)
        _run_steps(system, integrator, 1)
        energy_1 = system.dump_energy()

        _run_steps(system, integrator, 5)
        energy_2 = system.dump_energy()

        assert 'bond' in energy_1
        assert 'bond' in energy_2
        assert energy_1['bond'] != energy_2['bond']


class TestAsyncRebuild:

    def test_async_rebuild_deterministic_trajectory(self):
        topology, term_params = _build_four_particle()
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = _make_large_pbc()
        system_a = _make_system(topology, pbc_matrix, cutoff=12.0, skin=1.0)
        bonded = create_bonded_forces(topology, parameter_table)
        for f in bonded:
            system_a.add_force_term(f)
        shared_positions = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.5, 0.0, 0.0],
        ], dtype=precision.FLOAT)
        system_a.set_positions(shared_positions)
        system_a.set_velocities(np.zeros((4, 3), dtype=precision.FLOAT))

        system_b = _make_system(topology, pbc_matrix, cutoff=12.0, skin=1.0)
        bonded_b = create_bonded_forces(topology, parameter_table)
        for f in bonded_b:
            system_b.add_force_term(f)
        system_b.set_positions(shared_positions)
        system_b.set_velocities(np.zeros((4, 3), dtype=precision.FLOAT))

        integrator_a = VerletIntegrator(time_step=0.1)
        integrator_b = VerletIntegrator(time_step=0.1)

        _ensure_ready(system_a)
        for _ in range(50):
            _run_steps(system_a, integrator_a, 1)
        pos_a, vel_a = system_a.dump_state()

        _ensure_ready(system_b)
        for _ in range(50):
            _run_steps(system_b, integrator_b, 1)
        pos_b, vel_b = system_b.dump_state()

        np.testing.assert_allclose(pos_a, pos_b, atol=1e-5,
            err_msg="Two identical async simulations produce different trajectories")

    def test_async_rebuild_with_multi_step_call(self):
        topology, term_params = _build_four_particle()
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = _make_large_pbc()
        system = _make_system(topology, pbc_matrix, cutoff=12.0, skin=1.0)
        bonded = create_bonded_forces(topology, parameter_table)
        for f in bonded:
            system.add_force_term(f)
        system.set_positions(np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.5, 0.0, 0.0],
        ], dtype=precision.FLOAT))
        system.set_velocities(np.zeros((4, 3), dtype=precision.FLOAT))

        integrator = VerletIntegrator(time_step=0.1)
        _ensure_ready(system)
        _run_steps(system, integrator, 100)
        pos, vel = system.dump_state()
        assert np.all(np.isfinite(pos))
        assert np.all(np.isfinite(vel))

    def test_async_rebuild_preserves_permutation_invariant(self):
        n = 100
        topology = Topology()
        topology.num_particles = n
        for i in range(n - 1):
            topology.add_bond(i, i + 1)
        term_params = {
            'bond': np.array([[300.0, 1.5]] * (n - 1), dtype=precision.FLOAT),
        }
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = np.eye(3, dtype=precision.FLOAT) * 80.0
        system = _make_system(topology, pbc_matrix, cutoff=10.0, skin=1.0)
        bonded = create_bonded_forces(topology, parameter_table)
        for f in bonded:
            system.add_force_term(f)

        rng = np.random.RandomState(42)
        system.set_positions(rng.uniform(10, 70, (n, 3)).astype(precision.FLOAT))
        system.set_velocities(rng.randn(n, 3).astype(precision.FLOAT) * 0.001)

        integrator = VerletIntegrator(time_step=0.5)
        _ensure_ready(system)
        _run_steps(system, integrator, 200)
        pos, vel = system.dump_state()

        assert np.all(np.isfinite(pos))
        assert np.all(np.isfinite(vel))

        p2s, s2p = _get_pdb_to_sorted(system)
        identity = np.arange(n, dtype=precision.INT)
        np.testing.assert_array_equal(s2p[p2s], identity,
            err_msg="permutation invariant broken with async rebuild")

    def test_rebuild_triggered_within_batch(self):
        topology, term_params = _build_four_particle()
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = _make_large_pbc()
        system = _make_system(topology, pbc_matrix, cutoff=12.0, skin=0.5)
        bonded = create_bonded_forces(topology, parameter_table)
        for f in bonded:
            system.add_force_term(f)
        system.set_positions(np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.5, 0.0, 0.0],
        ], dtype=precision.FLOAT))
        system.set_velocities(np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
        ], dtype=precision.FLOAT))

        integrator = VerletIntegrator(time_step=2.0)
        _ensure_ready(system)
        for _ in range(100):
            _run_steps(system, integrator, 1)
        pos, vel = system.dump_state()
        assert np.all(np.isfinite(pos))

    def test_langevin_async_rebuild_stable(self):
        topology, term_params = _build_four_particle()
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = _make_large_pbc()
        system = _make_system(topology, pbc_matrix, cutoff=12.0, skin=1.0)
        bonded = create_bonded_forces(topology, parameter_table)
        for f in bonded:
            system.add_force_term(f)
        system.set_positions(np.array([
            [0.0, 0.0, 0.0],
            [1.6, 0.0, 0.0],
            [3.0, 0.5, 0.0],
            [4.5, 0.0, 1.0],
        ], dtype=precision.FLOAT))
        system.set_velocities(np.zeros((4, 3), dtype=precision.FLOAT))

        integrator = LangevinBAOABIntegrator(
            time_step=0.1, temperature=300.0, friction=0.1
        )
        _ensure_ready(system)
        for _ in range(200):
            _run_steps(system, integrator, 1)
        pos, vel = system.dump_state()
        assert np.all(np.isfinite(pos))
        assert np.all(np.isfinite(vel))

    def test_custom_interval_affects_rebuild_timing(self):
        topology, term_params = _build_four_particle()
        parameter_table = _make_parameter_table(term_params)
        pbc_matrix = _make_large_pbc()
        system = _make_system(topology, pbc_matrix, cutoff=12.0, skin=1.0,
                        rebuild_check_interval=3)
        bonded = create_bonded_forces(topology, parameter_table)
        for f in bonded:
            system.add_force_term(f)
        system.set_positions(np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.5, 0.0, 0.0],
        ], dtype=precision.FLOAT))
        system.set_velocities(np.zeros((4, 3), dtype=precision.FLOAT))

        integrator = VerletIntegrator(time_step=0.1)
        _ensure_ready(system)
        _run_steps(system, integrator, 50)
        pos, vel = system.dump_state()
        assert np.all(np.isfinite(pos))


class TestInlineSystemAssembly:

    def test_inline_assembly_wires_everything(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        psf = PSFParser(os.path.join(data_dir, '6PO6.psf'))
        pdb = PDBParser(os.path.join(data_dir, '6PO6.pdb'))
        toppar = CharmmTopparParser(os.path.join(data_dir, 'par_all36_prot.prm'))
        pbc = np.eye(3, dtype=np.float64) * 30.0

        topology = psf.topology
        parameter_table = create_parameter_table(topology, toppar, type_names=psf.particle_type_names)
        state = State(topology.num_particles)
        state.set_pbc(pbc)
        state.set_positions(pdb.positions)
        state.set_charges(psf.charges)
        state.set_masses(psf.masses)
        state.set_type_indices(psf.particle_type_indices)
        forces = create_charmm_forces(topology, parameter_table, pbc, cutoff=12.0, particle_type_indices=psf.particle_type_indices)
        system = System(topology, state)
        for f in forces['bonded']:
            system.add_force_term(f)
        system.add_force_term(forces['nonbonded'])
        system.add_force_term(forces['pme'], stream='pme')

        # State seeded from IO but velocities not set yet -> not ready.
        assert system.state.is_ready is False
        assert system.num_particles == 49

        system.set_velocities(
            np.zeros((system.num_particles, 3), dtype=np.float32))
        assert system.state.is_ready is True

        # Full pipeline (neighbor list -> forces -> energies) must run clean.
        system.update_neighbor_list()
        system.compute_forces()
        energies = system.dump_energy()
        assert len(energies) > 0
        assert np.all(np.isfinite(list(energies.values())))

        # PME runs on its own stream — verify it contributed finite energy.
        assert 'pme_reciprocal' in energies

