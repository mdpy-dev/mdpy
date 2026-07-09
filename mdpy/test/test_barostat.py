"""Tests for mdpy.barostat — Berendsen, Monte Carlo, and Langevin piston barostats."""

import numpy as np
import pytest

from mdpy import precision
from mdpy.core.topology import Topology
from mdpy.core.state import State
from mdpy.core.parameter_set import ParameterSet
from mdpy.force.bonded_force import BondedForce
from mdpy.force.factories.charmm import create_bonded_forces
from mdpy.force.nonbonded_transpiler import nonbonded_expression
from mdpy.force.markers import param
from mdpy.force.expressions.geometry import distance
from mdpy.force.nonbonded_force import NonbondedForce
from mdpy.system import System
from mdpy.integrator.verlet import VerletIntegrator
from mdpy.barostat.berendsen import BerendsenBarostat
from mdpy.barostat.monte_carlo import MonteCarloBarostat
from mdpy.barostat.langevin_piston import LangevinPistonBarostat
from mdpy.unit import Quantity, bar, default_pressure_unit

_PRESSURE_1BAR = float(Quantity(1, bar).convert_to(default_pressure_unit).value)


def _make_single_bond_system():
    """Two-particle system with a harmonic bond in a box."""
    topology = Topology()
    topology.num_particles = 2
    topology.add_bond(0, 1)

    n = 2
    state = State(n)
    state.set_particle_masses(np.ones(n, dtype=precision.FLOAT) * 12.0)
    state.set_particle_charges(np.zeros(n, dtype=precision.FLOAT))
    state.set_particle_type_indices(np.zeros(n, dtype=precision.INT))
    system = System(topology, state)
    system.set_pbc(np.eye(3, dtype=precision.FLOAT) * 30.0)

    pt = ParameterSet()
    pt.add_term_parameter("bond", np.array([[100.0, 1.5]], dtype=precision.FLOAT))
    bonded_forces = create_bonded_forces(topology, pt)
    for f in bonded_forces:
        system.add_force_term(f)

    pos = np.array([[15.0, 15.0, 15.0], [16.5, 15.0, 15.0]], dtype=precision.FLOAT)
    vel = np.zeros((2, 3), dtype=precision.FLOAT)
    system.set_positions(pos)
    system.set_velocities(vel)

    system._cutoff = 12.0
    return system


def _run_npt_steps(
    system, integrator, barostat, n, temperature, time_step, sync_interval=10
):
    for i in range(n):
        system.update_neighbor_list(sync_interval=sync_interval)
        system.compute_forces(compute_energy=True)
        barostat.apply(system, temperature, time_step)
        integrator.step(system)


class TestBerendsenBarostat:

    def test_initialization(self):
        b = BerendsenBarostat(
            target_pressure=_PRESSURE_1BAR, pressure_coupling_time=1000.0
        )
        assert b.target_pressure == pytest.approx(_PRESSURE_1BAR, rel=1e-6)
        assert b.pressure_coupling_time == 1000.0

    def test_apply_scales_box(self):
        system = _make_single_bond_system()
        b = BerendsenBarostat(
            target_pressure=_PRESSURE_1BAR, pressure_coupling_time=1000.0
        )

        old_box_x = system.state.box_x

        system.compute_forces(compute_energy=True)
        b.apply(system, temperature=300.0, time_step=1.0)

        new_box_x = system.state.box_x
        assert new_box_x > 0
        assert isinstance(new_box_x, (float, np.floating))

    def test_apply_scales_positions_isotropically(self):
        system = _make_single_bond_system()
        b = BerendsenBarostat(
            target_pressure=_PRESSURE_1BAR, pressure_coupling_time=10.0
        )

        pos_before = system.state.download_positions()
        dist_before = np.linalg.norm(pos_before[0] - pos_before[1])

        system.compute_forces(compute_energy=True)
        b.apply(system, temperature=300.0, time_step=1.0)

        pos_after = system.state.download_positions()
        dist_after = np.linalg.norm(pos_after[0] - pos_after[1])

        box_ratio = (
            system.state.box_x / 30.0
            + system.state.box_y / 30.0
            + system.state.box_z / 30.0
        ) / 3.0
        expected_ratio = dist_before * box_ratio
        assert dist_after == pytest.approx(expected_ratio, rel=1e-3)

    def test_apply_scales_velocities(self):
        system = _make_single_bond_system()
        vel = np.ones((2, 3), dtype=precision.FLOAT) * 0.1
        system.set_velocities(vel)

        b = BerendsenBarostat(
            target_pressure=_PRESSURE_1BAR, pressure_coupling_time=0.01
        )

        system.compute_forces(compute_energy=True)
        b.apply(system, temperature=300.0, time_step=1.0)

        vel_after = system.state.download_velocities()
        vel_diff = np.abs(vel_after - 0.1).max()
        assert vel_diff > 1e-9, f"Velocities unchanged, max change: {vel_diff}"

    def test_no_zero_division_with_tiny_box(self):
        system = _make_single_bond_system()
        system.state._box_x = 1e-10
        system.state._box_y = 1e-10
        system.state._box_z = 1e-10

        b = BerendsenBarostat(
            target_pressure=_PRESSURE_1BAR, pressure_coupling_time=1000.0
        )

        system.compute_forces(compute_energy=True)
        try:
            b.apply(system, temperature=300.0, time_step=1.0)
        except Exception as e:
            pytest.fail(f"Barostat failed with tiny box: {e}")

    def test_npt_run_does_not_explode(self):
        system = _make_single_bond_system()
        integrator = VerletIntegrator(1.0)
        barostat = BerendsenBarostat(
            target_pressure=_PRESSURE_1BAR, pressure_coupling_time=100.0
        )

        _run_npt_steps(
            system, integrator, barostat, n=50, temperature=300.0, time_step=1.0
        )

        pos = system.state.download_positions()
        assert np.all(np.isfinite(pos))
        assert system.state.box_x > 0
        assert system.state.box_y > 0
        assert system.state.box_z > 0


class TestMonteCarloBarostat:

    def test_initialization(self):
        b = MonteCarloBarostat(target_pressure=6e-9, temperature=300.0, frequency=25)
        assert b.target_pressure == 6e-9
        assert b.frequency == 25
        assert b.acceptance_rate == 0.0

    def test_does_nothing_on_non_frequency_steps(self):
        system = _make_single_bond_system()
        b = MonteCarloBarostat(target_pressure=6e-9, temperature=300.0, frequency=10)

        old_box_x = system.state.box_x
        old_box_y = system.state.box_y
        old_box_z = system.state.box_z

        system.compute_forces(compute_energy=True)
        b._step_counter = 0
        b.apply(system, temperature=300.0, time_step=1.0)

        assert system.state.box_x == old_box_x
        assert system.state.box_y == old_box_y
        assert system.state.box_z == old_box_z

    def test_step_counter_increments(self):
        b = MonteCarloBarostat(target_pressure=6e-9, temperature=300.0, frequency=25)
        system = _make_single_bond_system()
        system.compute_forces(compute_energy=True)

        assert b._step_counter == 0
        b.apply(system, temperature=300.0, time_step=1.0)
        assert b._step_counter == 1
        b.apply(system, temperature=300.0, time_step=1.0)
        assert b._step_counter == 2

    def test_npt_run_does_not_explode(self):
        system = _make_single_bond_system()
        integrator = VerletIntegrator(1.0)
        barostat = MonteCarloBarostat(
            target_pressure=6e-9, temperature=300.0, frequency=200
        )

        _run_npt_steps(
            system, integrator, barostat, n=50, temperature=300.0, time_step=1.0
        )

        pos = system.state.download_positions()
        assert np.all(np.isfinite(pos))
        assert system.state.box_x > 0
        assert 0.0 <= barostat.acceptance_rate <= 1.0


class TestLangevinPistonBarostat:

    def test_initialization(self):
        b = LangevinPistonBarostat(
            target_pressure=6e-9, temperature=300.0, piston_friction=0.1
        )
        assert b.target_pressure == 6e-9
        assert b.piston_friction == 0.1
        assert b._piston_mass is None

    def test_apply_scales_box(self):
        system = _make_single_bond_system()
        b = LangevinPistonBarostat(
            target_pressure=6e-9, temperature=300.0, piston_friction=0.01
        )

        old_box_x = system.state.box_x
        system.compute_forces(compute_energy=True)
        b.apply(system, temperature=300.0, time_step=1.0)

        new_box_x = system.state.box_x
        assert new_box_x > 0
        assert isinstance(new_box_x, (float, np.floating))

    def test_npt_run_does_not_explode(self):
        system = _make_single_bond_system()
        integrator = VerletIntegrator(1.0)
        barostat = LangevinPistonBarostat(
            target_pressure=6e-9, temperature=300.0, piston_friction=0.1
        )

        _run_npt_steps(
            system, integrator, barostat, n=100, temperature=300.0, time_step=1.0
        )

        pos = system.state.download_positions()
        assert np.all(np.isfinite(pos))
        assert system.state.box_x > 0
