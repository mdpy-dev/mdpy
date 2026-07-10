"""Tests for mdpy.barostat — Monte Carlo barostat and supporting infrastructure."""

import numpy as np
import cupy as cp
import pytest

from mdpy import precision
from mdpy.core.topology import Topology
from mdpy.core.state import State
from mdpy.core.parameter_set import ParameterSet
from mdpy.force.bonded_force import BondedForce
from mdpy.force.bonded_transpiler import bonded_expression
from mdpy.force.nonbonded_transpiler import nonbonded_expression
from mdpy.force.markers import param
from mdpy.force.expressions.geometry import distance
from mdpy.force.nonbonded_force import NonbondedForce
from mdpy.force.pme_reciprocal_force import PMEReciprocalForce, precompute_bk_factors
from mdpy.system import System


@pytest.fixture(autouse=True)
def _sync_gpu():
    yield
    cp.cuda.Stream.null.synchronize()


def _make_pme(cutoff=12.0, fourier_spacing=1.0):
    """Create a minimal PMEReciprocalForce for testing."""
    topology = Topology()
    topology.num_particles = 2
    topology.add_bond(0, 1)
    pt = ParameterSet()
    pbc = np.eye(3, dtype=np.float64) * 50.0
    pme = PMEReciprocalForce(cutoff, fourier_spacing=fourier_spacing)
    pme.initialize_grid(topology, pt, pbc_matrix=pbc)
    return pme


class TestPMEUpdateBox:
    def test_bk_factors_change_after_update_box(self):
        pme = _make_pme()
        original_bk = pme._d_bk_factors.copy()

        pme.update_box(60.0, 60.0, 60.0, block_list=None)

        new_bk = pme._d_bk_factors
        assert not cp.allclose(original_bk, new_bk), \
            "bk_factors must change when box dimensions change"

    def test_bk_factors_match_manual_recompute(self):
        pme = _make_pme()
        new_box_x, new_box_y, new_box_z = 55.0, 55.0, 55.0

        pme.update_box(new_box_x, new_box_y, new_box_z, block_list=None)
        expected_bk = precompute_bk_factors(
            pme.alpha, pme.grid_x, pme.grid_y, pme.grid_z,
            pme.order, new_box_x, new_box_y, new_box_z)

        assert cp.allclose(pme._d_bk_factors, cp.asarray(expected_bk))

    def test_subgrid_not_invalidated_when_num_cells_unchanged(self):
        """When block_list.num_cells hasn't changed, subgrid cache stays valid."""
        pme = _make_pme()

        # Build a fake block_list-like object with stable num_cells
        class FakeBlockList:
            num_cells_x = 5
            num_cells_y = 5
            num_cells_z = 5

        bl = FakeBlockList()

        # Simulate subgrid initialization by running through the lazy init path
        pme._subgrid_dx = -(-pme.grid_x // 5) + 2 * pme.order
        pme._subgrid_dy = -(-pme.grid_y // 5) + 2 * pme.order
        pme._subgrid_dz = -(-pme.grid_z // 5) + 2 * pme.order
        pme._subgrid_initialized = True

        pme.update_box(66.0, 66.0, 66.0, block_list=bl)

        assert pme._subgrid_initialized is True, \
            "subgrid should NOT be invalidated when num_cells hasn't changed"

    def test_subgrid_invalidated_when_num_cells_changes(self):
        """When block_list.num_cells changes, subgrid cache must be invalidated."""
        pme = _make_pme()

        class FakeBlockList:
            num_cells_x = 3  # Different from original (was 5 in setup)
            num_cells_y = 3
            num_cells_z = 3

        bl = FakeBlockList()

        # Set up initial subgrid as if computed with num_cells=5
        pme._subgrid_dx = -(-pme.grid_x // 5) + 2 * pme.order
        pme._subgrid_dy = -(-pme.grid_y // 5) + 2 * pme.order
        pme._subgrid_dz = -(-pme.grid_z // 5) + 2 * pme.order
        pme._subgrid_initialized = True

        # update_box with block_list that has num_cells=3 — should invalidate
        pme.update_box(40.0, 40.0, 40.0, block_list=bl)

        assert pme._subgrid_initialized is False, \
            "subgrid MUST be invalidated when num_cells changed"


@bonded_expression(body=2)
def _harmonic_bond(p1, p2, k=param, r0=param):
    r = distance(p1, p2)
    dr = r - r0
    return k * dr * dr


def _make_minimal_system(num_particles=4, box_size=50.0, cutoff=12.0):
    """Create a system with bonded + nonbonded forces for testing."""
    topology = Topology()
    topology.num_particles = num_particles
    for i in range(num_particles - 1):
        topology.add_bond(i, i + 1)

    n = num_particles
    state = State(n)
    state.set_particle_masses(np.full(n, 12.0, dtype=precision.FLOAT))
    state.set_particle_charges(np.zeros(n, dtype=precision.FLOAT))
    state.set_particle_type_indices(np.zeros(n, dtype=precision.INT))

    system = System(topology, state)
    system.set_pbc(np.eye(3, dtype=precision.FLOAT) * box_size)

    # Simple harmonic bond
    bond_force = BondedForce(_harmonic_bond)
    for i in range(num_particles - 1):
        bond_force.add([i, i + 1], k=np.float32(100.0), r0=np.float32(1.5))
    system.add_force_term(bond_force)

    # Simple repulsive nonbonded
    @nonbonded_expression
    def repulsive(p1, p2, sigma=param, epsilon=param):
        r = distance(p1, p2)
        return 4.0 * epsilon * (sigma / r) ** 12

    nb = NonbondedForce(repulsive, cutoff=cutoff)
    num_types = 1
    lj_pair = np.array([2.0, 0.1] * (num_types * num_types), dtype=precision.FLOAT)
    nb.set_pair_parameter('sigma', lj_pair[0::2].astype(precision.FLOAT))
    nb.set_pair_parameter('epsilon', lj_pair[1::2].astype(precision.FLOAT))
    nb.name = 'nonbonded'
    system.add_force_term(nb)

    # Place particles spread across the box
    positions = np.zeros((n, 3), dtype=precision.FLOAT)
    for i in range(n):
        positions[i, 0] = (i + 1) * box_size / (n + 1)
    system.set_positions(positions)
    system.set_velocities(np.zeros((n, 3), dtype=precision.FLOAT))
    system.update_neighbor_list(force_rebuild=True)

    return system


class TestSystemResizeBox:
    def test_resize_box_updates_state_pbc(self):
        system = _make_minimal_system()
        new_pbc = np.eye(3, dtype=precision.FLOAT) * 60.0
        system.resize_box(new_pbc)
        assert abs(system.state.box_x - 60.0) < 1e-5
        assert abs(system.state.box_y - 60.0) < 1e-5
        assert abs(system.state.box_z - 60.0) < 1e-5

    def test_resize_box_triggers_block_list_rebuild(self):
        system = _make_minimal_system()
        old_num_cells = system.block_list.num_cells_total
        system.resize_box(np.eye(3, dtype=precision.FLOAT) * 100.0)
        new_num_cells = system.block_list.num_cells_total
        # Larger box → more cells
        assert new_num_cells > old_num_cells

    def test_resize_box_calls_pme_update_box(self):
        """If PME is present, resize_box should call its update_box."""
        system = _make_minimal_system()
        # Add a mock force term with update_box
        call_log = []

        class MockBoxDependent:
            name = 'mock'
            _cutoff = None

            def compute(self, state, block_list=None, compute_energy=True):
                pass

            def update_box(self, box_x, box_y, box_z, block_list=None):
                call_log.append((box_x, box_y, box_z))

        system.add_force_term(MockBoxDependent())
        system.resize_box(np.eye(3, dtype=precision.FLOAT) * 60.0)
        assert len(call_log) == 1
        assert call_log[0] == (60.0, 60.0, 60.0)

    def test_compute_total_energy_returns_float(self):
        system = _make_minimal_system()
        energy = system.compute_total_energy()
        assert isinstance(energy, float)

    def test_compute_total_energy_positive_for_repulsive(self):
        system = _make_minimal_system()
        energy = system.compute_total_energy()
        assert energy > 0.0, "Repulsive system should have positive energy"


class TestSystemBarostatIntegration:
    def test_add_barostat_and_apply(self):
        """add_barostat stores barostat; apply_barostats calls apply(system)."""
        system = _make_minimal_system()
        call_count = [0]

        class MockBarostat:
            def apply(self, system):
                call_count[0] += 1

        system.add_barostat(MockBarostat())
        assert len(system.barostats) == 1
        system.apply_barostats()
        assert call_count[0] == 1
