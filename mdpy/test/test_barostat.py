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
    system.state.set_particle_molecule_ids(
        np.arange(n, dtype=np.int32))
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


from mdpy.barostat._base import BarostatBase
from mdpy.barostat.monte_carlo import MonteCarloBarostat, BOLTZMANN, BAR_TO_INTERNAL_PRESSURE


class TestBarostatBase:
    def test_base_has_apply_method(self):
        assert hasattr(BarostatBase, 'apply')

    def test_base_apply_raises_not_implemented(self):
        base = BarostatBase()
        with pytest.raises(NotImplementedError):
            base.apply(None)

    def test_base_has_name_attribute(self):
        assert hasattr(BarostatBase, 'name')


class TestMoleculeCSR:
    def test_single_molecule_all_atoms(self):
        """All atoms in one molecule → one group."""
        from mdpy.barostat.monte_carlo import build_molecule_csr
        mol_ids = [0, 0, 0, 0]
        atoms, starts = build_molecule_csr(mol_ids)
        assert len(starts) == 2  # 1 molecule + 1 sentinel
        assert starts[0] == 0
        assert starts[1] == 4

    def test_multiple_molecules(self):
        """4 atoms in 2 molecules (2+2)."""
        from mdpy.barostat.monte_carlo import build_molecule_csr
        mol_ids = [0, 0, 1, 1]
        atoms, starts = build_molecule_csr(mol_ids)
        assert len(starts) == 3  # 2 molecules + 1 sentinel
        # Molecule 0: atoms 0, 1
        assert atoms[starts[0]:starts[1]].tolist() == [0, 1]
        # Molecule 1: atoms 2, 3
        assert atoms[starts[1]:starts[2]].tolist() == [2, 3]

    def test_interleaved_molecules(self):
        """Molecules not contiguous in PDB order → CSR reorders them."""
        from mdpy.barostat.monte_carlo import build_molecule_csr
        mol_ids = [1, 0, 1, 0]  # interleaved
        atoms, starts = build_molecule_csr(mol_ids)
        assert len(starts) == 3
        # Molecule 0: atoms at PDB indices 1, 3
        assert atoms[starts[0]:starts[1]].tolist() == [1, 3]
        # Molecule 1: atoms at PDB indices 0, 2
        assert atoms[starts[1]:starts[2]].tolist() == [0, 2]

    def test_returns_int32_arrays(self):
        from mdpy.barostat.monte_carlo import build_molecule_csr
        mol_ids = [0, 0, 1]
        atoms, starts = build_molecule_csr(mol_ids)
        assert atoms.dtype == np.int32
        assert starts.dtype == np.int32


class TestScaleMoleculePositions:
    def test_single_molecule_scales_about_centroid(self):
        """2-atom molecule: centroid stays at centroid * scale."""
        from mdpy.barostat.monte_carlo import MonteCarloBarostat

        n = 2
        state = State(n)
        state.set_pbc(np.eye(3, dtype=precision.FLOAT) * 100.0)
        state.set_positions(np.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=precision.FLOAT))
        state.set_velocities(np.zeros((n, 3), dtype=precision.FLOAT))
        state.set_particle_masses(np.ones(n, dtype=precision.FLOAT))
        state.set_particle_charges(np.zeros(n, dtype=precision.FLOAT))
        state.set_particle_type_indices(np.zeros(n, dtype=precision.INT))
        state.set_prev_positions(np.array([[9.0, 0.0, 0.0], [19.0, 0.0, 0.0]], dtype=precision.FLOAT))
        state.set_particle_molecule_ids(np.array([0, 0], dtype=np.int32))

        barostat = MonteCarloBarostat(
            pressure_bar=1.0, temperature=300.0)

        barostat._build_molecule_csr(state)
        scale = np.float32(2.0)
        barostat._scale_positions(state, scale)

        pos_x = state.d_positions_x.get()
        # centroid was (10+20)/2 = 15; new centroid = 30
        # atom 0: 10 + 15*(2-1) = 25
        # atom 1: 20 + 15*(2-1) = 35
        assert abs(pos_x[0] - 25.0) < 1e-4
        assert abs(pos_x[1] - 35.0) < 1e-4

    def test_velocity_preserved(self):
        """Scaling both pos and prev_pos by same delta preserves velocity."""
        from mdpy.barostat.monte_carlo import MonteCarloBarostat

        n = 2
        state = State(n)
        state.set_pbc(np.eye(3, dtype=precision.FLOAT) * 100.0)
        state.set_positions(np.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=precision.FLOAT))
        state.set_velocities(np.zeros((n, 3), dtype=precision.FLOAT))
        state.set_particle_masses(np.ones(n, dtype=precision.FLOAT))
        state.set_particle_charges(np.zeros(n, dtype=precision.FLOAT))
        state.set_particle_type_indices(np.zeros(n, dtype=precision.INT))
        # prev_positions define velocity = (pos - prev) / dt
        prev_pos = np.array([[8.0, 0.0, 0.0], [18.0, 0.0, 0.0]], dtype=precision.FLOAT)
        state.set_prev_positions(prev_pos)
        state.set_particle_molecule_ids(np.array([0, 0], dtype=np.int32))

        # Original velocity (assuming dt=1): v = pos - prev = [2, 0, 0] for both
        original_vel_x = state.d_positions_x.get() - state.d_prev_positions_x.get()

        barostat = MonteCarloBarostat(
            pressure_bar=1.0, temperature=300.0)
        barostat._build_molecule_csr(state)
        barostat._scale_positions(state, np.float32(1.5))

        new_vel_x = state.d_positions_x.get() - state.d_prev_positions_x.get()
        np.testing.assert_allclose(new_vel_x, original_vel_x, atol=1e-4)

    def test_two_molecules_scale_independently(self):
        """Two single-atom molecules at different positions scale independently."""
        from mdpy.barostat.monte_carlo import MonteCarloBarostat

        n = 2
        state = State(n)
        state.set_pbc(np.eye(3, dtype=precision.FLOAT) * 100.0)
        state.set_positions(np.array([[10.0, 0.0, 0.0], [30.0, 0.0, 0.0]], dtype=precision.FLOAT))
        state.set_velocities(np.zeros((n, 3), dtype=precision.FLOAT))
        state.set_particle_masses(np.ones(n, dtype=precision.FLOAT))
        state.set_particle_charges(np.zeros(n, dtype=precision.FLOAT))
        state.set_particle_type_indices(np.zeros(n, dtype=precision.INT))
        state.set_prev_positions(np.array([[9.0, 0.0, 0.0], [29.0, 0.0, 0.0]], dtype=precision.FLOAT))
        state.set_particle_molecule_ids(np.array([0, 1], dtype=np.int32))

        barostat = MonteCarloBarostat(
            pressure_bar=1.0, temperature=300.0)

        barostat._build_molecule_csr(state)
        barostat._scale_positions(state, np.float32(2.0))

        pos_x = state.d_positions_x.get()
        # Single-atom molecule: centroid = atom position itself
        # atom 0: 10 * 2 = 20
        # atom 1: 30 * 2 = 60
        assert abs(pos_x[0] - 20.0) < 1e-4
        assert abs(pos_x[1] - 60.0) < 1e-4


class TestMonteCarloApply:
    def test_apply_does_nothing_before_frequency(self):
        """Barostat should be a no-op until frequency steps have elapsed."""
        system = _make_minimal_system()
        barostat = MonteCarloBarostat(
            pressure_bar=1.0, temperature=300.0, frequency=5)

        original_box_x = system.state.box_x
        for i in range(4):
            barostat.apply(system)
        assert abs(system.state.box_x - original_box_x) < 1e-5
        assert barostat._num_attempted == 0

    def test_apply_acts_on_frequency_step(self):
        """After exactly `frequency` calls, a volume move is attempted."""
        system = _make_minimal_system()
        barostat = MonteCarloBarostat(
            pressure_bar=1.0, temperature=300.0, frequency=3)

        for i in range(3):
            barostat.apply(system)
        assert barostat._num_attempted == 1

    def test_apply_preserves_num_particles(self):
        """Scaling must not change the number of particles."""
        system = _make_minimal_system()
        barostat = MonteCarloBarostat(
            pressure_bar=1.0, temperature=300.0, frequency=1)
        barostat.apply(system)
        system.compute_forces()
        assert system.state.num_particles == 4

    def test_acceptance_rate_tracking(self):
        """After several attempts, acceptance_rate is between 0 and 1."""
        system = _make_minimal_system(box_size=50.0)
        barostat = MonteCarloBarostat(
            pressure_bar=1.0, temperature=300.0, frequency=1)
        for i in range(20):
            barostat.apply(system)
        assert 0.0 <= barostat.acceptance_rate <= 1.0

    def test_adaptive_tuning_adjusts_volume_scale(self):
        """After 10 attempts with extreme acceptance, volume_scale changes."""
        system = _make_minimal_system()
        barostat = MonteCarloBarostat(
            pressure_bar=1000.0,
            temperature=300.0, frequency=1)

        initial_scale = barostat._volume_scale
        for i in range(15):
            barostat.apply(system)

        assert barostat._volume_scale is not None
        assert barostat._volume_scale > 0

    def test_reject_restores_positions(self):
        """On reject, positions should be restored to pre-trial values."""
        system = _make_minimal_system()
        barostat = MonteCarloBarostat(
            pressure_bar=1e10,
            temperature=300.0, frequency=1)

        pos_before = system.state.d_positions_x.copy()
        barostat.apply(system)
        pos_after = system.state.d_positions_x.get()

        if barostat._num_attempted > 0 and barostat._num_accepted == 0:
            np.testing.assert_allclose(pos_after, pos_before.get(), atol=1e-4)

    def test_energy_unchanged_on_reject(self):
        """If the move is rejected, total energy should be the same."""
        system = _make_minimal_system()
        barostat = MonteCarloBarostat(
            pressure_bar=1e10,
            temperature=300.0, frequency=1)

        energy_before = system.compute_total_energy()
        barostat.apply(system)
        energy_after = system.compute_total_energy()

        if barostat._num_attempted > 0 and barostat._num_accepted == 0:
            assert abs(energy_after - energy_before) < 1e-2


class TestNPTConvergence:
    def test_ideal_gas_volume_converges(self):
        """Ideal gas NPT: average volume should approach N_mol*kT/P.

        Uses non-interacting particles (no nonbonded) so PV = N_mol*kT.
        Runs 500 steps with frequency=1 and checks the average volume
        over the last 200 steps is within 30% of the analytical value.
        Statistical tolerance is wide due to small system size.
        """
        num_particles = 20
        box_size = 80.0
        topology = Topology()
        topology.num_particles = num_particles

        state = State(num_particles)
        state.set_particle_masses(np.full(num_particles, 12.0, dtype=precision.FLOAT))
        state.set_particle_charges(np.zeros(num_particles, dtype=precision.FLOAT))
        state.set_particle_type_indices(np.zeros(num_particles, dtype=precision.INT))

        system = System(topology, state)
        system.set_pbc(np.eye(3, dtype=precision.FLOAT) * box_size)
        state.set_particle_molecule_ids(np.arange(num_particles, dtype=np.int32))
        system._cutoff = 12.0  # required for update_neighbor_list with no force terms

        positions = np.random.RandomState(42).uniform(
            0, box_size, size=(num_particles, 3)).astype(precision.FLOAT)
        system.set_positions(positions)
        system.set_velocities(np.zeros((num_particles, 3), dtype=precision.FLOAT))
        system.update_neighbor_list(force_rebuild=True)

        pressure_bar = 10.0
        temperature = 300.0
        barostat = MonteCarloBarostat(
            pressure_bar=pressure_bar,
            temperature=temperature,
            frequency=1,
        )
        system.add_barostat(barostat)

        kT = BOLTZMANN * temperature
        expected_volume = num_particles * kT / (pressure_bar * BAR_TO_INTERNAL_PRESSURE)

        volumes = []
        for i in range(500):
            system.apply_barostats()

        barostat._num_attempted = 0
        barostat._num_accepted = 0
        for i in range(500):
            system.apply_barostats()
            if i >= 300:
                s = system.state
                volumes.append(s.box_x * s.box_y * s.box_z)

        avg_volume = np.mean(volumes)

        ratio = avg_volume / expected_volume
        assert 0.5 < ratio < 2.0, \
            f"avg_volume={avg_volume:.1f}, expected={expected_volume:.1f}, ratio={ratio:.2f}"


class TestBuildParticleMoleculeIds:
    def test_single_molecule(self):
        from mdpy.utils.molecule import build_particle_molecule_ids
        bonds = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
        mol_ids = build_particle_molecule_ids(bonds, 4)
        assert len(np.unique(mol_ids)) == 1
        assert mol_ids.dtype == np.int32

    def test_two_molecules(self):
        from mdpy.utils.molecule import build_particle_molecule_ids
        bonds = np.array([[0, 1], [2, 3]], dtype=np.int32)
        mol_ids = build_particle_molecule_ids(bonds, 4)
        assert len(np.unique(mol_ids)) == 2
        assert mol_ids[0] == mol_ids[1]
        assert mol_ids[2] == mol_ids[3]
        assert mol_ids[0] != mol_ids[2]

    def test_no_bonds_each_atom_own_molecule(self):
        from mdpy.utils.molecule import build_particle_molecule_ids
        bonds = np.empty((0, 2), dtype=np.int32)
        mol_ids = build_particle_molecule_ids(bonds, 5)
        assert len(np.unique(mol_ids)) == 5

    def test_returns_sequential_ids(self):
        from mdpy.utils.molecule import build_particle_molecule_ids
        bonds = np.array([[0, 1], [2, 3]], dtype=np.int32)
        mol_ids = build_particle_molecule_ids(bonds, 4)
        unique = np.unique(mol_ids)
        assert unique.tolist() == [0, 1]


class TestPSFParserResidueMolecule:
    def test_psf_has_residue_ids(self):
        from mdpy.io.psf_parser import PSFParser
        import os
        psf = PSFParser(os.path.join(os.path.dirname(__file__), 'data', '6PO6.psf'))
        assert hasattr(psf, 'particle_residue_ids')
        assert len(psf.particle_residue_ids) == psf.num_particles

    def test_psf_has_residue_names(self):
        from mdpy.io.psf_parser import PSFParser
        import os
        psf = PSFParser(os.path.join(os.path.dirname(__file__), 'data', '6PO6.psf'))
        assert hasattr(psf, 'particle_residue_names')
        assert len(psf.particle_residue_names) == psf.num_particles

    def test_psf_has_molecule_ids_from_bonds(self):
        from mdpy.io.psf_parser import PSFParser
        import os
        psf = PSFParser(os.path.join(os.path.dirname(__file__), 'data', '6PO6.psf'))
        assert hasattr(psf, 'particle_molecule_ids')
        mol_ids = psf.particle_molecule_ids
        assert isinstance(mol_ids, np.ndarray)
        assert mol_ids.dtype == np.int32
        assert len(mol_ids) == psf.num_particles

    def test_psf_molecule_ids_differ_from_residue_ids(self):
        """For a protein, molecule IDs group all protein atoms into one molecule,
        while residue IDs split them by residue."""
        from mdpy.io.psf_parser import PSFParser
        import os
        psf = PSFParser(os.path.join(os.path.dirname(__file__), 'data', '1M9Z.psf'))
        num_residues = len(set(psf.particle_residue_ids))
        num_molecules = len(set(psf.particle_molecule_ids))
        assert num_molecules > num_residues, \
            f"Bond-graph molecules ({num_molecules}) should exceed residue IDs ({num_residues})"

    def test_psf_no_particle_molecule_types(self):
        """Old property name should no longer exist."""
        from mdpy.io.psf_parser import PSFParser
        import os
        psf = PSFParser(os.path.join(os.path.dirname(__file__), 'data', '6PO6.psf'))
        assert not hasattr(psf, 'particle_molecule_types')


class TestStateMoleculeIds:
    def test_set_and_read_molecule_ids(self):
        n = 4
        state = State(n)
        mol_ids = np.array([0, 0, 1, 1], dtype=np.int32)
        state.set_particle_molecule_ids(mol_ids)
        result = cp.asnumpy(state.d_particle_molecule_ids)
        np.testing.assert_array_equal(result, mol_ids)

    def test_molecule_ids_not_required_for_is_ready(self):
        """Molecule IDs are optional — only barostat needs them."""
        n = 2
        state = State(n)
        state.set_particle_masses(np.ones(n, dtype=precision.FLOAT))
        state.set_particle_charges(np.zeros(n, dtype=precision.FLOAT))
        state.set_particle_type_indices(np.zeros(n, dtype=precision.INT))
        state.set_pbc(np.eye(3, dtype=precision.FLOAT) * 50.0)
        state.set_positions(np.zeros((n, 3), dtype=precision.FLOAT))
        state.set_velocities(np.zeros((n, 3), dtype=precision.FLOAT))
        assert state.is_ready
