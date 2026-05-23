import os
import numpy as np
import pytest
from mdpy import env
from mdpy.forcefield.charmm_forcefield import CharmmForcefield
from mdpy.forcefield.parameters import ParameterTable
from mdpy.integrator.verlet import VerletIntegrator

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


class TestCharmmForcefieldTopology:

    def test_topology_particle_count(self):
        forcefield = CharmmForcefield(
            os.path.join(DATA_DIR, '6PO6.psf'),
            os.path.join(DATA_DIR, '6PO6.pdb'),
            os.path.join(DATA_DIR, 'par_all36_prot.prm'),
        )
        topology = forcefield.create_topology()
        assert topology.num_particles == 49

    def test_topology_bond_count(self):
        forcefield = CharmmForcefield(
            os.path.join(DATA_DIR, '6PO6.psf'),
            os.path.join(DATA_DIR, '6PO6.pdb'),
            os.path.join(DATA_DIR, 'par_all36_prot.prm'),
        )
        topology = forcefield.create_topology()
        assert topology.num_bonds == 49

    def test_topology_angle_count(self):
        forcefield = CharmmForcefield(
            os.path.join(DATA_DIR, '6PO6.psf'),
            os.path.join(DATA_DIR, '6PO6.pdb'),
            os.path.join(DATA_DIR, 'par_all36_prot.prm'),
        )
        topology = forcefield.create_topology()
        assert topology.num_angles > 0

    def test_topology_dihedral_count(self):
        forcefield = CharmmForcefield(
            os.path.join(DATA_DIR, '6PO6.psf'),
            os.path.join(DATA_DIR, '6PO6.pdb'),
            os.path.join(DATA_DIR, 'par_all36_prot.prm'),
        )
        topology = forcefield.create_topology()
        assert topology.num_dihedrals > 0

    def test_topology_masses_charges_arrays(self):
        forcefield = CharmmForcefield(
            os.path.join(DATA_DIR, '6PO6.psf'),
            os.path.join(DATA_DIR, '6PO6.pdb'),
            os.path.join(DATA_DIR, 'par_all36_prot.prm'),
        )
        topology = forcefield.create_topology()
        assert topology.masses.dtype == env.NUMPY_FLOAT
        assert topology.charges.dtype == env.NUMPY_FLOAT
        assert topology.masses.shape == (49,)
        assert topology.charges.shape == (49,)
        assert np.all(topology.masses > 0)

    def test_topology_type_names(self):
        forcefield = CharmmForcefield(
            os.path.join(DATA_DIR, '6PO6.psf'),
            os.path.join(DATA_DIR, '6PO6.pdb'),
            os.path.join(DATA_DIR, 'par_all36_prot.prm'),
        )
        topology = forcefield.create_topology()
        assert len(topology.type_names) == 49
        assert len(topology.particle_names) == 49

    def test_exclusion_map_built(self):
        forcefield = CharmmForcefield(
            os.path.join(DATA_DIR, '6PO6.psf'),
            os.path.join(DATA_DIR, '6PO6.pdb'),
            os.path.join(DATA_DIR, 'par_all36_prot.prm'),
        )
        topology = forcefield.create_topology()
        assert topology.exclusion_offset.shape == (50,)
        assert len(topology.exclusion_neighbors) > 0


class TestCharmmForcefieldParameterTable:

    def test_parameter_table_has_sigma_epsilon(self):
        forcefield = CharmmForcefield(
            os.path.join(DATA_DIR, '6PO6.psf'),
            os.path.join(DATA_DIR, '6PO6.pdb'),
            os.path.join(DATA_DIR, 'par_all36_prot.prm'),
        )
        table = forcefield.create_parameter_table()
        assert 'sigma' in table.per_type
        assert 'epsilon' in table.per_type
        assert 'charge' in table.per_atom

    def test_parameter_table_values_positive(self):
        forcefield = CharmmForcefield(
            os.path.join(DATA_DIR, '6PO6.psf'),
            os.path.join(DATA_DIR, '6PO6.pdb'),
            os.path.join(DATA_DIR, 'par_all36_prot.prm'),
        )
        table = forcefield.create_parameter_table()
        sigma = table.per_type['sigma']
        epsilon = table.per_type['epsilon']
        assert np.all(sigma > 0)
        assert np.all(epsilon > 0)

    def test_parameter_table_charge_per_atom(self):
        forcefield = CharmmForcefield(
            os.path.join(DATA_DIR, '6PO6.psf'),
            os.path.join(DATA_DIR, '6PO6.pdb'),
            os.path.join(DATA_DIR, 'par_all36_prot.prm'),
        )
        table = forcefield.create_parameter_table()
        charges = table.per_atom['charge']
        assert charges.shape == (49,)

    def test_sigma_conversion_factor(self):
        forcefield = CharmmForcefield(
            os.path.join(DATA_DIR, '6PO6.psf'),
            os.path.join(DATA_DIR, '6PO6.pdb'),
            os.path.join(DATA_DIR, 'par_all36_prot.prm'),
        )
        table = forcefield.create_parameter_table()
        from mdpy.io.charmm_toppar_parser import RMIN_TO_SIGMA_FACTOR
        expected_factor = float(2**(-1/6))
        assert abs(float(RMIN_TO_SIGMA_FACTOR) - expected_factor) < 1e-6


class TestCharmmForcefieldSystem:

    def test_create_system(self):
        forcefield = CharmmForcefield(
            os.path.join(DATA_DIR, '6PO6.psf'),
            os.path.join(DATA_DIR, '6PO6.pdb'),
            os.path.join(DATA_DIR, 'par_all36_prot.prm'),
        )
        pbc_matrix = np.eye(3, dtype=env.NUMPY_FLOAT) * 100.0
        system = forcefield.create_system(pbc_matrix=pbc_matrix)
        assert system.topology.num_particles == 49
        assert len(system.force_terms) == 2
        assert np.all(np.isfinite(system.particles.positions))

    def test_system_compute_forces(self):
        forcefield = CharmmForcefield(
            os.path.join(DATA_DIR, '6PO6.psf'),
            os.path.join(DATA_DIR, '6PO6.pdb'),
            os.path.join(DATA_DIR, 'par_all36_prot.prm'),
        )
        pbc_matrix = np.eye(3, dtype=env.NUMPY_FLOAT) * 100.0
        system = forcefield.create_system(pbc_matrix=pbc_matrix)
        system.compute_forces()
        assert all(np.isfinite(v) for v in system.dump_energy().values())
        assert sum(system.dump_energy().values()) != 0.0
        assert np.all(np.isfinite(system.particles.forces))

    def test_system_100_steps(self):
        forcefield = CharmmForcefield(
            os.path.join(DATA_DIR, '6PO6.psf'),
            os.path.join(DATA_DIR, '6PO6.pdb'),
            os.path.join(DATA_DIR, 'par_all36_prot.prm'),
        )
        pbc_matrix = np.eye(3, dtype=env.NUMPY_FLOAT) * 100.0
        system = forcefield.create_system(pbc_matrix=pbc_matrix)
        integrator = VerletIntegrator(time_step=0.5)
        energies = []
        for step in range(100):
            system.step(integrator, number_steps=1)
            energies.append(sum(system.dump_energy().values()))
        assert all(np.isfinite(energy) for energy in energies)
        assert system.step_count == 100

    def test_bond_energy_finite(self):
        forcefield = CharmmForcefield(
            os.path.join(DATA_DIR, '6PO6.psf'),
            os.path.join(DATA_DIR, '6PO6.pdb'),
            os.path.join(DATA_DIR, 'par_all36_prot.prm'),
        )
        pbc_matrix = np.eye(3, dtype=env.NUMPY_FLOAT) * 100.0
        system = forcefield.create_system(pbc_matrix=pbc_matrix)
        system.compute_forces()
        energies = system.dump_energy()
        assert 'bonded' in energies
        assert np.isfinite(energies['bonded'])

    def test_dihedral_multi_term(self):
        forcefield = CharmmForcefield(
            os.path.join(DATA_DIR, '6PO6.psf'),
            os.path.join(DATA_DIR, '6PO6.pdb'),
            os.path.join(DATA_DIR, 'par_all36_prot.prm'),
        )
        topology = forcefield.create_topology()
        psf = forcefield._psf
        assert topology.num_dihedrals >= psf.num_dihedrals


class TestCharmmForcefieldMissingParameters:

    def test_missing_bond_skipped(self):
        forcefield = CharmmForcefield(
            os.path.join(DATA_DIR, '6PO6.psf'),
            os.path.join(DATA_DIR, '6PO6.pdb'),
            os.path.join(DATA_DIR, 'par_all36_prot.prm'),
        )
        topology = forcefield.create_topology()
        assert topology.num_bonds > 0
