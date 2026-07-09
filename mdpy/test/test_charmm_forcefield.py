import os
import numpy as np
import pytest
from mdpy import precision
from mdpy.io.psf_parser import PSFParser
from mdpy.io.pdb_parser import PDBParser
from mdpy.io.charmm_toppar_parser import CharmmTopparParser
from mdpy.core.parameter_set import ParameterSet
from mdpy.core.state import State
from mdpy.system import System
from mdpy.force.bonded_force import BondedForce
from mdpy.force.factories.charmm import create_bonded_forces
from mdpy.force.nonbonded_force import NonbondedForce
from mdpy.force.expressions.lennard_jones import lennard_jones
from mdpy.force.expressions.coulomb import coulomb
from mdpy.integrator.verlet import VerletIntegrator

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def _run_steps(system, integrator, n):
    for i in range(n):
        system.update_neighbor_list(sync_interval=n)
        system.compute_forces()
        integrator.step(system)


class TestTopology:

    def test_topology_particle_count(self):
        psf = PSFParser(os.path.join(DATA_DIR, '6PO6.psf'))
        topology = psf.topology
        assert topology.num_particles == 49

    def test_topology_bond_count(self):
        psf = PSFParser(os.path.join(DATA_DIR, '6PO6.psf'))
        topology = psf.topology
        assert topology.num_bonds == 49

    def test_topology_angle_count(self):
        psf = PSFParser(os.path.join(DATA_DIR, '6PO6.psf'))
        topology = psf.topology
        assert topology.num_angles > 0

    def test_topology_dihedral_count(self):
        psf = PSFParser(os.path.join(DATA_DIR, '6PO6.psf'))
        topology = psf.topology
        assert topology.num_dihedrals > 0

    def test_topology_masses_charges_arrays(self):
        psf = PSFParser(os.path.join(DATA_DIR, '6PO6.psf'))
        topology = psf.topology
        assert psf.particle_masses.dtype == precision.FLOAT
        assert psf.particle_charges.dtype == precision.FLOAT
        assert psf.particle_masses.shape == (49,)
        assert psf.particle_charges.shape == (49,)
        assert np.all(psf.particle_masses > 0)

    def test_topology_type_names(self):
        psf = PSFParser(os.path.join(DATA_DIR, '6PO6.psf'))
        topology = psf.topology
        assert len(psf.particle_type_names) == 49
        assert len(psf.particle_names) == 49

    def test_exclusion_map_built(self):
        psf = PSFParser(os.path.join(DATA_DIR, '6PO6.psf'))
        topology = psf.topology
        offset, neighbors = topology.exclusion_csr
        assert offset.shape[0] == topology.num_particles + 1
        assert int(neighbors.shape[0]) > 0


class TestParameterSet:

    def test_parameter_set_has_sigma_epsilon(self):
        psf = PSFParser(os.path.join(DATA_DIR, '6PO6.psf'))
        toppar = CharmmTopparParser(os.path.join(DATA_DIR, 'par_all36_prot.prm'))
        topology = psf.topology
        parameter_set = toppar.resolve_parameter_set(topology, psf.particle_type_names)
        assert 'sigma' in parameter_set.type_parameters
        assert 'epsilon' in parameter_set.type_parameters

    def test_parameter_set_values_positive(self):
        psf = PSFParser(os.path.join(DATA_DIR, '6PO6.psf'))
        toppar = CharmmTopparParser(os.path.join(DATA_DIR, 'par_all36_prot.prm'))
        topology = psf.topology
        parameter_set = toppar.resolve_parameter_set(topology, psf.particle_type_names)
        sigma = parameter_set.type_parameters['sigma']
        epsilon = parameter_set.type_parameters['epsilon']
        assert np.all(sigma > 0)
        assert np.all(epsilon > 0)

    def test_charge_not_in_parameter_set(self):
        psf = PSFParser(os.path.join(DATA_DIR, '6PO6.psf'))
        toppar = CharmmTopparParser(os.path.join(DATA_DIR, 'par_all36_prot.prm'))
        topology = psf.topology
        parameter_set = toppar.resolve_parameter_set(topology, psf.particle_type_names)
        assert 'charge' not in parameter_set.particle_parameters
        assert 'charge_14' not in parameter_set.particle_parameters
        assert psf.particle_charges.shape == (49,)

    def test_sigma_conversion_factor(self):
        from mdpy.io.charmm_toppar_parser import RMIN_TO_SIGMA_FACTOR
        expected_factor = float(2**(-1/6))
        assert abs(float(RMIN_TO_SIGMA_FACTOR) - expected_factor) < 1e-6


class TestSystem:

    def test_create_system(self):
        psf = PSFParser(os.path.join(DATA_DIR, '6PO6.psf'))
        pdb = PDBParser(os.path.join(DATA_DIR, '6PO6.pdb'))
        toppar = CharmmTopparParser(os.path.join(DATA_DIR, 'par_all36_prot.prm'))
        topology = psf.topology
        parameter_set = toppar.resolve_parameter_set(topology, psf.particle_type_names)
        pbc_matrix = np.eye(3, dtype=precision.FLOAT) * 100.0

        state = State(topology.num_particles)
        state.set_masses(psf.particle_masses)
        state.set_charges(psf.particle_charges)
        state.set_type_indices(parameter_set.particle_type_indices)
        system = System(topology, state)

        system.set_pbc(pbc_matrix)
        for f in create_bonded_forces(topology, parameter_set):
            system.add_force_term(f)
        nb = NonbondedForce(lennard_jones + coulomb, cutoff=12.0)
        lj_pair = parameter_set.type_pair_parameters['lj_pair']
        nb.set_pair_parameter('sigma', lj_pair[0::2].astype(precision.FLOAT))
        nb.set_pair_parameter('epsilon', lj_pair[1::2].astype(precision.FLOAT))
        system.add_force_term(nb)
        pbc_inv = np.linalg.inv(pbc_matrix)
        raw = pdb.positions.astype(np.float64)
        frac = raw @ pbc_inv
        frac -= np.floor(frac)
        positions = (frac @ pbc_matrix).astype(np.float32)
        system.set_positions(positions)

        assert system.topology.num_particles == 49
        assert len(system.force_terms) == 5
        assert np.all(np.isfinite(positions))

    def test_system_compute_forces(self):
        psf = PSFParser(os.path.join(DATA_DIR, '6PO6.psf'))
        pdb = PDBParser(os.path.join(DATA_DIR, '6PO6.pdb'))
        toppar = CharmmTopparParser(os.path.join(DATA_DIR, 'par_all36_prot.prm'))
        topology = psf.topology
        parameter_set = toppar.resolve_parameter_set(topology, psf.particle_type_names)
        pbc_matrix = np.eye(3, dtype=precision.FLOAT) * 100.0

        state = State(topology.num_particles)
        state.set_masses(psf.particle_masses)
        state.set_charges(psf.particle_charges)
        state.set_type_indices(parameter_set.particle_type_indices)
        system = System(topology, state)

        system.set_pbc(pbc_matrix)
        for f in create_bonded_forces(topology, parameter_set):
            system.add_force_term(f)
        nb = NonbondedForce(lennard_jones + coulomb, cutoff=12.0)
        lj_pair = parameter_set.type_pair_parameters['lj_pair']
        nb.set_pair_parameter('sigma', lj_pair[0::2].astype(precision.FLOAT))
        nb.set_pair_parameter('epsilon', lj_pair[1::2].astype(precision.FLOAT))
        system.add_force_term(nb)
        pbc_inv = np.linalg.inv(pbc_matrix)
        raw = pdb.positions.astype(np.float64)
        frac = raw @ pbc_inv
        frac -= np.floor(frac)
        positions = (frac @ pbc_matrix).astype(np.float32)
        system.set_positions(positions)
        velocities = np.zeros_like(positions)
        system.set_velocities(velocities)

        system.compute_forces()
        assert all(np.isfinite(v) for v in system.dump_energy().values())
        assert sum(system.dump_energy().values()) != 0.0
        forces = system.dump_forces()
        assert np.all(np.isfinite(forces))

    def test_system_100_steps(self):
        psf = PSFParser(os.path.join(DATA_DIR, '6PO6.psf'))
        pdb = PDBParser(os.path.join(DATA_DIR, '6PO6.pdb'))
        toppar = CharmmTopparParser(os.path.join(DATA_DIR, 'par_all36_prot.prm'))
        topology = psf.topology
        parameter_set = toppar.resolve_parameter_set(topology, psf.particle_type_names)
        pbc_matrix = np.eye(3, dtype=precision.FLOAT) * 100.0

        state = State(topology.num_particles)
        state.set_masses(psf.particle_masses)
        state.set_charges(psf.particle_charges)
        state.set_type_indices(parameter_set.particle_type_indices)
        system = System(topology, state)

        system.set_pbc(pbc_matrix)
        for f in create_bonded_forces(topology, parameter_set):
            system.add_force_term(f)
        nb = NonbondedForce(lennard_jones + coulomb, cutoff=12.0)
        lj_pair = parameter_set.type_pair_parameters['lj_pair']
        nb.set_pair_parameter('sigma', lj_pair[0::2].astype(precision.FLOAT))
        nb.set_pair_parameter('epsilon', lj_pair[1::2].astype(precision.FLOAT))
        system.add_force_term(nb)
        pbc_inv = np.linalg.inv(pbc_matrix)
        raw = pdb.positions.astype(np.float64)
        frac = raw @ pbc_inv
        frac -= np.floor(frac)
        positions = (frac @ pbc_matrix).astype(np.float32)
        system.set_positions(positions)
        velocities = np.zeros_like(positions)
        system.set_velocities(velocities)
        integrator = VerletIntegrator(time_step=0.5)
        energies = []
        for step in range(100):
            _run_steps(system, integrator, 1)
            energies.append(sum(system.dump_energy().values()))
        assert all(np.isfinite(energy) for energy in energies)

    def test_bond_energy_finite(self):
        psf = PSFParser(os.path.join(DATA_DIR, '6PO6.psf'))
        pdb = PDBParser(os.path.join(DATA_DIR, '6PO6.pdb'))
        toppar = CharmmTopparParser(os.path.join(DATA_DIR, 'par_all36_prot.prm'))
        topology = psf.topology
        parameter_set = toppar.resolve_parameter_set(topology, psf.particle_type_names)
        pbc_matrix = np.eye(3, dtype=precision.FLOAT) * 100.0

        state = State(topology.num_particles)
        state.set_masses(psf.particle_masses)
        state.set_charges(psf.particle_charges)
        state.set_type_indices(parameter_set.particle_type_indices)
        system = System(topology, state)

        system.set_pbc(pbc_matrix)
        for f in create_bonded_forces(topology, parameter_set):
            system.add_force_term(f)
        nb = NonbondedForce(lennard_jones + coulomb, cutoff=12.0)
        lj_pair = parameter_set.type_pair_parameters['lj_pair']
        nb.set_pair_parameter('sigma', lj_pair[0::2].astype(precision.FLOAT))
        nb.set_pair_parameter('epsilon', lj_pair[1::2].astype(precision.FLOAT))
        system.add_force_term(nb)
        pbc_inv = np.linalg.inv(pbc_matrix)
        raw = pdb.positions.astype(np.float64)
        frac = raw @ pbc_inv
        frac -= np.floor(frac)
        positions = (frac @ pbc_matrix).astype(np.float32)
        system.set_positions(positions)
        velocities = np.zeros_like(positions)
        system.set_velocities(velocities)

        system.compute_forces()
        energies = system.dump_energy()
        assert 'bond' in energies
        assert np.isfinite(energies['bond'])

    def test_dihedral_multi_term(self):
        psf = PSFParser(os.path.join(DATA_DIR, '6PO6.psf'))
        toppar = CharmmTopparParser(os.path.join(DATA_DIR, 'par_all36_prot.prm'))
        topology = psf.topology
        assert topology.num_dihedrals >= psf.num_dihedrals


class TestMissingParameters:

    def test_missing_bond_skipped(self):
        psf = PSFParser(os.path.join(DATA_DIR, '6PO6.psf'))
        toppar = CharmmTopparParser(os.path.join(DATA_DIR, 'par_all36_prot.prm'))
        topology = psf.topology
        parameter_set = toppar.resolve_parameter_set(topology, psf.particle_type_names)
        bond_params = parameter_set.get_term_parameter('bond')
        assert topology.num_bonds > 0
        assert bond_params.shape[0] == topology.num_bonds
