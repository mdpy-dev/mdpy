import os
import numpy as np
import pytest

from mdpy import env
from mdpy.io.psf_parser import PSFParser
from mdpy.io.charmm_toppar_parser import CharmmTopparParser, create_parameter_table
from mdpy.forcefield.charmm_forces import (
    create_charmm_forces,
    charmm_bond_v2,
    charmm_angle_v2,
    charmm_dihedral_v2,
    charmm_improper_v2,
)
from mdpy.force.bonded_force import BondedForceV2
from mdpy.force.nonbonded_force import NonbondedForceV2

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture(scope='module')
def topology_and_table():
    psf = PSFParser(os.path.join(DATA_DIR, '6PO6.psf'))
    toppar = CharmmTopparParser(os.path.join(DATA_DIR, 'par_all36_prot.prm'))
    topology = psf.topology
    parameter_table = create_parameter_table(topology, toppar)
    return topology, parameter_table


class TestFactoryCreation:

    def test_returns_dict_with_all_keys(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, topology.num_particles)
        expected_keys = {'bond', 'angle', 'dihedral', 'improper', 'nb14', 'nonbonded'}
        assert set(forces.keys()) == expected_keys

    def test_bond_is_bonded_force_v2(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, topology.num_particles)
        assert isinstance(forces['bond'], BondedForceV2)

    def test_nonbonded_is_nonbonded_force_v2(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, topology.num_particles)
        assert isinstance(forces['nonbonded'], NonbondedForceV2)

    def test_unique_names(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, topology.num_particles)
        names = [f.name for f in forces.values()]
        assert len(names) == len(set(names)), f"Duplicate names: {names}"


class TestBondForce:

    def test_bond_count_matches_topology(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, topology.num_particles)
        assert forces['bond']._count == topology.num_bonds

    def test_bond_parameters_extracted(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, topology.num_particles)
        bond_params = table.get_term_parameter('bond')
        assert forces['bond']._count > 0
        assert forces['bond']._parameters_per_term == 2


class TestAngleForce:

    def test_angle_count_matches_topology(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, topology.num_particles)
        assert forces['angle']._count == topology.num_angles

    def test_angle_parameters_extracted(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, topology.num_particles)
        assert forces['angle']._parameters_per_term == 4


class TestDihedralForce:

    def test_dihedral_count_matches_topology(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, topology.num_particles)
        assert forces['dihedral']._count == topology.num_dihedrals

    def test_dihedral_parameters_extracted(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, topology.num_particles)
        assert forces['dihedral']._parameters_per_term == 3


class TestImproperForce:

    def test_improper_count_matches_topology(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, topology.num_particles)
        assert forces['improper']._count == topology.num_impropers


class TestNb14Force:

    def test_nb14_is_bonded_force_v2(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, topology.num_particles)
        assert isinstance(forces['nb14'], BondedForceV2)

    def test_nb14_has_charges(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, topology.num_particles)
        assert 'charge' in forces['nb14']._per_particle_gpu

    def test_nb14_count_positive(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, topology.num_particles)
        assert forces['nb14']._count > 0

    def test_nb14_charge_scale_is_one(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, topology.num_particles)
        assert forces['nb14']._parameters_per_term == 3


class TestNonbondedForce:

    def test_nonbonded_has_charge(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, topology.num_particles)
        assert 'charge' in forces['nonbonded']._per_particle_data

    def test_nonbonded_has_sigma_epsilon(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, topology.num_particles)
        assert 'sigma' in forces['nonbonded']._pair_param_data
        assert 'epsilon' in forces['nonbonded']._pair_param_data

    def test_nonbonded_sigma_matrix_shape(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, topology.num_particles)
        n_types = len(table.type_parameters['sigma'])
        sigma = forces['nonbonded']._pair_param_data['sigma']
        assert sigma.shape[0] == n_types * n_types


class TestEnergyComputation:

    def test_bond_energy_finite(self, topology_and_table):
        import cupy as cp

        topology, table = topology_and_table
        from mdpy.io.pdb_parser import PDBParser
        pdb = PDBParser(os.path.join(DATA_DIR, '6PO6.pdb'))
        pbc_matrix = np.eye(3, dtype=env.NUMPY_FLOAT) * 100.0

        forces = create_charmm_forces(topology, table, topology.num_particles)
        bond_force = forces['bond']

        positions = pdb.positions.astype(env.NUMPY_FLOAT)

        class MockContext:
            def __init__(self, positions, pbc_matrix):
                pos = positions.astype(np.float32)
                self.d_positions_x = cp.asarray(pos[:, 0])
                self.d_positions_y = cp.asarray(pos[:, 1])
                self.d_positions_z = cp.asarray(pos[:, 2])
                N = positions.shape[0]
                self.d_forces_x = cp.zeros(N, dtype=np.float32)
                self.d_forces_y = cp.zeros(N, dtype=np.float32)
                self.d_forces_z = cp.zeros(N, dtype=np.float32)
                self.d_energy = cp.zeros(1, dtype=np.float32)
                pbc_inv = np.linalg.inv(pbc_matrix)
                self.d_pbc_inv = cp.asarray(
                    np.ascontiguousarray(pbc_inv, dtype=np.float32).ravel()
                )
                self.d_pbc_matrix = cp.asarray(
                    np.ascontiguousarray(pbc_matrix, dtype=np.float32).ravel()
                )

        context = MockContext(positions, pbc_matrix)
        bond_force.compute(context)
        energy = float(context.d_energy[0])
        assert np.isfinite(energy)
        assert energy != 0.0
