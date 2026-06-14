import os
import numpy as np
import pytest

from mdpy import env
from mdpy.io.psf_parser import PSFParser
from mdpy.io.charmm_toppar_parser import CharmmTopparParser, create_parameter_table
from mdpy.force.factories.charmm import (
    create_charmm_forces,
)
from mdpy.force.bonded_force import BondedForce
from mdpy.force.nonbonded_force import NonbondedForce

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
        forces = create_charmm_forces(topology, table, np.eye(3)*108.0)
        expected_keys = {'bonded', 'nonbonded', 'pme'}
        assert set(forces.keys()) == expected_keys

    def test_bonded_is_force_group(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, np.eye(3)*108.0)
        from mdpy.force.force_group import ForceGroup
        assert isinstance(forces['bonded'], ForceGroup)

    def test_nonbonded_is_nonbonded_force(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, np.eye(3)*108.0)
        from mdpy.force.nonbonded_force import NonbondedForce
        assert isinstance(forces['nonbonded'], NonbondedForce)

    def test_unique_names(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, np.eye(3)*108.0)
        names = [forces['bonded'].name, forces['nonbonded'].name, forces['pme'].name]
        assert len(names) == len(set(names)), f"Duplicate names: {names}"


class TestBondForce:

    def test_bond_count_matches_topology(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, np.eye(3)*108.0)
        bonded = forces['bonded']
        bond_force = [f for f in bonded._forces if f.name == 'bond'][0]
        assert bond_force._count == topology.num_bonds

    def test_bond_parameters_extracted(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, np.eye(3)*108.0)
        bonded = forces['bonded']
        bond_force = [f for f in bonded._forces if f.name == 'bond'][0]
        bond_params = table.get_term_parameter('bond')
        assert bond_force._count > 0
        assert bond_force._parameters_per_term == 2


class TestAngleForce:

    def test_angle_count_matches_topology(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, np.eye(3)*108.0)
        bonded = forces['bonded']
        angle_force = [f for f in bonded._forces if f.name == 'angle'][0]
        assert angle_force._count == topology.num_angles

    def test_angle_parameters_extracted(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, np.eye(3)*108.0)
        bonded = forces['bonded']
        angle_force = [f for f in bonded._forces if f.name == 'angle'][0]
        assert angle_force._parameters_per_term == 4


class TestDihedralForce:

    def test_dihedral_count_matches_topology(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, np.eye(3)*108.0)
        bonded = forces['bonded']
        dihedral_force = [f for f in bonded._forces if f.name == 'dihedral'][0]
        assert dihedral_force._count == topology.num_dihedrals

    def test_dihedral_parameters_extracted(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, np.eye(3)*108.0)
        bonded = forces['bonded']
        dihedral_force = [f for f in bonded._forces if f.name == 'dihedral'][0]
        assert dihedral_force._parameters_per_term == 3


class TestImproperForce:

    def test_improper_count_matches_topology(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, np.eye(3)*108.0)
        bonded = forces['bonded']
        improper_force = [f for f in bonded._forces if f.name == 'improper'][0]
        assert improper_force._count == topology.num_impropers


class TestNb14Force:

    def test_nb14_is_bonded_force(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, np.eye(3)*108.0)
        bonded = forces['bonded']
        nb14_force = [f for f in bonded._forces if f.name == 'nb14'][0]
        assert isinstance(nb14_force, BondedForce)

    def test_nb14_has_charges(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, np.eye(3)*108.0)
        bonded = forces['bonded']
        nb14_force = [f for f in bonded._forces if f.name == 'nb14'][0]
        assert 'charge' in nb14_force._per_particle_gpu

    def test_nb14_count_positive(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, np.eye(3)*108.0)
        bonded = forces['bonded']
        nb14_force = [f for f in bonded._forces if f.name == 'nb14'][0]
        assert nb14_force._count > 0

    def test_nb14_has_two_params(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, np.eye(3)*108.0)
        bonded = forces['bonded']
        nb14_force = [f for f in bonded._forces if f.name == 'nb14'][0]
        assert nb14_force._parameters_per_term == 2


class TestNonbondedForce:

    def test_nonbonded_has_charge(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, np.eye(3)*108.0)
        nb = forces['nonbonded']
        assert 'charge' in nb._prop_bases

    def test_nonbonded_has_sigma_epsilon(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, np.eye(3)*108.0)
        nb = forces['nonbonded']
        assert 'sigma' in nb._pair_param_data
        assert 'epsilon' in nb._pair_param_data

    def test_nonbonded_sigma_matrix_shape(self, topology_and_table):
        topology, table = topology_and_table
        forces = create_charmm_forces(topology, table, np.eye(3)*108.0)
        nb = forces['nonbonded']
        n_types = len(table.type_parameters['sigma'])
        sigma = nb._pair_param_data['sigma']
        assert sigma.shape[0] == n_types * n_types


class TestEnergyComputation:

    def test_bond_energy_finite(self, topology_and_table):
        import cupy as cp

        topology, table = topology_and_table
        from mdpy.io.pdb_parser import PDBParser
        pdb = PDBParser(os.path.join(DATA_DIR, '6PO6.pdb'))
        pbc_matrix = np.eye(3, dtype=env.NUMPY_FLOAT) * 100.0

        forces = create_charmm_forces(topology, table, np.eye(3)*108.0)
        bonded_group = forces['bonded']
        bond_force = [f for f in bonded_group._forces if f.name == 'bond'][0]

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
                self.d_charges = None
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
