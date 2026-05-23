import numpy as np
from mdpy import env
from mdpy.core.topology import Builder
from mdpy.forcefield.parameters import ParameterTable
from mdpy.force.bonded_force import BondedForce
from mdpy.force.nonbonded_force import NonbondedForce
from mdpy.force.expressions.lennard_jones import lennard_jones
from mdpy.force.expressions.coulomb import coulomb
from mdpy.system import System


class CharmmForcefield:

    def __init__(self, psf_path, pdb_path, parameter_paths, cutoff=12.0):
        from mdpy.io.psf_parser import PSFParser
        from mdpy.io.pdb_parser import PDBParser
        from mdpy.io.charmm_toppar_parser import CharmmTopparParser

        self._psf = PSFParser(psf_path)
        self._pdb = PDBParser(pdb_path)
        if isinstance(parameter_paths, str):
            parameter_paths = [parameter_paths]
        self._toppar = CharmmTopparParser(*parameter_paths)
        self._cutoff = cutoff
        self._term_params = {}

    def create_topology(self):
        psf = self._psf
        toppar = self._toppar
        parameters = toppar.parameters

        type_name_to_index = {}
        type_names_sorted = sorted(set(psf.particle_types))
        for index, type_name in enumerate(type_names_sorted):
            type_name_to_index[type_name] = index

        masses = psf._masses.copy()
        charges = psf._charges.copy()
        particle_types = np.array(
            [type_name_to_index[t] for t in psf.particle_types],
            dtype=env.NUMPY_INT,
        )
        molecule_ids = np.array(psf.molecule_ids, dtype=env.NUMPY_INT)

        builder = Builder()
        builder.set_particles(
            masses=masses,
            charges=charges,
            particle_types=particle_types,
            molecule_ids=molecule_ids,
            particle_names=psf.particle_names,
            type_names=psf.particle_types,
            chain_ids=psf.chain_ids,
            molecule_types=psf.molecule_types,
        )

        _resolve_bonds(builder, psf, parameters)
        _resolve_angles(builder, psf, parameters)
        _resolve_dihedrals(builder, psf, parameters)
        _resolve_impropers(builder, psf, parameters)

        builder.build_exclusion_map(scale_14=1.0)
        topology, self._term_params = builder.build()
        return topology

    def create_parameter_table(self):
        toppar = self._toppar
        parameters = toppar.parameters
        psf = self._psf

        type_name_to_index = {}
        type_names_sorted = sorted(set(psf.particle_types))
        for index, type_name in enumerate(type_names_sorted):
            type_name_to_index[type_name] = index

        table = ParameterTable()
        num_types = len(type_names_sorted)

        sigma_array = np.zeros(num_types, dtype=env.NUMPY_FLOAT)
        epsilon_array = np.zeros(num_types, dtype=env.NUMPY_FLOAT)
        sigma_14_array = np.zeros(num_types, dtype=env.NUMPY_FLOAT)
        epsilon_14_array = np.zeros(num_types, dtype=env.NUMPY_FLOAT)

        for type_name, type_index in type_name_to_index.items():
            nonbonded = parameters['nonbonded'].get(type_name)
            if nonbonded is not None:
                epsilon_array[type_index] = nonbonded[0]
                sigma_array[type_index] = nonbonded[1]
                if len(nonbonded) == 4:
                    epsilon_14_array[type_index] = nonbonded[2]
                    sigma_14_array[type_index] = nonbonded[3]
                else:
                    epsilon_14_array[type_index] = nonbonded[0]
                    sigma_14_array[type_index] = nonbonded[1]

        table.add_per_type('sigma', sigma_array)
        table.add_per_type('epsilon', epsilon_array)
        table.add_per_type('sigma_14', sigma_14_array)
        table.add_per_type('epsilon_14', epsilon_14_array)
        table.add_per_atom('charge', psf._charges.copy())
        table.add_per_atom('charge_14', psf._charges.copy())

        for term_name, params in self._term_params.items():
            table.add_per_term(term_name, params)

        return table

    def create_system(self, pbc_matrix=None):
        topology = self.create_topology()
        parameter_table = self.create_parameter_table()

        if pbc_matrix is None:
            pbc_matrix = self._pdb.pbc_matrix
        if pbc_matrix is None:
            pbc_matrix = np.eye(3, dtype=env.NUMPY_FLOAT) * 100.0

        system = System(topology, pbc_matrix, cutoff=self._cutoff)

        bonded = BondedForce.charmm(topology, parameter_table)
        system.add_force_term(bonded)

        lj_expression = lennard_jones + coulomb
        nonbonded = NonbondedForce(lj_expression)
        nonbonded.bind(topology, parameter_table, self._cutoff)
        system.add_force_term(nonbonded)

        system.particles.positions[:] = self._pdb.positions
        system.gpu.upload_positions(system.particles)
        system.gpu.upload_velocities(system.particles)

        return system


def _resolve_bonds(builder, psf, parameters):
    bond_parameters = parameters.get('bond', {})
    for bond in psf._bonds:
        type_i = psf.particle_types[bond[0]]
        type_j = psf.particle_types[bond[1]]
        key_forward = '%s-%s' % (type_i, type_j)
        key_reverse = '%s-%s' % (type_j, type_i)
        params = bond_parameters.get(key_forward) or bond_parameters.get(key_reverse)
        if params is None:
            continue
        builder.add_bond(bond[0], bond[1], params[0], params[1])


def _resolve_angles(builder, psf, parameters):
    angle_parameters = parameters.get('angle', {})
    for angle in psf._angles:
        type_i = psf.particle_types[angle[0]]
        type_j = psf.particle_types[angle[1]]
        type_k = psf.particle_types[angle[2]]
        key_forward = '%s-%s-%s' % (type_i, type_j, type_k)
        key_reverse = '%s-%s-%s' % (type_k, type_j, type_i)
        params = angle_parameters.get(key_forward) or angle_parameters.get(key_reverse)
        if params is None:
            continue
        builder.add_angle(
            angle[0], angle[1], angle[2],
            params[0], params[1], params[2], params[3],
        )


def _resolve_dihedrals(builder, psf, parameters):
    dihedral_parameters = parameters.get('dihedral', {})
    for dihedral in psf._dihedrals:
        type_i = psf.particle_types[dihedral[0]]
        type_j = psf.particle_types[dihedral[1]]
        type_k = psf.particle_types[dihedral[2]]
        type_l = psf.particle_types[dihedral[3]]
        key_forward = '%s-%s-%s-%s' % (type_i, type_j, type_k, type_l)
        key_reverse = '%s-%s-%s-%s' % (type_l, type_k, type_j, type_i)
        term_list = dihedral_parameters.get(key_forward) or dihedral_parameters.get(key_reverse)
        if term_list is None:
            continue
        for term in term_list:
            builder.add_dihedral(
                dihedral[0], dihedral[1], dihedral[2], dihedral[3],
                term[0], term[1], term[2],
            )


def _resolve_impropers(builder, psf, parameters):
    improper_parameters = parameters.get('improper', {})
    for improper in psf._impropers:
        type_i = psf.particle_types[improper[0]]
        type_j = psf.particle_types[improper[1]]
        type_k = psf.particle_types[improper[2]]
        type_l = psf.particle_types[improper[3]]
        key_forward = '%s-%s-%s-%s' % (type_i, type_j, type_k, type_l)
        key_reverse = '%s-%s-%s-%s' % (type_l, type_k, type_j, type_i)
        params = improper_parameters.get(key_forward) or improper_parameters.get(key_reverse)
        if params is None:
            continue
        builder.add_improper(
            improper[0], improper[1], improper[2], improper[3],
            params[0], params[1],
        )
