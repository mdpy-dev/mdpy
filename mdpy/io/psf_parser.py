#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : psf_file.py
created time : 2021/10/05
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import warnings
import numpy as np
import MDAnalysis as mda
from mdpy import env
from mdpy.core.topology import Topology, Builder
from mdpy.error import *

class PSFParser:
    def __init__(self, file_path: str) -> None:
        if not file_path.endswith('.psf'):
            raise FileFormatError('The file should end with .psf suffix')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._parser = mda.topology.PSFParser.PSFParser(file_path).parse()
        self._num_particles = self._parser.n_atoms
        self._particle_ids = list(self._parser.ids.values)
        self._type_names = list(self._parser.types.values)
        self._particle_names = list(self._parser.names.values)

        molecule_ids = self._parser.resids.values
        molecule_types = self._parser.resnames.values
        chain_ids = self._parser.segids.values

        self._molecule_ids = []
        self._molecule_types = []
        self._chain_ids = []
        for i in range(self._num_particles):
            resid = self._parser.tt.atoms2residues(i)
            segid = self._parser.tt.atoms2segments(i)
            self._molecule_ids.append(int(molecule_ids[resid]))
            self._molecule_types.append(molecule_types[resid])
            self._chain_ids.append(chain_ids[segid])

        self._masses = np.array(self._parser.masses.values, dtype=env.NUMPY_FLOAT)
        self._charges = np.array(self._parser.charges.values, dtype=env.NUMPY_FLOAT)

        self._bonds = [list(i) for i in self._parser.bonds.values]
        self._angles = [list(i) for i in self._parser.angles.values]
        self._dihedrals = [list(i) for i in self._parser.dihedrals.values]
        self._impropers = [list(i) for i in self._parser.impropers.values]

        unique_types = sorted(set(self._type_names))
        self._type_name_to_index = {name: idx for idx, name in enumerate(unique_types)}
        self._particle_type_indices = np.array(
            [self._type_name_to_index[t] for t in self._type_names], dtype=env.NUMPY_INT
        )
        self._unique_type_names = unique_types

        self._topology = self._create_topology()

    def _create_topology(self):
        builder = Builder()
        builder.set_particles(
            masses=self._masses,
            charges=self._charges,
            particle_types=self._particle_type_indices,
            molecule_ids=np.array(self._molecule_ids, dtype=env.NUMPY_INT),
            particle_names=self._particle_names,
            type_names=self._type_names,
            chain_ids=self._chain_ids,
            molecule_types=self._molecule_types,
        )
        for i, j in self._bonds:
            builder.add_bond(i, j, 0.0, 0.0)
        for i, j, k in self._angles:
            builder.add_angle(i, j, k, 0.0, 0.0)
        for i, j, k, l in self._dihedrals:
            builder.add_dihedral(i, j, k, l, 0.0, 0.0, 0.0)
        for i, j, k, l in self._impropers:
            builder.add_improper(i, j, k, l, 0.0, 0.0)
        builder.build_exclusion_map()
        return builder.build()

    def get_matrix_id(self, particle_id):
        return self._particle_ids.index(particle_id)

    @property
    def num_particles(self):
        return self._num_particles

    @property
    def num_bonds(self):
        return len(self._bonds)

    @property
    def num_angles(self):
        return len(self._angles)

    @property
    def num_dihedrals(self):
        return len(self._dihedrals)

    @property
    def num_impropers(self):
        return len(self._impropers)

    @property
    def particle_ids(self):
        return self._particle_ids

    @property
    def particle_types(self):
        return self._type_names

    @property
    def particle_names(self):
        return self._particle_names

    @property
    def molecule_ids(self):
        return self._molecule_ids

    @property
    def molecule_types(self):
        return self._molecule_types

    @property
    def chain_ids(self):
        return self._chain_ids

    @property
    def type_names(self):
        return self._type_names

    @property
    def unique_type_names(self):
        return self._unique_type_names

    @property
    def particle_type_indices(self):
        return self._particle_type_indices

    @property
    def topology(self) -> Topology:
        return self._topology
