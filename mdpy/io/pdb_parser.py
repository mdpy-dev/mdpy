#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : pdb_parser.py
created time : 2021/10/03
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import warnings
import numpy as np
import MDAnalysis as mda
from MDAnalysis.topology.guessers import guess_atom_type
from .. import env
from ..error import *

class PDBParser:
    def __init__(self, file_path) -> None:
        # Initial reader and parser setting
        if not file_path.endswith('.pdb'):
            raise FileFormatError('The file should end with .pdb suffix')
        self._file_path = file_path
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._reader = mda.coordinates.PDB.PDBReader(self._file_path)
            self._parser = mda.topology.PDBParser.PDBParser(self._file_path).parse()
        # Parse data
        self._num_particles = self._parser.n_atoms
        self._particle_ids = list(self._parser.ids.values)
        self._particle_types = list(map(guess_atom_type, self._parser.names.values))
        self._particle_names = list(self._parser.names.values)
        self._matrix_ids = list(np.linspace(0, self._num_particles-1, self._num_particles, dtype=env.NUMPY_INT))
        molecule_ids, molecule_types = self._parser.resids.values, self._parser.resnames.values
        self._molecule_ids, self._molecule_types = [], []
        for i in range(self._parser.n_atoms):
            resid = self._parser.tt.atoms2residues(i)
            self._molecule_ids.append(molecule_ids[resid])
            self._molecule_types.append(molecule_types[resid])
        self._chain_ids = list(self._parser.chainIDs.values)
        if self._reader.n_frames == 1:
            self._positions = self._reader.ts.positions.astype(env.NUMPY_FLOAT)
        else:
            self._positions = [ts.positions.astype(env.NUMPY_FLOAT) for ts in self._reader.trajectory]
            self._positions = np.stack(self._positions)
        self._pbc_matrix = self._reader.ts.triclinic_dimensions

    def get_matrix_id(self, particle_id):
        return self._particle_ids.index(particle_id)

    def get_particle_info(self, particle_id):
        matrix_id = self.get_matrix_id(particle_id)
        return {
            'particle_id': self._particle_ids[matrix_id],
            'particle_type': self._particle_types[matrix_id],
            'particle_name': self._particle_names[matrix_id],
            'molecule_id': self._molecule_ids[matrix_id],
            'molecule_type': self._molecule_types[matrix_id],
            'chain_id': self._chain_ids[matrix_id],
            'matrix_id': matrix_id,
            'position': self._positions[matrix_id, :]
        }

    @property
    def num_particles(self):
        return self._num_particles

    @property
    def particle_ids(self):
        return self._particle_ids

    @property
    def particle_types(self):
        return self._particle_types

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
    def positions(self) -> np.ndarray:
        return self._positions.copy()

    @property
    def pbc_matrix(self) -> np.ndarray:
        return self._pbc_matrix.copy()