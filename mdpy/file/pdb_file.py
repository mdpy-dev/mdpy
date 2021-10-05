#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : pdb_file.py
created time : 2021/10/03
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

from contextlib import redirect_stdout
from functools import partial
import numpy as np
import MDAnalysis as mda
from MDAnalysis.coordinates.PDB import PDBReader
from MDAnalysis.topology.PDBParser import PDBParser
from MDAnalysis.topology.guessers import guess_atom_type
from numpy.matrixlib.defmatrix import matrix

from mdpy.core.particle import Particle

class PDBFile:
    def __init__(self, file_path) -> None:
        # Initial reader and parser setting
        self._reader = PDBReader(file_path)
        self._parser = PDBParser(file_path).parse()
        # Parse data
        self._num_particles = self._parser.n_atoms
        self._particle_ids = list(self._parser.ids.values)
        self._particle_types = list(map(guess_atom_type, self._parser.names.values))
        self._particle_names = list(self._parser.names.values)
        self._matrix_ids = list(np.linspace(0, self._num_particles-1, self._num_particles, dtype=np.int))
        molecule_ids, molecule_types = self._parser.resids.values, self._parser.resnames.values
        self._molecule_ids, self._molecule_types = [], []
        for i in range(self._parser.n_atoms):
            resid = self._parser.tt.atoms2residues(i)
            self._molecule_ids.append(molecule_ids[resid])
            self._molecule_types.append(molecule_types[resid])
        self._chain_ids = list(self._parser.chainIDs.values)
        self._positions = self._reader.ts.positions

    def create_particles(self):
        particles = []
        for i in range(self._num_particles):
            particles.append(
                Particle(
                    self._particle_ids[i], self._particle_types[i], 
                    matrix_id=self._matrix_ids[i],
                    molecule_id=self._molecule_ids[i], 
                    molecule_type=self._molecule_types[i], 
                    chain_id=self._chain_ids[i]
                )
            )
        return particles

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
    def positions(self):
        return self._positions