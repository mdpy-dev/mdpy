#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : psf_file.py
created time : 2021/10/05
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np
from MDAnalysis.topology.PSFParser import PSFParser
from ..core import Particle, Topology
from ..unit import *

class PSFFile:
    def __init__(self, file_path: str) -> None:
        self._parser = PSFParser(file_path).parse()
        
        self._num_particles = self._parser.n_atoms
        self._particle_ids = list(self._parser.ids.values)
        # The definition of type in PSFParser corresponding to name in Particle class
        self._particle_types = list(self._parser.names.values)
        self._particle_names = list(self._parser.types.values)
        self._matrix_ids = list(np.linspace(0, self._num_particles-1, self._num_particles, dtype=np.int))
        molecule_ids, molecule_types = self._parser.resids.values, self._parser.resnames.values
        chain_ids = self._parser.segids.values
        self._molecule_ids, self._molecule_types = [], []
        self._chain_ids = []
        for i in range(self._parser.n_atoms):
            resid = self._parser.tt.atoms2residues(i)
            segid = self._parser.tt.atoms2segments(i)
            self._molecule_ids.append(molecule_ids[resid])
            self._molecule_types.append(molecule_types[resid])
            self._chain_ids.append(chain_ids[segid])

        self._masses = list(Quantity(self._parser.masses.values, dalton).convert_to(default_mass_unit).value)
        self._charges = list(Quantity(self._parser.charges.values, e).convert_to(default_charge_unit).value)
        self._bonds = [list(i) for i in self._parser.bonds.values]
        self._num_bonds = len(self._bonds)
        self._angles = [list(i) for i in self._parser.angles.values]
        self._num_angles = len(self._angles)
        self._dihedrals = [list(i) for i in self._parser.dihedrals.values]
        self._num_dihedrals = len(self._dihedrals)
        self._impropers = [list(i) for i in self._parser.impropers.values]
        self._num_impropers = len(self._impropers)

    def create_topology(self):
        topology = Topology()
        particles = []
        for i in range(self._num_particles):
            particles.append(
                Particle(
                    particle_id=self._particle_ids[i], 
                    particle_type=self._particle_types[i],
                    particle_name=self._particle_names[i], 
                    matrix_id=self._matrix_ids[i],
                    molecule_id=self._molecule_ids[i], 
                    molecule_type=self._molecule_types[i], 
                    chain_id=self._chain_ids[i],
                    mass=self._masses[i], charge=self._charges[i]
                )
            )
        topology.add_particles(particles)
        [topology.add_bond([i, j]) for i, j in self._bonds]
        [topology.add_angle([i, j, k]) for i, j, k in self._angles]
        [topology.add_dihedral([i, j, k, l]) for i, j, k, l in self._dihedrals]
        [topology.add_improper([i, j, k, l]) for i, j, k, l in self._impropers]
        return topology

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
    def num_bonds(self):
        return self._num_bonds

    @property
    def num_angles(self):
        return self._num_angles

    @property
    def num_dihedrals(self):
        return self._num_dihedrals

    @property
    def num_impropers(self):
        return self._num_impropers

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