#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : psf_file.py
created time : 2021/10/05
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import numpy as np
from mdpy import env
from mdpy.core.topology import Topology
from mdpy.error import *


class PSFParser:
    def __init__(self, file_path: str) -> None:
        if not file_path.endswith('.psf'):
            raise FileFormatError('The file should end with .psf suffix')
        self._file_path = file_path
        with open(file_path, 'r') as f:
            lines = f.readlines()
        self._parse(lines)

    def _parse(self, lines):
        idx = 0
        idx = self._skip_header(lines, idx)
        idx = self._parse_atoms(lines, idx)
        idx = self._parse_section(lines, idx, '!NBOND:', 2, '_bonds')
        idx = self._parse_section(lines, idx, '!NTHETA:', 3, '_angles')
        idx = self._parse_section(lines, idx, '!NPHI:', 4, '_dihedrals')
        idx = self._parse_section(lines, idx, '!NIMPHI:', 4, '_impropers')
        self._build_type_index()
        self._topology = self._create_topology()

    def _skip_header(self, lines, idx):
        while idx < len(lines):
            stripped = lines[idx].strip()
            if stripped.startswith('!NTITLE') or stripped.endswith('!NTITLE'):
                parts = stripped.split()
                n_title = int(parts[0])
                return idx + 1 + n_title
            idx += 1
        raise ParserPoorlyDefinedError('PSF file missing !NTITLE section')

    def _parse_atoms(self, lines, idx):
        while idx < len(lines):
            stripped = lines[idx].strip()
            if not stripped:
                idx += 1
                continue
            parts = stripped.split()
            if '!NATOM' in stripped:
                n_atoms = int(parts[0])
                idx += 1
                break
            idx += 1
        else:
            raise ParserPoorlyDefinedError('PSF file missing !NATOM section')

        self._num_particles = n_atoms
        self._particle_ids = []
        self._type_names = []
        self._particle_names = []
        self._molecule_ids = []
        self._molecule_types = []
        self._chain_ids = []
        self._masses = np.zeros(n_atoms, dtype=env.NUMPY_FLOAT)
        self._charges = np.zeros(n_atoms, dtype=env.NUMPY_FLOAT)

        for i in range(n_atoms):
            parts = lines[idx + i].split()
            self._particle_ids.append(int(parts[0]))
            self._chain_ids.append(parts[1])
            self._molecule_ids.append(int(parts[2]))
            self._molecule_types.append(parts[3])
            self._particle_names.append(parts[4])
            self._type_names.append(parts[5])
            self._charges[i] = float(parts[6])
            self._masses[i] = float(parts[7])

        return idx + n_atoms

    def _parse_section(self, lines, idx, marker, width, attr_name):
        while idx < len(lines):
            stripped = lines[idx].strip()
            if not stripped:
                idx += 1
                continue
            if marker in stripped:
                parts = stripped.split()
                count = int(parts[0])
                idx += 1
                break
            idx += 1
        else:
            setattr(self, attr_name, [])
            return idx

        all_ints = []
        while len(all_ints) < count * width and idx < len(lines):
            stripped = lines[idx].strip()
            if not stripped:
                idx += 1
                continue
            if stripped.startswith('!'):
                break
            all_ints.extend(int(x) for x in stripped.split())
            idx += 1

        interactions = []
        for i in range(0, count * width, width):
            interactions.append([all_ints[i + j] - 1 for j in range(width)])
        setattr(self, attr_name, interactions)
        return idx

    def _build_type_index(self):
        unique_types = sorted(set(self._type_names))
        self._type_name_to_index = {name: idx for idx, name in enumerate(unique_types)}
        self._particle_type_indices = np.array(
            [self._type_name_to_index[t] for t in self._type_names], dtype=env.NUMPY_INT
        )
        self._unique_type_names = unique_types

    def _create_topology(self):
        topology = Topology()
        topology.num_particles = self._num_particles
        for i, j in self._bonds:
            topology.add_bond(i, j)
        for i, j, k in self._angles:
            topology.add_angle(i, j, k)
        for i, j, k, l in self._dihedrals:
            topology.add_dihedral(i, j, k, l)
        for i, j, k, l in self._impropers:
            topology.add_improper(i, j, k, l)
        return topology

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
    def particle_type_names(self):
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
    def unique_type_names(self):
        return self._unique_type_names

    @property
    def particle_type_indices(self):
        return self._particle_type_indices

    @property
    def charges(self):
        return self._charges

    @property
    def masses(self):
        return self._masses

    @property
    def topology(self) -> Topology:
        return self._topology
