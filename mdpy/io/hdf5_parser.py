#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : hdf5_parser.py
created time : 2022/02/24
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import h5py
import numpy as np
from copy import copy
from mdpy import env, SPATIAL_DIM
from mdpy.core import Particle, Topology, Trajectory
from mdpy.error import *
from mdpy.io.hdf5_writer import NONE_LABEL

ROOT_KEYS = [
    'topology', 'positions', 'pbc_matrix'
]
TOPOLOGY_KEYS = [
    'particles', 'num_particles',
    'bonds', 'num_bonds',
    'angles', 'num_angles',
    'dihedrals', 'num_dihedrals',
    'impropers', 'num_impropers'
]
PARTICLES_KEYS = [
    'particle_id', 'particle_type', 'particle_name',
    'molecule_id', 'molecule_type',
    'chain_id', 'matrix_id',
    'mass', 'charge'
]

class HDF5Parser:
    def __init__(self, file_path: str, is_parse_all=True) -> None:
        # Initial reader and parser setting
        if not file_path.endswith('.hdf5'):
            raise FileFormatError('The file should end with .hdf5 suffix')
        self._file_path = file_path
        self._is_parse_all = is_parse_all
        self._file = h5py.File(self._file_path, 'r')
        self._check_hdf5_file()
        self._topology = self._parse_topology()
        self._pbc_matrix = self._parse_pbc_matrix()
        self._num_frames = self._file['positions/num_frames'][()]
        self._num_particles = self._topology.num_particles
        if self._is_parse_all:
            self._positions = self._parse_positions()
            self._trajectory = Trajectory(self._topology)
            self._trajectory.append(positions=self._positions)
            self._trajectory.set_pbc_matrix(self._pbc_matrix)
        self._file.close()

    def _check_hdf5_file(self):
        is_hdf5_file_poor_defined = False
        # /
        keys = list(self._file.keys())
        if not self._check_keys(keys, ROOT_KEYS):
            is_hdf5_file_poor_defined = True
        # /topology
        keys = list(self._file['topology'].keys())
        if not self._check_keys(keys, TOPOLOGY_KEYS):
            is_hdf5_file_poor_defined = True
        # /topology/particles
        keys = list(self._file['topology/particles'].keys())
        if not self._check_keys(keys, PARTICLES_KEYS):
            is_hdf5_file_poor_defined = True
        # /positions
        keys = list(self._file['positions'].keys())
        if not 'num_frames' in keys:
            is_hdf5_file_poor_defined = True
        # Output
        if is_hdf5_file_poor_defined:
            raise HDF5FilePoorDefinedError(
                '%s does not meet the hierarchy of mdpy created HDF5, ' \
                'please check mdpy.io.HDF5_FILE_HIERARCHY' %self._file_path
            )

    @staticmethod
    def _check_keys(exist_keys, target_keys):
        for key in target_keys:
            if not key in exist_keys:
                return False
        return True

    @staticmethod
    def _check_topology_none(val):
        if val == type(val)(NONE_LABEL):
            return None
        else:
            return val

    def _parse_topology(self):
        topology = Topology()
        # Particle
        particle_id = self._file['topology/particles/particle_id'][()].astype(env.NUMPY_INT)
        particle_type = self._file['topology/particles/particle_type'][()]
        particle_name = self._file['topology/particles/particle_name'][()]
        matrix_id = self._file['topology/particles/matrix_id'][()].astype(env.NUMPY_INT)
        molecule_id = self._file['topology/particles/molecule_id'][()].astype(env.NUMPY_INT)
        molecule_type = self._file['topology/particles/molecule_type'][()]
        chain_id = self._file['topology/particles/chain_id'][()]
        mass = self._file['topology/particles/mass'][()].astype(env.NUMPY_FLOAT)
        charge = self._file['topology/particles/charge'][()].astype(env.NUMPY_FLOAT)
        num_particles = self._file['topology/num_particles'][()]
        particles = []
        for index in range(num_particles):
            particles.append(Particle(
                particle_id=self._check_topology_none(particle_id[index]),
                particle_type=self._check_topology_none(bytes.decode(particle_type[index])),
                particle_name=self._check_topology_none(bytes.decode(particle_name[index])),
                matrix_id=self._check_topology_none(matrix_id[index]),
                molecule_id=self._check_topology_none(molecule_id[index]),
                molecule_type=self._check_topology_none(bytes.decode(molecule_type[index])),
                chain_id=self._check_topology_none(bytes.decode(chain_id[index])),
                mass=self._check_topology_none(mass[index]),
                charge=self._check_topology_none(charge[index])
            ))
        topology.add_particles(particles)
        # Bond
        bonds = self._file['topology/bonds'][()]
        num_bonds = self._file['topology/num_bonds'][()]
        for bond in range(num_bonds):
            topology.add_bond(list(bonds[bond, :]))
        # Angle
        angles = self._file['topology/angles'][()]
        num_angles = self._file['topology/num_angles'][()]
        for angle in range(num_angles):
            topology.add_angle(list(angles[angle, :]))
        # Dihedral
        dihedrals = self._file['topology/dihedrals'][()]
        num_dihedrals = self._file['topology/num_dihedrals'][()]
        for dihedral in range(num_dihedrals):
            topology.add_dihedral(list(dihedrals[dihedral, :]))
        # Improper
        impropers = self._file['topology/impropers'][()]
        num_impropers = self._file['topology/num_impropers'][()]
        for improper in range(num_impropers):
            topology.add_improper(list(impropers[improper, :]))
        topology.join()
        return topology

    def _parse_positions(self):
        if self._num_frames == 0:
            return None
        postions = np.empty([self._num_frames, self._num_particles, SPATIAL_DIM])
        for frame in range(self._num_frames):
            postions[frame, :, :] = self._file['positions/frame-%d' %frame][()].astype(env.NUMPY_FLOAT)
        return postions if self._num_frames != 1 else postions[0, :, :]

    def get_positions(self, *frames):
        self._file = h5py.File(self._file_path, 'r')
        num_target_frames = len(frames)
        if num_target_frames == 1:
            if frames[0] >= self._num_frames:
                raise ArrayDimError(
                    '%d beyond the number of frames %d stored in hdf5 file'
                    %(frames[0], self._num_frames)
                )
            result = self._file['positions/frame-%d' %frames[0]][()].copy().astype(env.NUMPY_FLOAT)
        else:
            result = np.zeros([num_target_frames, self._num_particles, SPATIAL_DIM])
            for index, frame in enumerate(frames):
                if frame >= self._num_frames:
                    raise ArrayDimError(
                        '%d beyond the number of frames %d stored in hdf5 file'
                        %(frame, self._num_frames)
                    )
                result[index, :, :] = self._file['positions/frame-%d' %frame][()].astype(env.NUMPY_FLOAT)
        self._file.close()
        return result

    def _parse_pbc_matrix(self):
        return self._file['pbc_matrix'][()].astype(env.NUMPY_FLOAT)

    @property
    def num_frames(self):
        return self._num_frames

    @property
    def num_particles(self):
        return self._num_particles

    @property
    def topology(self) -> Topology:
        return copy(self._topology)

    @property
    def positions(self) -> np.ndarray:
        if not self._is_parse_all:
            raise ParserPoorDefinedError(
                'positions property is not supported as `is_parse_all==False`, calling `get_position` method'
            )
        return self._positions.copy()

    @property
    def pbc_matrix(self) -> np.ndarray:
        return self._pbc_matrix.copy()

    @property
    def trajectory(self) -> Trajectory:
        if not self._is_parse_all:
            raise ParserPoorDefinedError(
                'trajectory property is not supported as `is_parse_all==False`'
            )
        return copy(self._trajectory)