#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : hdf5_writer.py
created time : 2022/02/24
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import h5py 
import numpy as np
from .. import env
from ..core import Topology
from ..utils import check_pbc_matrix, check_quantity_value
from ..unit import *
from ..error import *

class HDF5Writer:
    def __init__(
        self, file_path: str, mode: str = 'w', 
        topology: Topology = Topology(),
        pbc_matrix = np.diag([1]*3).astype(env.NUMPY_FLOAT)
    ) -> None:
        if not file_path.endswith('.hdf5'):
            raise FileFormatError('The file should end with .hdf5 suffix')
        self._file_path = file_path
        self._mode = mode
        if not isinstance(topology, Topology):
            raise TypeError(
                'The topology attribute should be the instance of mdpy.core.Topology class'
            )
        self._topology = topology
        pbc_matrix = check_quantity_value(pbc_matrix, default_length_unit)
        check_pbc_matrix(pbc_matrix)
        self._pbc_matrix = pbc_matrix

        with h5py.File(self._file_path, self._mode) as f:
            f.create_group('topology')
            f.create_group('positions')
            f['positions/num_frames'] = 0
            f['pbc_matrix'] = self._pbc_matrix
        self._write_topology()
        self._write_pbc_matrix()
        self._cur_frame = 0

    def write(self, positions=None):
        shape, is_shape_error = positions.shape, False
        is_2d_array = True if len(shape) == 2 else False
        if is_2d_array and shape[0] != self._topology.num_particles:
            is_shape_error = True
        elif not is_2d_array and len(shape) != 3:
            is_shape_error = True
        elif not is_2d_array and shape[1] != self._topology.num_particles:
            is_shape_error = True
        if is_shape_error:
            raise ArrayDimError(
                'The topology contain %s particles while a positions array with shape %s is provided' 
                %(self._topology.num_particles, list(shape))
            )
        with h5py.File(self._file_path, 'a') as h5f:
            if is_2d_array:
                h5f['positions/frame-%s' %self._cur_frame] = positions
                self._cur_frame += 1
            else:
                for frame in range(shape[0]):
                    h5f['positions/frame-%d' %self._cur_frame] = positions[frame, :, :]
                    self._cur_frame += 1
            del h5f['positions/num_frames']
            h5f['positions/num_frames'] = self._cur_frame

    def _write_pbc_matrix(self):
        with h5py.File(self._file_path, 'a') as h5f:
            del h5f['pbc_matrix']
            h5f['pbc_matrix'] = self._pbc_matrix
        
    def _write_topology(self):
        with h5py.File(self._file_path, 'a') as h5f:
            del h5f['topology']
            h5f.create_group('topology')
            h5f.create_group('topology/particles')
            # Assign data
            particle_id = np.empty([self._topology.num_particles], dtype=env.NUMPY_INT)
            particle_type = []
            particle_name = []
            matrix_id = np.empty([self._topology.num_particles], dtype=env.NUMPY_INT)
            molecule_id = np.empty([self._topology.num_particles], dtype=env.NUMPY_INT)
            molecule_type = []
            chain_id = []
            mass = np.empty([self._topology.num_particles], dtype=env.NUMPY_FLOAT)
            charge = np.empty([self._topology.num_particles], dtype=env.NUMPY_FLOAT)
            for index, particle in enumerate(self._topology.particles):
                particle_id[index] = self._check_none(particle.particle_id, env.NUMPY_INT)
                particle_type.append(self._check_none(particle.particle_type, str))
                particle_name.append(self._check_none(particle.particle_name, str))
                matrix_id[index] = self._check_none(particle.matrix_id, env.NUMPY_INT)
                molecule_id[index] = self._check_none(particle.molecule_id, env.NUMPY_INT)
                molecule_type.append(self._check_none(particle.molecule_type, str))
                chain_id.append(self._check_none(particle.chain_id, str))
                mass[index] = self._check_none(particle.mass, env.NUMPY_FLOAT)
                charge[index] = self._check_none(particle.charge, env.NUMPY_FLOAT)
            h5f['topology/particles/particle_id'] = particle_id
            h5f['topology/particles/particle_type'] = particle_type
            h5f['topology/particles/particle_name'] = particle_name
            h5f['topology/particles/matrix_id'] = matrix_id
            h5f['topology/particles/molecule_id'] = molecule_id
            h5f['topology/particles/molecule_type'] = molecule_type
            h5f['topology/particles/chain_id'] = chain_id
            h5f['topology/particles/mass'] = mass
            h5f['topology/particles/charge'] = charge
            h5f['topology/num_particles'] = env.NUMPY_INT(self._topology.num_particles)
            h5f['topology/bonds'] = np.array(self._topology.bonds).astype(env.NUMPY_INT)
            h5f['topology/num_bonds'] = env.NUMPY_INT(self._topology.num_bonds)
            h5f['topology/angles'] = np.array(self._topology.angles).astype(env.NUMPY_INT)
            h5f['topology/num_angles'] = env.NUMPY_INT(self._topology.num_angles)
            h5f['topology/dihedrals'] = np.array(self._topology.dihedrals).astype(env.NUMPY_INT)
            h5f['topology/num_dihedrals'] = env.NUMPY_INT(self._topology.num_dihedrals)
            h5f['topology/impropers'] = np.array(self._topology.impropers).astype(env.NUMPY_INT)
            h5f['topology/num_impropers'] = env.NUMPY_INT(self._topology.num_impropers)
            
    @staticmethod
    def _check_none(val, target_type):
        return target_type(val) if not isinstance(val, type(None)) else target_type(-1)
    
    @property
    def file_path(self):
        return self._file_path
    
    @property
    def mode(self):
        return self._mode

    @property
    def topology(self):
        return self._topology

    @topology.setter
    def topology(self, topology: Topology):
        if not isinstance(topology, Topology):
            raise TypeError(
                'The topology attribute should be the instance of mdpy.core.Topology class'
            )
        self._topology = topology
        self._write_topology()

    @property
    def pbc_matrix(self):
        return self._pbc_matrix

    @pbc_matrix.setter
    def pbc_matrix(self, pbc_matrix: np.ndarray):
        pbc_matrix = check_quantity_value(pbc_matrix, default_length_unit)
        check_pbc_matrix(pbc_matrix)
        self._pbc_matrix = check_quantity_value(pbc_matrix, default_length_unit)
        self._write_pbc_matrix()