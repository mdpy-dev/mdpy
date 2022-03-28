#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : pdb_writer.py
created time : 2022/02/24
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import datetime
import numpy as np
from mdpy import env
from mdpy.core import Topology
from mdpy.utils import check_quantity_value, get_included_angle, check_pbc_matrix
from mdpy.unit import *
from mdpy.error import *

STD_RES_NAMES = [
    'ALA', 'ARG', 'ASN', 'ASP',
    'CYS', 'GLN', 'GLU', 'GLY',
    'HIS', 'ILE', 'LEU', 'LYS',
    'MET', 'PHE', 'PRO', 'SER',
    'THR', 'TRP', 'TYR', 'VAL'
]

# COLUMNS       DATA  TYPE     FIELD             DEFINITION
# ------------------------------------------------------------------------------------
#  1 -  6       Record name    "HEADER"
# 11 - 50       String(40)     classification    Classifies the molecule(s).
# 51 - 59       Date           depDate           Deposition date. This is the date the
#                                                coordinates  were received at the PDB.
# 63 - 66       IDcode         idCode            This identifier is unique within the PDB.
HEADER = 'HEADER' + ' '*4 + '%-40s%-11s\n'
# COLUMNS       DATA  TYPE    FIELD          DEFINITION
# -------------------------------------------------------------
#  1 -  6       Record name   "CRYST1"
#  7 - 15       Real(9.3)     a              a (Angstroms).
# 16 - 24       Real(9.3)     b              b (Angstroms).
# 25 - 33       Real(9.3)     c              c (Angstroms).
# 34 - 40       Real(7.2)     alpha          alpha (degrees).
# 41 - 47       Real(7.2)     beta           beta (degrees).
# 48 - 54       Real(7.2)     gamma          gamma (degrees).
# 56 - 66       LString       sGroup         Space  group.
# 67 - 70       Integer       z              Z value.
CRYST1 = 'CRYST1' + '%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f\n'
# COLUMNS        DATA  TYPE    FIELD          DEFINITION
# ---------------------------------------------------------------------------------------
#  1 -  6        Record name   "MODEL "
# 11 - 14        Integer       serial         Model serial number.
MODEL = 'MODEL ' + ' '*4 + '%4d\n'
ENDMDL = 'ENDMDL\n'
# COLUMNS        DATA  TYPE    FIELD        DEFINITION
# -------------------------------------------------------------------------------------
#  1 -  6        Record name   "ATOM  "
#  7 - 11        Integer       serial       Atom  serial number.
# 13 - 16        Atom          name         Atom name.
# 17             Character     altLoc       Alternate location indicator.
# 18 - 20        Residue name  resName      Residue name.
# 22             Character     chainID      Chain identifier.
# 23 - 26        Integer       resSeq       Residue sequence number.
# 27             AChar         iCode        Code for insertion of residues.
# 31 - 38        Real(8.3)     x            Orthogonal coordinates for X in Angstroms.
# 39 - 46        Real(8.3)     y            Orthogonal coordinates for Y in Angstroms.
# 47 - 54        Real(8.3)     z            Orthogonal coordinates for Z in Angstroms.
# 55 - 60        Real(6.2)     occupancy    Occupancy.
# 61 - 66        Real(6.2)     tempFactor   Temperature  factor.
# 77 - 78        LString(2)    element      Element symbol, right-justified.
# 79 - 80        LString(2)    charge       Charge  on the atom.
# Serial atomname resname chainid resid x y z 0 0 element
# ATOM = 'ATOM  ' + '%5d' + ' '*2 + '%-4s%-4s%1s%4d' + ' '*3 + '%8.3f%8.3f%8.3f%6.2f%6.2f' + ' '*10 + '%-2s\n'
def ATOM(serial, particle_name, molecule_name, chain_id, molecule_id, x, y, z, element):
    res = 'ATOM  '
    len_serial = len('%5d' %serial)
    res += '%5s' %serial + ' ' * (7 - len_serial)
    res += '%-4s%-4s%1s' %(particle_name[:4], molecule_name[:4], chain_id[:1])
    len_molecule_id = len('%4d' %molecule_id)
    res += '%4d' %molecule_id + ' ' * (8 - len_molecule_id)
    res += '%8.3f%8.3f%8.3f%6.2f%6.2f' %(x, y, z, 0, 0)
    res += ' ' * 10
    res += '%-2s\n' %element[:2]
    return res

# COLUMNS       DATA  TYPE     FIELD         DEFINITION
# -----------------------------------------------------------------------
#  1 - 6        Record name    "HETATM")
#  7 - 11       Integer        serial        Atom serial number.
# 14 - 16       Atom           name          Atom name.
# 17            Character      altLoc        Alternate location indicator.
# 18 - 20       Residue name   resName       Residue name.
# 22            Character      chainID       Chain identifier.
# 23 - 26       Integer        resSeq        Residue sequence number.
# 27            AChar          iCode         Code for insertion of residues.
# 31 - 38       Real(8.3)      x             Orthogonal coordinates for X.
# 39 - 46       Real(8.3)      y             Orthogonal coordinates for Y.
# 47 - 54       Real(8.3)      z             Orthogonal coordinates for Z.
# 55 - 60       Real(6.2)      occupancy     Occupancy.
# 61 - 66       Real(6.2)      tempFactor    Temperature factor.
# 77 - 78       LString(2)     element       Element symbol; right-justified.
# 79 - 80       LString(2)     charge        Charge on the atom.
# Serial atomname resname chainid resid x y z 0 0 element
# HETATM = 'HETATM' + '%5d' + ' '*2 + '%-4s%-4s%1s%5d' + ' '*3 + '%8.3f%8.3f%8.3f%6.2f%6.2f' + ' '*10 + '%-2s\n'
def HETATM(serial, particle_name, molecule_name, chain_id, molecule_id, x, y, z, element):
    res = 'HETATM'
    len_serial = len('%5d' %serial)
    res += '%5s' %serial + ' ' * (7 - len_serial)
    res += '%-4s%-4s%1s' %(particle_name[:4], molecule_name[:4], chain_id[:1])
    len_molecule_id = len('%4d' %molecule_id)
    res += '%4d' %molecule_id + ' ' * (8 - len_molecule_id)
    res += '%8.3f%8.3f%8.3f%6.2f%6.2f' %(x, y, z, 0, 0)
    res += ' ' * 10
    res += '%-2s\n' %element[:2]
    return res
# COLUMNS        DATA  TYPE    FIELD           DEFINITION
# -------------------------------------------------------------------------
#  1 -  6        Record name   "TER   "
#  7 - 11        Integer       serial          Serial number.
# 18 - 20        Residue name  resName         Residue name.
# 22             Character     chainID         Chain identifier.
# 23 - 26        Integer       resSeq          Residue sequence number.
# 27             AChar         iCode           Insertion code.
# Serial resname chainid resid
# TER = 'TER   ' + '%5d' + ' '*6 + '%-4s%1s%5d\n'
def TER(serial, molecule_name, chain_id, molecule_id):
    res = 'TER   '
    len_serial = len('%5d' %serial)
    res += '%5s' %serial + ' ' * (11 - len_serial)
    res += '%-4s%1s' %(molecule_name[:4], chain_id[:1])
    len_molecule_id = len('%4d' %molecule_id)
    res += '%4d' %molecule_id + ' ' * (8 - len_molecule_id) + '\n'
    return res

END = 'END'

class PDBWriter:
    def __init__(
        self, file_path: str, mode: str = 'w',
        topology: Topology=Topology(),
        pbc_matrix = np.diag([0]*3).astype(env.NUMPY_FLOAT)
    ) -> None:
        if not file_path.endswith('.pdb'):
            raise FileFormatError('The file should end with .pdb suffix')
        self._file_path = file_path
        self._mode = mode
        if not isinstance(topology, Topology):
            raise TypeError(
                'The topology attribute should be the instance of mdpy.core.Topology class'
            )
        self._topology = topology
        self._pbc_matrix = check_quantity_value(pbc_matrix, default_length_unit)
        f = open(file_path, mode)
        f.close()
        self._is_header = True
        self._cur_model = 0

    def write(self, positions: np.ndarray, is_end=False):
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
        # HEADER
        if self._is_header and not 'a' in self._mode:
            self._write_header()
            self._is_header = False
        # MODEL
        if is_2d_array:
            self._write_model(positions)
        else:
            for frame in range(shape[0]):
                self._write_model(positions[frame, :])
        # End
        if is_end:
            self._write_end()

    def _write_info(self, info: str):
        with open(self._file_path, 'a') as f:
            print(info, file=f, end='')

    def _write_header(self):
        header = HEADER %(
            'PDB FILE CREATED WITH MDPY',
            datetime.date.today().strftime('%d-%b-%Y').upper()
        )
        # Transport for the correction of origin transport in mdpy.core.State class
        pbc_len = np.linalg.norm(self._pbc_matrix, axis=0)
        alpha = get_included_angle(self._pbc_matrix[0, :], self._pbc_matrix[1, :], is_angular=False)
        beta = get_included_angle(self._pbc_matrix[0, :], self._pbc_matrix[2, :], is_angular=False)
        gamma = get_included_angle(self._pbc_matrix[1, :], self._pbc_matrix[2, :], is_angular=False)
        header += CRYST1 %(
            pbc_len[0], pbc_len[1], pbc_len[2],
            alpha, beta, gamma
        )
        self._write_info(header)

    def _write_model(self, positions: np.ndarray):
        model = MODEL % self._cur_model
        cur_chain_id, serial = self._topology.particles[0].chain_id[:1], 1
        for index, particle in enumerate(self._topology.particles):
            if cur_chain_id != particle.chain_id[:1]:
                cur_chain_id = particle.chain_id[:1]
                # Serial resname chainid resid
                pre_particle = self._topology.particles[index-1]
                model += TER(
                    serial, pre_particle.molecule_type,
                    pre_particle.chain_id, pre_particle.molecule_id
                )
                serial += 1
            # Serial atomname resname chainid resid x y z 0 0 element
            model += ATOM(
                serial, particle.particle_name, particle.molecule_type,
                particle.chain_id, particle.molecule_id,
                positions[index, 0], positions[index, 1], positions[index, 2],
                particle.particle_type
            )
            # if particle.molecule_type in STD_RES_NAMES:
            #     model += ATOM(
            #         serial, particle.particle_name, particle.molecule_type,
            #         particle.chain_id, particle.molecule_id,
            #         positions[index, 0], positions[index, 1], positions[index, 2],
            #         particle.particle_type
            #     )
            # else:
            #     model += HETATM(
            #         serial, particle.particle_name, particle.molecule_type,
            #         particle.chain_id, particle.molecule_id,
            #         positions[index, 0], positions[index, 1], positions[index, 2],
            #         particle.particle_type
            #     )
            serial += 1
        pre_particle = self._topology.particles[-1]
        model += TER(
            serial, pre_particle.molecule_type,
            pre_particle.chain_id, pre_particle.molecule_id
        )
        model += ENDMDL
        self._write_info(model)
        self._cur_model += 1

    def _write_end(self):
        self._write_info(END)

    def close(self):
        self._write_end()

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

    @property
    def pbc_matrix(self):
        return self._pbc_matrix

    @pbc_matrix.setter
    def pbc_matrix(self, pbc_matrix: np.ndarray):
        pbc_matrix = check_quantity_value(pbc_matrix, default_length_unit)
        self._pbc_matrix = check_pbc_matrix(pbc_matrix)
