#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : pdb_parser.py
created time : 2021/10/03
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import numpy as np
from mdpy import env, SPATIAL_DIM
from mdpy.error import FileFormatError, ArrayDimensionError, ParserPoorlyDefinedError


def _guess_element(atom_name):
    name = atom_name.strip()
    if not name:
        return ''
    first = name[0]
    if first.isdigit():
        return name[1] if len(name) > 1 else ''
    return first.upper()


class PDBParser:
    def __init__(self, file_path, is_parse_all=True) -> None:
        if not file_path.endswith('.pdb'):
            raise FileFormatError('The file should end with .pdb suffix')
        self._file_path = file_path
        self._is_parse_all = is_parse_all
        with open(file_path, 'r') as f:
            lines = f.readlines()
        self._parse(lines)

    def _parse(self, lines):
        frames = []
        current_frame = []

        self._particle_ids = []
        self._particle_types = []
        self._particle_names = []
        self._molecule_ids = []
        self._molecule_types = []
        self._chain_ids = []
        self._matrix_ids = []
        self._pbc_matrix = None

        atom_index = 0
        first_atom_section = True

        for line in lines:
            record = line[:6].strip() if len(line) >= 6 else line.strip()

            if record == 'CRYST1' and self._pbc_matrix is None:
                self._pbc_matrix = self._parse_cryst1(line)

            if record in ('ATOM', 'HETATM'):
                if first_atom_section:
                    self._particle_ids.append(int(line[6:11]))
                    self._particle_names.append(line[12:16].strip())
                    self._particle_types.append(_guess_element(line[12:16]))
                    self._molecule_ids.append(int(line[22:26]))
                    self._molecule_types.append(line[17:21].strip())
                    self._chain_ids.append(line[21])
                    self._matrix_ids.append(atom_index)
                    atom_index += 1

                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                current_frame.append([x, y, z])

            elif record == 'ENDMDL':
                if current_frame:
                    frames.append(np.array(current_frame, dtype=env.NUMPY_FLOAT))
                    current_frame = []
                    first_atom_section = False

            elif record == 'END':
                if current_frame:
                    frames.append(np.array(current_frame, dtype=env.NUMPY_FLOAT))
                    current_frame = []
                    first_atom_section = False

        if current_frame:
            frames.append(np.array(current_frame, dtype=env.NUMPY_FLOAT))

        self._num_particles = len(self._particle_ids)
        self._num_frames = len(frames)

        if self._is_parse_all:
            if self._num_frames == 1:
                self._positions = frames[0] if frames else np.zeros((0, SPATIAL_DIM), dtype=env.NUMPY_FLOAT)
            else:
                self._positions = np.stack(frames) if frames else np.zeros((0, self._num_particles, SPATIAL_DIM), dtype=env.NUMPY_FLOAT)
        else:
            self._positions = None
            self._frames = frames

    @staticmethod
    def _parse_cryst1(line):
        a = float(line[6:15])
        b = float(line[15:24])
        c = float(line[24:33])
        alpha = float(line[33:40])
        beta = float(line[40:47])
        gamma = float(line[47:54])
        alpha_r = np.radians(alpha)
        beta_r = np.radians(beta)
        gamma_r = np.radians(gamma)
        box = np.zeros((3, 3))
        box[0, 0] = a
        box[1, 0] = b * np.cos(gamma_r)
        box[1, 1] = b * np.sin(gamma_r)
        box[2, 0] = c * np.cos(beta_r)
        box[2, 1] = c * (np.cos(alpha_r) - np.cos(beta_r) * np.cos(gamma_r)) / np.sin(gamma_r)
        box[2, 2] = np.sqrt(c * c - box[2, 0] ** 2 - box[2, 1] ** 2)
        return box

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

    def get_positions(self, *frames):
        num_target_frames = len(frames)
        if self._is_parse_all:
            if num_target_frames == 1:
                if frames[0] >= self._num_frames:
                    raise ArrayDimensionError(
                        '%d beyond the number of frames %d stored in pdb file'
                        %(frames[0], self._num_frames)
                    )
                return self._positions[frames[0]].copy() if self._num_frames > 1 else self._positions.copy()
            else:
                result = np.zeros([num_target_frames, self._num_particles, SPATIAL_DIM])
                for index, frame in enumerate(frames):
                    if frame >= self._num_frames:
                        raise ArrayDimensionError(
                            '%d beyond the number of frames %d stored in pdb file'
                            %(frame, self._num_frames)
                        )
                    result[index, :, :] = self._positions[frame]
                return result
        else:
            if num_target_frames == 1:
                if frames[0] >= self._num_frames:
                    raise ArrayDimensionError(
                        '%d beyond the number of frames %d stored in pdb file'
                        %(frames[0], self._num_frames)
                    )
                return self._frames[frames[0]].copy()
            else:
                result = np.zeros([num_target_frames, self._num_particles, SPATIAL_DIM])
                for index, frame in enumerate(frames):
                    if frame >= self._num_frames:
                        raise ArrayDimensionError(
                            '%d beyond the number of frames %d stored in pdb file'
                            %(frame, self._num_frames)
                        )
                    result[index, :, :] = self._frames[frame]
                return result

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
    def num_frames(self):
        return self._num_frames

    @property
    def num_particles(self):
        return self._num_particles

    @property
    def positions(self) -> np.ndarray:
        if not self._is_parse_all:
            raise ParserPoorlyDefinedError(
                'positions property is not supported as `is_parse_all==False`, calling `get_position` method'
            )
        return self._positions.copy()

    @property
    def pbc_matrix(self) -> np.ndarray:
        if self._pbc_matrix is None:
            return None
        return self._pbc_matrix.copy()
