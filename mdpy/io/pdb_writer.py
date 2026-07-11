#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : pdb_writer.py
created time : 2026/07/10
author : mdpy organization
copyright : (C)Copyright 2021-present, mdpy organization
'''

import numpy as np

from mdpy import SPATIAL_DIM


_ATOM_FORMAT = (
    "ATOM  %5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s  \n"
)
_CRYST1_FORMAT = (
    "CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1           1\n"
)


def _format_atom_name(name, element):
    name = str(name)[:4]
    if len(str(element).strip()) == 1 and len(name) <= 3:
        return (' ' + name).ljust(4)
    return name.ljust(4)


def _box_matrix_to_cell_params(box):
    box = np.asarray(box, dtype=np.float64)
    a = float(box[0, 0])

    b_x, b_y = float(box[1, 0]), float(box[1, 1])
    b = float(np.sqrt(b_x ** 2 + b_y ** 2))
    if b < 1e-10:
        gamma = 90.0
    else:
        gamma = float(np.degrees(np.arccos(np.clip(b_x / b, -1, 1))))

    c_x, c_y, c_z = float(box[2, 0]), float(box[2, 1]), float(box[2, 2])
    c = float(np.sqrt(c_x ** 2 + c_y ** 2 + c_z ** 2))
    if c < 1e-10:
        beta = 90.0
        alpha = 90.0
    else:
        beta = float(np.degrees(np.arccos(np.clip(c_x / c, -1, 1))))
        sin_gamma = float(np.sin(np.radians(gamma)))
        cos_gamma = float(np.cos(np.radians(gamma)))
        cos_beta = float(np.cos(np.radians(beta)))
        if sin_gamma < 1e-10:
            alpha = 90.0
        else:
            cos_alpha = c_y * sin_gamma / c + cos_beta * cos_gamma
            cos_alpha = float(np.clip(cos_alpha, -1, 1))
            alpha = float(np.degrees(np.arccos(cos_alpha)))

    return a, b, c, alpha, beta, gamma


class PDBWriter:
    """Write particle positions to a PDB file.

    Atom metadata (names, residue info, chain IDs, element symbols) is taken
    from a :class:`PDBParser` instance to ensure round-trip consistency
    between read and write.
    """

    def __init__(self, file_path):
        self._file_path = str(file_path)

    @property
    def file_path(self):
        return self._file_path

    def write(self, positions, pdb_parser, pbc_matrix=None):
        """Write *positions* to the PDB file.

        Parameters
        ----------
        positions : array_like, shape (N, 3)
            Particle coordinates in Angstroms, PDB order.
        pdb_parser : PDBParser
            Parser instance that supplies atom metadata.
        pbc_matrix : array_like, shape (3, 3), optional
            Periodic box matrix for the CRYST1 record.  When *None* the
            value from ``pdb_parser.pbc_matrix`` is used (may also be
            *None*, in which case no CRYST1 record is written).
        """
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != SPATIAL_DIM:
            raise ValueError(
                'positions must have shape (N, %d), got %s'
                % (SPATIAL_DIM, positions.shape)
            )

        num_particles = pdb_parser.num_particles
        if positions.shape[0] != num_particles:
            raise ValueError(
                'positions has %d particles but pdb_parser has %d'
                % (positions.shape[0], num_particles)
            )

        particle_names = pdb_parser.particle_names
        particle_ids = pdb_parser.particle_ids
        particle_residue_names = pdb_parser.particle_residue_names
        particle_chain_ids = pdb_parser.particle_chain_ids
        particle_residue_ids = pdb_parser.particle_residue_ids
        particle_type_names = pdb_parser.particle_type_names

        lines = []

        box = pbc_matrix if pbc_matrix is not None else pdb_parser.pbc_matrix
        if box is not None:
            a, b, c, alpha, beta, gamma = _box_matrix_to_cell_params(box)
            lines.append(_CRYST1_FORMAT % (a, b, c, alpha, beta, gamma))

        prev_chain = None
        for i in range(num_particles):
            chain_id = str(particle_chain_ids[i])[:1] if particle_chain_ids[i] else ' '
            if prev_chain is not None and chain_id != prev_chain:
                lines.append("TER\n")

            name_field = _format_atom_name(
                particle_names[i], particle_type_names[i]
            )
            element = str(particle_type_names[i])[:2] if particle_type_names[i] else '  '
            res_name = str(particle_residue_names[i])[:3] if particle_residue_names[i] else '   '
            serial = int(particle_ids[i])
            res_seq = int(particle_residue_ids[i])
            x, y, z = float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2])

            lines.append(_ATOM_FORMAT % (
                serial, name_field, res_name, chain_id, res_seq,
                x, y, z, 1.0, 0.0, element,
            ))
            prev_chain = chain_id

        lines.append("TER\n")
        lines.append("END\n")

        with open(self._file_path, 'w') as f:
            f.writelines(lines)
