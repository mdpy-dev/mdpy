from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from mdpy.io.pdb_parser import PDBParser
from mdpy.io.pdb_writer import PDBWriter

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
PDB_6PO6 = os.path.join(DATA_DIR, '6PO6.pdb')
PDB_1M9Z = os.path.join(DATA_DIR, '1M9Z.pdb')


class TestPDBWriterRoundTrip:

    def test_round_trip_6po6(self):
        pdb = PDBParser(PDB_6PO6)
        original_positions = pdb.positions

        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False, mode='w') as f:
            tmp_path = f.name
        try:
            writer = PDBWriter(tmp_path)
            writer.write(original_positions, pdb)

            pdb2 = PDBParser(tmp_path)
            roundtrip_positions = pdb2.positions

            assert pdb2.num_particles == pdb.num_particles
            np.testing.assert_allclose(roundtrip_positions, original_positions, atol=0.002)
        finally:
            os.unlink(tmp_path)

    def test_round_trip_preserves_atom_metadata(self):
        pdb = PDBParser(PDB_6PO6)
        original_positions = pdb.positions

        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False, mode='w') as f:
            tmp_path = f.name
        try:
            writer = PDBWriter(tmp_path)
            writer.write(original_positions, pdb)

            pdb2 = PDBParser(tmp_path)
            assert pdb2.particle_names == pdb.particle_names
            assert pdb2.particle_molecule_types == pdb.particle_molecule_types
            assert pdb2.particle_molecule_ids == pdb.particle_molecule_ids
            assert pdb2.particle_chain_ids == pdb.particle_chain_ids
            assert pdb2.particle_ids == pdb.particle_ids
        finally:
            os.unlink(tmp_path)

    def test_round_trip_cryst1_preserved(self):
        pdb = PDBParser(PDB_1M9Z)
        original_positions = pdb.positions

        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False, mode='w') as f:
            tmp_path = f.name
        try:
            writer = PDBWriter(tmp_path)
            writer.write(original_positions, pdb)

            pdb2 = PDBParser(tmp_path)
            assert pdb2.pbc_matrix is not None
            np.testing.assert_allclose(pdb2.pbc_matrix, pdb.pbc_matrix, atol=0.01)
        finally:
            os.unlink(tmp_path)

    def test_no_cryst1_when_pbc_none(self):
        pdb = PDBParser(PDB_6PO6)
        assert pdb.pbc_matrix is None

        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False, mode='w') as f:
            tmp_path = f.name
        try:
            writer = PDBWriter(tmp_path)
            writer.write(pdb.positions, pdb, pbc_matrix=None)

            with open(tmp_path, 'r') as f:
                first_line = f.readline()
            assert not first_line.startswith('CRYST1')
        finally:
            os.unlink(tmp_path)

    def test_explicit_pbc_matrix_overrides_parser(self):
        pdb = PDBParser(PDB_6PO6)
        custom_box = np.diag([80.0, 90.0, 100.0])

        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False, mode='w') as f:
            tmp_path = f.name
        try:
            writer = PDBWriter(tmp_path)
            writer.write(pdb.positions, pdb, pbc_matrix=custom_box)

            pdb2 = PDBParser(tmp_path)
            assert pdb2.pbc_matrix is not None
            np.testing.assert_allclose(pdb2.pbc_matrix, custom_box, atol=0.01)
        finally:
            os.unlink(tmp_path)

    def test_modified_positions_preserved(self):
        pdb = PDBParser(PDB_6PO6)
        shifted = pdb.positions + 1.5

        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False, mode='w') as f:
            tmp_path = f.name
        try:
            writer = PDBWriter(tmp_path)
            writer.write(shifted, pdb)

            pdb2 = PDBParser(tmp_path)
            np.testing.assert_allclose(pdb2.positions, shifted, atol=0.002)
        finally:
            os.unlink(tmp_path)

    def test_atom_record_column_layout(self):
        pdb = PDBParser(PDB_6PO6)

        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False, mode='w') as f:
            tmp_path = f.name
        try:
            writer = PDBWriter(tmp_path)
            writer.write(pdb.positions, pdb)

            with open(tmp_path, 'r') as f:
                lines = f.readlines()

            atom_lines = [l for l in lines if l.startswith('ATOM')]
            assert len(atom_lines) == pdb.num_particles

            first = atom_lines[0]
            assert first[:6] == 'ATOM  '
            assert int(first[6:11]) == pdb.particle_ids[0]
            assert first[12:16].strip() == pdb.particle_names[0]
            assert first[17:21].strip() == pdb.particle_molecule_types[0]
            assert first[21] == pdb.particle_chain_ids[0]
            assert int(first[22:26]) == pdb.particle_molecule_ids[0]
            float(first[30:38])
            float(first[38:46])
            float(first[46:54])
        finally:
            os.unlink(tmp_path)
