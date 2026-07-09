"""Tests for PDBWriter — round-trip fidelity (write -> parse -> verify)."""

import os
import tempfile

import numpy as np
import pytest

from mdpy.io.pdb_writer import PDBWriter
from mdpy.io.pdb_parser import PDBParser

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _make_metadata(num_particles):
    particle_ids = np.arange(1, num_particles + 1, dtype=np.int32)
    # Cycle through atom names for large systems
    name_pool = ["CA", "CB", "N", "O", "H1", "H2", "H3", "C", "HA", "HB"]
    particle_names = [name_pool[i % len(name_pool)] for i in range(num_particles)]
    particle_molecule_ids = np.arange(1, num_particles + 1, dtype=np.int32)
    # Cycle through residue types
    res_pool = ["ALA", "VAL", "LEU", "PHE", "GLY"]
    particle_molecule_types = [res_pool[i % len(res_pool)] for i in range(num_particles)]
    particle_chain_ids = [chr(65 + (i // 200) % 26) for i in range(num_particles)]  # A, B, ...
    return (
        particle_ids,
        particle_names,
        particle_molecule_ids,
        particle_molecule_types,
        particle_chain_ids,
    )


class TestPDBWriter:

    def test_round_trip_positions(self):
        n = 10
        metadata = _make_metadata(n)
        writer = PDBWriter(*metadata)

        rng = np.random.default_rng(42)
        positions = rng.normal(20, 5, (n, 3)).astype(np.float64)
        pbc = np.eye(3) * 50.0

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            tmp = f.name

        try:
            writer.write(tmp, positions=positions, pbc_matrix=pbc)

            parsed = PDBParser(tmp)
            assert len(parsed.positions) == n
            np.testing.assert_allclose(parsed.positions, positions, atol=1e-3)
            assert parsed.pbc_matrix is not None
            assert parsed.pbc_matrix[0, 0] == pytest.approx(50.0)
        finally:
            os.unlink(tmp)

    def test_round_trip_metadata(self):
        n = 5
        metadata = _make_metadata(n)
        ids, names, _, mtypes, chains = metadata
        writer = PDBWriter(*metadata)

        positions = np.zeros((n, 3), dtype=np.float64) + 10.0

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            tmp = f.name

        try:
            writer.write(tmp, positions=positions)

            parsed = PDBParser(tmp)
            assert parsed.particle_ids == list(range(1, n + 1))
            assert parsed.particle_names == names[:n]
            assert parsed.particle_molecule_types == mtypes[:n]
            assert parsed.particle_chain_ids == chains[:n]
        finally:
            os.unlink(tmp)

    def test_round_trip_large_system(self):
        n = 1000
        metadata = _make_metadata(n)
        writer = PDBWriter(*metadata)

        rng = np.random.default_rng(42)
        positions = rng.normal(0, 10, (n, 3)).astype(np.float64)

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            tmp = f.name

        try:
            writer.write(tmp, positions=positions)

            parsed = PDBParser(tmp)
            assert len(parsed.positions) == n
            np.testing.assert_allclose(parsed.positions, positions, atol=1e-3)
        finally:
            os.unlink(tmp)

    def test_round_trip_with_velocities(self):
        n = 5
        metadata = _make_metadata(n)
        writer = PDBWriter(*metadata)

        positions = np.zeros((n, 3), dtype=np.float64) + 10.0
        velocities = np.ones((n, 3), dtype=np.float64) * 2.0

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            tmp = f.name

        try:
            writer.write(tmp, positions=positions, velocities=velocities)
            parsed = PDBParser(tmp)
            assert len(parsed.positions) == n
        finally:
            os.unlink(tmp)

    def test_round_trip_no_pbc(self):
        n = 5
        metadata = _make_metadata(n)
        writer = PDBWriter(*metadata)

        positions = np.zeros((n, 3), dtype=np.float64) + 10.0

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            tmp = f.name

        try:
            writer.write(tmp, positions=positions)
            parsed = PDBParser(tmp)
            assert parsed.pbc_matrix is None
            assert len(parsed.positions) == n
        finally:
            os.unlink(tmp)

    def test_round_trip_real_pdb(self):
        pdb_path = os.path.join(DATA_DIR, "6PO6.pdb")
        pdb = PDBParser(pdb_path)

        writer = PDBWriter(
            particle_ids=np.array(pdb.particle_ids),
            particle_names=pdb.particle_names,
            particle_molecule_ids=np.array(pdb.particle_molecule_ids),
            particle_molecule_types=pdb.particle_molecule_types,
            particle_chain_ids=pdb.particle_chain_ids,
        )

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            tmp = f.name

        try:
            writer.write(tmp, positions=pdb.positions, pbc_matrix=pdb.pbc_matrix)

            parsed = PDBParser(tmp)
            assert len(parsed.positions) == len(pdb.positions)
            np.testing.assert_allclose(parsed.positions, pdb.positions, atol=1e-3)
            assert parsed.particle_names == pdb.particle_names
            assert parsed.particle_molecule_types == pdb.particle_molecule_types
        finally:
            os.unlink(tmp)
