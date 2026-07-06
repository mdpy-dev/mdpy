"""Tests for pre-allocated block arrays (zero-readback path).

Validates that max_blocks / max_total_padded are correct upper bounds so
block arrays can be sized without reading num_blocks / total_padded back
from the GPU.
"""

import numpy as np
import pytest
from mdpy.core.topology import Builder
from mdpy.core.block_list import BlockList, BLOCK_SIZE


def _make_topology(n):
    builder = Builder()
    builder.set_particles(
        masses=np.ones(n, dtype=np.float32),
        charges=np.zeros(n, dtype=np.float32),
        particle_types=np.zeros(n, dtype=np.int32),
    )
    builder.build_exclusion_map()
    topology, _ = builder.build()
    return topology


def _make_positions(n, box=50.0, seed=42):
    rng = np.random.RandomState(seed)
    return rng.uniform(0, box, (n, 3)).astype(np.float32)


class TestMaxBlocksComputed:
    def test_max_blocks_formula(self):
        """_compute_cell_grid stores max_blocks = ceil(N/32) + nc_total."""
        bl = BlockList(cutoff=12.0, skin=2.0)
        pbc = np.eye(3, dtype=np.float64) * 50.0
        N = 1000
        bl._compute_cell_grid(pbc, N)
        expected = (N + BLOCK_SIZE - 1) // BLOCK_SIZE + bl.nc_total
        assert bl.max_blocks == expected
        assert bl.max_total_padded == bl.max_blocks * BLOCK_SIZE

    def test_max_blocks_strictly_exceeds_nc_total(self):
        """max_blocks must account for multi-block cells, not just nc_total."""
        bl = BlockList(cutoff=12.0, skin=2.0)
        pbc = np.eye(3, dtype=np.float64) * 50.0
        bl._compute_cell_grid(pbc, 1000)
        assert bl.max_blocks > bl.nc_total

    def test_max_blocks_init_zero(self):
        bl = BlockList(cutoff=12.0, skin=2.0)
        assert bl.max_blocks == 0
        assert bl.max_total_padded == 0


class TestMaxBlocksUpperBound:
    """After a real rebuild, num_blocks <= max_blocks and block arrays
    are large enough to hold all data."""

    @pytest.mark.parametrize("n,box", [
        (100, 50.0),
        (1000, 50.0),
        (5000, 80.0),
    ])
    def test_num_blocks_within_bound(self, n, box):
        positions = _make_positions(n, box)
        topology = _make_topology(n)
        pbc_matrix = np.eye(3, dtype=np.float32) * box
        pbc_inv = np.linalg.inv(pbc_matrix)
        bl = BlockList(cutoff=10.0, skin=2.0)
        bl.rebuild(positions, topology, pbc_matrix, pbc_inv, force=True)
        assert bl.num_blocks <= bl.max_blocks, (
            f"num_blocks={bl.num_blocks} exceeds max_blocks={bl.max_blocks}"
        )

    @pytest.mark.parametrize("n,box", [
        (100, 50.0),
        (1000, 50.0),
    ])
    def test_block_atoms_large_enough(self, n, box):
        """block_atoms (sized max_total_padded) holds all real atoms."""
        import cupy as cp
        positions = _make_positions(n, box)
        topology = _make_topology(n)
        pbc_matrix = np.eye(3, dtype=np.float32) * box
        pbc_inv = np.linalg.inv(pbc_matrix)
        bl = BlockList(cutoff=10.0, skin=2.0)
        bl.rebuild(positions, topology, pbc_matrix, pbc_inv, force=True)
        ba = cp.asnumpy(bl.d_block_atoms)
        assert ba.size == bl.max_total_padded
        real = ba[ba >= 0]
        assert len(np.unique(real)) == n

    def test_block_center_arrays_sized_max(self):
        """block_center/size arrays are pre-allocated to max_blocks."""
        positions = _make_positions(1000, 50.0)
        topology = _make_topology(1000)
        pbc_matrix = np.eye(3, dtype=np.float32) * 50.0
        pbc_inv = np.linalg.inv(pbc_matrix)
        bl = BlockList(cutoff=10.0, skin=2.0)
        bl.rebuild(positions, topology, pbc_matrix, pbc_inv, force=True)
        for attr in ("d_block_center_x", "d_block_center_y", "d_block_center_z",
                      "d_block_size_x", "d_block_size_y", "d_block_size_z"):
            arr = getattr(bl, attr)
            assert arr.size >= bl.max_blocks, (
                f"{attr}.size={arr.size} < max_blocks={bl.max_blocks}"
            )
