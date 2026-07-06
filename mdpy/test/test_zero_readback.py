"""Tests for pre-allocated block arrays (zero-readback path).

Validates that max_blocks / max_total_padded are correct upper bounds so
block arrays can be sized without reading num_blocks / total_padded back
from the GPU.
"""

import numpy as np
import pytest
from mdpy import env
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


class TestCellAssignFlagCheck:
    def test_flag_zero_skips_cell_assign(self):
        """When force=False and d_rebuild_flag=0, cell_assign skips entirely so
        cell_counts stays zero. The prefix-sum kernels are also flag-guarded,
        so d_num_blocks is preserved (see TestPrefixSumFlagGuard) rather than
        being overwritten to 0."""
        import cupy as cp

        n = 1000
        topology = _make_topology(n)
        positions = _make_positions(n)
        pbc = np.eye(3, dtype=np.float32) * 50.0
        pbc_inv = np.linalg.inv(pbc)

        bl = BlockList(cutoff=10.0, skin=2.0)

        # force=True: normal rebuild produces blocks
        bl.rebuild(positions, topology, pbc, pbc_inv, force=True)
        d_num_blocks = bl._pool.get(("num_blocks", env.NUMPY_INT))
        prev_count = int(d_num_blocks[0].get())
        assert prev_count > 0

        # force=False with flag=0: cell_assign skips (cell_counts stays zero)
        # and the prefix-sum kernels are guarded so num_blocks is preserved.
        bl.d_rebuild_flag[0] = 0
        bl.rebuild(positions, topology, pbc, pbc_inv, force=False)

        cell_counts = cp.asnumpy(bl._d_cell_counts)
        assert not cell_counts.any(), "cell_assign should have been skipped"
        assert int(d_num_blocks[0].get()) == prev_count, (
            f"flag=0 should preserve num_blocks={prev_count}"
        )

    def test_flag_zero_preserves_force_data(self):
        """When flag=0 rebuild is skipped, block_atoms and d_counters are
        preserved so the force kernel can reuse the previous valid block list.

        In production, update_neighbor_list returns early (no _do_rebuild call)
        when no rebuild is needed — neither rebuild nor build_block_pairs runs.
        This test verifies that rebuild(force=False) alone does NOT corrupt
        block_atoms (via conditional_fill) and that d_counters is untouched
        because build_block_pairs is not called (matching production flow)."""
        import cupy as cp

        n = 1000
        topology = _make_topology(n)
        positions = _make_positions(n)
        pbc = np.eye(3, dtype=np.float32) * 50.0
        pbc_inv = np.linalg.inv(pbc)

        bl = BlockList(cutoff=10.0, skin=2.0)
        bl.rebuild(positions, topology, pbc, pbc_inv, force=True)
        bl.build_block_pairs(topology, pbc)

        prev_ba = cp.asnumpy(bl.d_block_atoms).copy()
        prev_pairs = int(bl._d_counters[0].get())
        assert prev_pairs > 0, "expected nonzero pair count after full rebuild"

        # Simulate production: when no rebuild needed, rebuild(force=False)
        # is called but build_block_pairs is NOT (update_neighbor_list returns
        # early). The conditional_fill in rebuild preserves block_atoms.
        bl.d_rebuild_flag[0] = 0
        bl.rebuild(positions, topology, pbc, pbc_inv, force=False)
        # NOTE: build_block_pairs is intentionally NOT called here (matches
        # production flow where update_neighbor_list returns early).

        curr_ba = cp.asnumpy(bl.d_block_atoms)
        curr_pairs = int(bl._d_counters[0].get())

        npt = np.testing
        npt.assert_array_equal(curr_ba, prev_ba,
            err_msg="block_atoms changed during flag=0 rebuild")
        assert curr_pairs == prev_pairs, (
            f"d_counters changed: {prev_pairs} → {curr_pairs}")


class TestPrefixSumFlagGuard:
    def test_flag_zero_preserves_num_blocks(self):
        """When flag=0, cell_prefix_sum must skip so d_num_blocks retains
        its previous valid value instead of being written to 0."""
        n = 1000
        topology = _make_topology(n)
        positions = _make_positions(n)
        pbc = np.eye(3, dtype=np.float32) * 50.0
        pbc_inv = np.linalg.inv(pbc)

        bl = BlockList(cutoff=10.0, skin=2.0)
        bl.rebuild(positions, topology, pbc, pbc_inv, force=True)
        d_num_blocks = bl._pool.get(("num_blocks", env.NUMPY_INT))
        old_count = int(d_num_blocks[0].get())
        assert old_count > 0

        bl.d_rebuild_flag[0] = 0
        bl.rebuild(positions, topology, pbc, pbc_inv, force=False)
        new_count = int(d_num_blocks[0].get())
        assert new_count == old_count, (
            f"d_num_blocks changed: {old_count} -> {new_count}"
        )
