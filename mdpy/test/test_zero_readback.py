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
        bl.rebuild(positions, topology, pbc_matrix, pbc_inv)
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
        bl.rebuild(positions, topology, pbc_matrix, pbc_inv)
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
        bl.rebuild(positions, topology, pbc_matrix, pbc_inv)
        for attr in ("d_block_center_x", "d_block_center_y", "d_block_center_z",
                      "d_block_size_x", "d_block_size_y", "d_block_size_z"):
            arr = getattr(bl, attr)
            assert arr.size >= bl.max_blocks, (
                f"{attr}.size={arr.size} < max_blocks={bl.max_blocks}"
            )


class TestCellAssignFlagCheck:
    def test_flag_zero_skips_cell_assign(self):
        """cell_assign must early-exit when d_rebuild_flag=0, leaving cell_counts zero.

        rebuild() unconditionally sets the flag to 1 before launching cell_assign
        (it is the "do a full rebuild" entry point), so this test launches the
        cell_assign kernel directly with flag=0 to verify the GPU-side guard that
        will power the future zero-readback pipeline.
        """
        import cupy as cp

        n = 100
        topology = _make_topology(n)
        positions = _make_positions(n)
        pbc = np.eye(3, dtype=np.float32) * 50.0
        pbc_inv = np.linalg.inv(pbc)

        bl = BlockList(cutoff=10.0, skin=2.0)
        bl._ensure_kernels()
        bl.d_rebuild_flag[0] = 1
        bl.rebuild(positions, topology, pbc, pbc_inv)
        assert bl.num_blocks > 0  # normal rebuild produces blocks

        # Re-launch cell_assign directly with flag=0. rebuild() already set up
        # _d_pbc_matrix/_d_pbc_inv/nc_*/_hilbert_levels, so reuse them.
        data = cp.asarray(np.ascontiguousarray(positions.ravel(), dtype=np.float32))
        pos_x = data[0::3].copy()
        pos_y = data[1::3].copy()
        pos_z = data[2::3].copy()

        cell_counts = bl._d_cell_counts
        cell_counts[:] = 0  # zero so we can detect any write by cell_assign
        bl.d_rebuild_flag[0] = 0

        composite_buckets = bl.nc_total * (1 << (3 * bl._hilbert_levels))
        d_composite_counts = bl._pool_get(
            "composite_counts", composite_buckets, env.NUMPY_INT, fill=0
        )
        sort_keys = bl._pool_get("sort_keys", n, np.uint64)
        cell_indices = bl._pool_get("cell_indices", n, env.NUMPY_INT)

        tpb = 256
        nm = (n + tpb - 1) // tpb
        bl._kernels["cell_assign"](
            (nm,), (tpb,),
            (
                pos_x, pos_y, pos_z,
                bl._d_pbc_matrix, bl._d_pbc_inv,
                np.int32(n),
                np.int32(bl.nc_x), np.int32(bl.nc_y), np.int32(bl.nc_z),
                np.int32(bl._hilbert_levels),
                cell_counts, d_composite_counts, sort_keys, cell_indices,
                bl.d_rebuild_flag,
            ),
        )

        total = int(cell_counts.sum().get())
        assert total == 0, (
            f"cell_assign should have skipped (flag=0), but cell_counts sum={total}"
        )
