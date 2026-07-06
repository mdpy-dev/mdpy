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
from mdpy.core.gpu_context import GPUContext


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


class TestPairFinderFlagGuard:
    def test_flag_zero_preserves_block_pairs(self):
        """When flag=0, find_interacting and build_masks must skip so block
        pair data, the pair counter, and exclusion masks all retain their
        previous valid values."""
        import cupy as cp

        n = 1000
        topology = _make_topology(n)
        positions = _make_positions(n)
        pbc = np.eye(3, dtype=np.float32) * 50.0
        pbc_inv = np.linalg.inv(pbc)

        bl = BlockList(cutoff=10.0, skin=2.0)
        bl.rebuild(positions, topology, pbc, pbc_inv, force=True)
        bl.build_block_pairs(topology, pbc)

        old_pairs = cp.asnumpy(bl._d_block_pair_buf).copy()
        old_counters = int(bl._d_counters[0].get())
        old_masks = cp.asnumpy(bl.d_exclusion_masks).copy()
        assert old_counters > 0, "expected nonzero pair count after full rebuild"

        bl.d_rebuild_flag[0] = 0
        bl.build_block_pairs(topology, pbc)

        new_pairs = cp.asnumpy(bl._d_block_pair_buf)
        new_counters = int(bl._d_counters[0].get())
        new_masks = cp.asnumpy(bl.d_exclusion_masks)

        assert new_counters == old_counters, (
            f"d_counters changed: {old_counters} -> {new_counters}")
        np.testing.assert_array_equal(new_pairs, old_pairs,
            err_msg="block pair buffer changed during flag=0 build_block_pairs")
        np.testing.assert_array_equal(new_masks, old_masks,
            err_msg="exclusion masks changed during flag=0 build_block_pairs")


class TestCaptureSnapshotFlagGuard:
    def test_flag_zero_preserves_baseline(self):
        """When flag=0, capture_snapshot kernel skips so the displacement
        baseline is not overwritten (no amnesia). Verified by passing
        different positions the second time: the baseline must NOT update."""
        import cupy as cp

        n = 500
        topology = _make_topology(n)
        positions = _make_positions(n)
        pbc = np.eye(3, dtype=np.float32) * 50.0
        pbc_inv = np.linalg.inv(pbc)

        bl = BlockList(cutoff=10.0, skin=2.0)
        bl.rebuild(positions, topology, pbc, pbc_inv, force=True)

        # flag=1 after force=True rebuild: first capture writes the baseline.
        pos_x = cp.asarray(np.ascontiguousarray(positions[:, 0], dtype=np.float32))
        pos_y = cp.asarray(np.ascontiguousarray(positions[:, 1], dtype=np.float32))
        pos_z = cp.asarray(np.ascontiguousarray(positions[:, 2], dtype=np.float32))
        bl.capture_snapshot((pos_x, pos_y, pos_z))

        old_snap = cp.asnumpy(bl.d_positions_at_rebuild_x).copy()

        # Build different positions that WOULD overwrite the baseline if the
        # kernel ran. Use a large offset so a copy would be unambiguous.
        shifted_x = pos_x + 100.0
        shifted_y = pos_y + 100.0
        shifted_z = pos_z + 100.0

        bl.d_rebuild_flag[0] = 0
        bl.capture_snapshot((shifted_x, shifted_y, shifted_z))

        new_snap = cp.asnumpy(bl.d_positions_at_rebuild_x)
        np.testing.assert_array_equal(new_snap, old_snap,
            err_msg="flag=0 capture_snapshot overwrote the baseline (amnesia)")


class TestPermuteStateArraysFlagGuard:
    """permute_state_arrays must skip entirely when d_rebuild_flag=0.

    Without the guard, applying the old permutation to already-permuted
    arrays (which is what happens if _permute_all_arrays runs on a no-op
    rebuild) double-permutes all 14 state arrays, corrupting positions,
    velocities, forces, prev_positions, masses, and charges.
    """

    def test_flag_zero_skips_permutation(self):
        """When flag=0 the kernel must not write to dst. Verified by
        pre-filling the next pool buffer with a sentinel: if the kernel
        ran it would overwrite the sentinel with permuted data."""
        import cupy as cp

        n = 1000
        topology = _make_topology(n)
        positions = _make_positions(n)
        pbc = np.eye(3, dtype=np.float32) * 50.0
        pbc_inv = np.linalg.inv(pbc)

        gpu = GPUContext()
        gpu.initialize(topology, pbc)
        gpu.upload_positions(positions)

        bl = BlockList(cutoff=10.0, skin=2.0)
        bl.rebuild(positions, topology, pbc, pbc_inv, force=True)
        # flag is 1 after force=True rebuild
        assert int(bl.d_rebuild_flag[0].get()) == 1

        perm = bl.d_raw_order
        perm_np = cp.asnumpy(perm)
        assert not np.array_equal(perm_np, np.arange(n, dtype=np.int32)), (
            "permutation is identity — test cannot detect a double-permute"
        )

        pairs = [
            ("d_positions_x", gpu.d_positions_x),
            ("d_positions_y", gpu.d_positions_y),
            ("d_positions_z", gpu.d_positions_z),
            ("d_velocities_x", gpu.d_velocities_x),
            ("d_velocities_y", gpu.d_velocities_y),
            ("d_velocities_z", gpu.d_velocities_z),
            ("d_forces_x", gpu.d_forces_x),
            ("d_forces_y", gpu.d_forces_y),
            ("d_forces_z", gpu.d_forces_z),
            ("d_prev_positions_x", gpu.d_prev_positions_x),
            ("d_prev_positions_y", gpu.d_prev_positions_y),
            ("d_prev_positions_z", gpu.d_prev_positions_z),
            ("d_masses", gpu.d_masses),
            ("d_charges", gpu.d_charges),
        ]

        # flag=1: real permute. Apply results so src becomes permuted
        # (simulating what _permute_all_arrays does via setattr).
        result1 = gpu.permute_state_arrays(perm, pairs, bl.d_rebuild_flag)
        for name, new_arr in result1:
            setattr(gpu, name, new_arr)

        # The first call used pool_A and flipped _perm_flip to True, so the
        # next call will write into pool_B. Pre-fill pool_B with a sentinel
        # so we can detect whether the kernel wrote to it.
        sentinel = -777.0
        for buf in gpu._perm_pool_B:
            buf[:n] = sentinel

        # flag=0: kernel must skip. src is now the permuted arrays, so if the
        # guard were missing the kernel would write permuted[perm] over the
        # sentinel.
        bl.d_rebuild_flag[0] = 0
        pairs_permuted = [
            ("d_positions_x", gpu.d_positions_x),
            ("d_positions_y", gpu.d_positions_y),
            ("d_positions_z", gpu.d_positions_z),
            ("d_velocities_x", gpu.d_velocities_x),
            ("d_velocities_y", gpu.d_velocities_y),
            ("d_velocities_z", gpu.d_velocities_z),
            ("d_forces_x", gpu.d_forces_x),
            ("d_forces_y", gpu.d_forces_y),
            ("d_forces_z", gpu.d_forces_z),
            ("d_prev_positions_x", gpu.d_prev_positions_x),
            ("d_prev_positions_y", gpu.d_prev_positions_y),
            ("d_prev_positions_z", gpu.d_prev_positions_z),
            ("d_masses", gpu.d_masses),
            ("d_charges", gpu.d_charges),
        ]
        result2 = gpu.permute_state_arrays(perm, pairs_permuted, bl.d_rebuild_flag)

        for name, arr in result2:
            arr_np = cp.asnumpy(arr)
            assert np.all(arr_np == sentinel), (
                f"{name} was written by flag=0 kernel — guard failed "
                f"(double-permute corruption)"
            )


class TestWrapCorrectFlagGuard:
    """wrap_correct_kernel must skip entirely when d_rebuild_flag=0.

    Without the guard, the kernel wraps positions back into the PBC box and
    applies the same displacement to prev_positions. If the guard were
    missing, calling wrap_positions_with_prev_correction on a no-rebuild
    step would corrupt both position arrays.
    """

    def test_flag_zero_skips_wrap_correct(self):
        """When flag=0 the kernel must not write to positions or
        prev_positions. Verified by pre-filling both arrays with sentinels
        outside the PBC box: if the kernel ran it would wrap them, so any
        change proves the guard failed.

        Box is 4.0 and the position sentinel is 5.0 (outside [0, 4)), so
        an unguarded kernel would wrap 5.0 -> 1.0 and shift prev by -4.0.
        """
        import cupy as cp

        n = 100
        topology = _make_topology(n)
        box = 4.0
        pbc = np.eye(3, dtype=np.float32) * box

        gpu = GPUContext()
        gpu.initialize(topology, pbc)

        pos_sentinel = 5.0
        prev_sentinel = 3.0
        gpu.d_positions_x[:] = pos_sentinel
        gpu.d_positions_y[:] = pos_sentinel
        gpu.d_positions_z[:] = pos_sentinel
        gpu.d_prev_positions_x[:] = prev_sentinel
        gpu.d_prev_positions_y[:] = prev_sentinel
        gpu.d_prev_positions_z[:] = prev_sentinel

        d_rebuild_flag = cp.array([0], dtype=cp.int32)
        gpu.wrap_positions_with_prev_correction(d_rebuild_flag)

        for arr in (gpu.d_positions_x, gpu.d_positions_y, gpu.d_positions_z):
            assert np.all(cp.asnumpy(arr) == pos_sentinel), (
                "positions changed during flag=0 wrap_correct — guard failed"
            )
        for arr in (
            gpu.d_prev_positions_x,
            gpu.d_prev_positions_y,
            gpu.d_prev_positions_z,
        ):
            assert np.all(cp.asnumpy(arr) == prev_sentinel), (
                "prev_positions changed during flag=0 wrap_correct — guard failed"
            )

    def test_flag_one_actually_wraps(self):
        """Sanity check proving the flag=0 test above is non-vacuous: with
        flag=1 the kernel runs and wraps the out-of-box sentinel (5.0 -> 1.0
        in a 4.0 box, prev shifted by the same -4.0 displacement)."""
        import cupy as cp

        n = 100
        topology = _make_topology(n)
        box = 4.0
        pbc = np.eye(3, dtype=np.float32) * box

        gpu = GPUContext()
        gpu.initialize(topology, pbc)

        gpu.d_positions_x[:] = 5.0
        gpu.d_positions_y[:] = 5.0
        gpu.d_positions_z[:] = 5.0
        gpu.d_prev_positions_x[:] = 3.0
        gpu.d_prev_positions_y[:] = 3.0
        gpu.d_prev_positions_z[:] = 3.0

        d_rebuild_flag = cp.array([1], dtype=cp.int32)
        gpu.wrap_positions_with_prev_correction(d_rebuild_flag)

        pos_np = cp.asnumpy(gpu.d_positions_x)
        assert np.allclose(pos_np, 1.0), (
            f"flag=1 should wrap 5.0 -> 1.0, got {pos_np[0]}"
        )
        prev_np = cp.asnumpy(gpu.d_prev_positions_x)
        assert np.allclose(prev_np, -1.0), (
            f"flag=1 should shift prev 3.0 -> -1.0, got {prev_np[0]}"
        )
