"""Tests for pre-allocated block arrays (zero-readback path).

Validates that max_blocks / max_total_padded are correct upper bounds so
block arrays can be sized without reading num_blocks / total_padded back
from the GPU.
"""

import numpy as np
import pytest
from mdpy import env
from mdpy.core.topology import Builder, permute_exclusion_pairs_gpu
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


class TestPermuteArrayFlagGuard:
    """permute_array_kernel / permute_int_array_kernel (used by
    permute_to_sorted and permute_to_sorted_inplace) must skip entirely
    when d_rebuild_flag=0.

    Without the guard, calling these on a no-rebuild step would gather
    src[perm] over dst even when no new permutation was produced,
    double-permuting any already-sorted data the caller is holding.
    """

    def test_flag_zero_skips_float_inplace(self):
        """When flag=0 the kernel must not write to dst. Verified by
        pre-filling dst with a sentinel: if the kernel ran it would
        overwrite the sentinel with src[perm]."""
        import cupy as cp

        n = 1000
        topology = _make_topology(n)
        positions = _make_positions(n)
        pbc = np.eye(3, dtype=np.float32) * 50.0
        pbc_inv = np.linalg.inv(pbc)

        gpu = GPUContext()
        gpu.initialize(topology, pbc)

        bl = BlockList(cutoff=10.0, skin=2.0)
        bl.rebuild(positions, topology, pbc, pbc_inv, force=True)

        perm = bl.d_raw_order
        src = cp.asarray(np.arange(n, dtype=np.float32))
        sentinel = -777.0
        dst = cp.full(n, sentinel, dtype=cp.float32)

        bl.d_rebuild_flag[0] = 0
        gpu.permute_to_sorted_inplace(perm, src, dst, bl.d_rebuild_flag)

        dst_np = cp.asnumpy(dst)
        assert np.all(dst_np == sentinel), (
            "flag=0 permute kernel wrote to dst — guard failed "
            "(double-permute corruption)"
        )

    def test_flag_one_actually_perms_float(self):
        """Sanity check proving the flag=0 test above is non-vacuous:
        with flag=1 the kernel runs and dst[i] == src[perm[i]]."""
        import cupy as cp

        n = 1000
        topology = _make_topology(n)
        positions = _make_positions(n)
        pbc = np.eye(3, dtype=np.float32) * 50.0
        pbc_inv = np.linalg.inv(pbc)

        gpu = GPUContext()
        gpu.initialize(topology, pbc)

        bl = BlockList(cutoff=10.0, skin=2.0)
        bl.rebuild(positions, topology, pbc, pbc_inv, force=True)
        assert int(bl.d_rebuild_flag[0].get()) == 1

        perm_np = cp.asnumpy(bl.d_raw_order)
        src = cp.asarray(np.arange(n, dtype=np.float32))
        dst = cp.zeros(n, dtype=cp.float32)

        gpu.permute_to_sorted_inplace(bl.d_raw_order, src, dst, bl.d_rebuild_flag)

        expected = cp.asnumpy(src)[perm_np]
        np.testing.assert_array_equal(cp.asnumpy(dst), expected)

    def test_flag_zero_skips_int_permute(self):
        """int variant: permute_to_sorted allocates dst internally so a
        sentinel pre-fill is not possible; instead verify the returned
        array does NOT equal the correctly-permuted reference. With
        flag=0 the kernel skips and cp.empty_like returns uninitialized
        memory, which will not match src[perm] for an arange source."""
        import cupy as cp

        n = 1000
        topology = _make_topology(n)
        positions = _make_positions(n)
        pbc = np.eye(3, dtype=np.float32) * 50.0
        pbc_inv = np.linalg.inv(pbc)

        gpu = GPUContext()
        gpu.initialize(topology, pbc)

        bl = BlockList(cutoff=10.0, skin=2.0)
        bl.rebuild(positions, topology, pbc, pbc_inv, force=True)

        perm_np = cp.asnumpy(bl.d_raw_order)
        src_np = np.arange(n, dtype=np.int32)
        expected = src_np[perm_np]

        bl.d_rebuild_flag[0] = 0
        arrays = {"x": cp.asarray(src_np)}
        gpu.permute_to_sorted(
            bl.d_raw_order, {}, arrays_int=arrays,
            d_rebuild_flag=bl.d_rebuild_flag,
        )
        result = cp.asnumpy(arrays["x"])
        assert not np.array_equal(result, expected), (
            "flag=0 int permute kernel wrote permuted data — guard failed"
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


class TestComposePermFlagGuard:
    """compose_perm kernel must skip entirely when d_rebuild_flag=0.

    Without the guard, calling the kernel on a no-rebuild step would
    overwrite d_composed_perm with out[perm[i]] = i, producing a stale
    composed permutation that double-permutes the exclusion pairs.
    """

    def test_flag_zero_skips_compose_perm(self):
        """When flag=0 the kernel must not write to out. Verified by
        pre-filling out with a sentinel: if the kernel ran it would
        overwrite the sentinel with out[perm[i]] = i."""
        import cupy as cp
        from mdpy.core._rebuild_kernels import compile_rebuild_kernels

        N = 1000
        perm = cp.arange(N, 0, -1, dtype=cp.int32)
        sentinel = -777
        out = cp.full(N, sentinel, dtype=cp.int32)
        flag = cp.array([0], dtype=cp.int32)

        rk = compile_rebuild_kernels()
        grid = ((N + 255) // 256,)
        rk["compose_perm"](grid, (256,), (out, perm, np.int32(N), flag))
        cp.cuda.Device().synchronize()

        out_np = cp.asnumpy(out)
        assert np.all(out_np == sentinel), (
            "flag=0 compose_perm wrote to out — guard failed"
        )

    def test_flag_one_actually_writes(self):
        """Sanity check proving the flag=0 test above is non-vacuous:
        with flag=1 the kernel runs and out[perm[i]] = i."""
        import cupy as cp
        from mdpy.core._rebuild_kernels import compile_rebuild_kernels

        N = 4
        perm = cp.array([3, 1, 0, 2], dtype=cp.int32)
        out = cp.full(N, -777, dtype=cp.int32)
        flag = cp.array([1], dtype=cp.int32)

        rk = compile_rebuild_kernels()
        rk["compose_perm"]((1,), (N,), (out, perm, np.int32(N), flag))
        cp.cuda.Device().synchronize()

        # out[perm[i]] = i => out[3]=0, out[1]=1, out[0]=2, out[2]=3
        np.testing.assert_array_equal(cp.asnumpy(out), [2, 1, 3, 0])


class TestExclusionPairsFlagGuard:
    """permute_exclusion_pairs_gpu's three internal kernels (permute_pairs,
    count_row, scatter_pairs) must each skip when d_rebuild_flag=0.

    Without the guards, a no-rebuild step would double-permute the cached
    exclusion pair indices and rewrite the CSR layout from stale data.
    """

    def test_flag_zero_skips_all_three_kernels(self):
        """When flag=0, permute_pairs and scatter_pairs must not write to
        their output buffers, and count_row must not atomicAdd to d_count.

        Verified by pre-filling the pool's kernel-output buffers with
        sentinels: if any kernel ran it would overwrite the sentinel.

        Note: d_count is zeroed by _excl_get(fill=0) and d_offset is then
        written by cp.cumsum — both cupy ops that run regardless of flag.
        With flag=0, count_row skips so d_count stays 0 and d_offset is
        cumsum-of-zeros. With flag=1, count_row would make d_count nonzero.
        """
        import cupy as cp

        num_particles = 5
        num_pairs = 3
        cached_i = cp.array([0, 1, 2], dtype=cp.int32)
        cached_j = cp.array([1, 2, 3], dtype=cp.int32)
        cached_scale = cp.array([0.0, 0.0, 0.0], dtype=cp.float32)
        composed_perm = cp.array([2, 0, 1, 3, 4], dtype=cp.int32)

        sentinel_i = -888
        sentinel_f = -777.0

        # Pre-fill the pool's kernel-output buffers with sentinels.
        # _excl_get returns these as-is (no fill) when they already exist.
        pool = {
            ("new_i", np.int32): cp.full(num_pairs, sentinel_i, dtype=cp.int32),
            ("new_j", np.int32): cp.full(num_pairs, sentinel_i, dtype=cp.int32),
            ("new_scale", np.float32): cp.full(num_pairs, sentinel_f, dtype=cp.float32),
            ("neighbors", np.int32): cp.full(num_pairs, sentinel_i, dtype=cp.int32),
            ("scale_out", np.float32): cp.full(num_pairs, sentinel_f, dtype=cp.float32),
        }

        flag = cp.array([0], dtype=cp.int32)
        d_offset, d_neighbors, d_scale_out, d_new_i, d_new_j, d_new_scale = \
            permute_exclusion_pairs_gpu(
                cached_i, cached_j, cached_scale,
                composed_perm, num_particles, pool, flag,
            )

        # permute_pairs skipped: new_i/j/scale keep sentinels
        assert np.all(cp.asnumpy(d_new_i) == sentinel_i), (
            "flag=0 permute_pairs wrote to d_new_i — guard failed")
        assert np.all(cp.asnumpy(d_new_j) == sentinel_i), (
            "flag=0 permute_pairs wrote to d_new_j — guard failed")
        assert np.all(cp.asnumpy(d_new_scale) == sentinel_f), (
            "flag=0 permute_pairs wrote to d_new_scale — guard failed")
        # scatter_pairs skipped: neighbors/scale_out keep sentinels
        assert np.all(cp.asnumpy(d_neighbors) == sentinel_i), (
            "flag=0 scatter_pairs wrote to d_neighbors — guard failed")
        assert np.all(cp.asnumpy(d_scale_out) == sentinel_f), (
            "flag=0 scatter_pairs wrote to d_scale_out — guard failed")
        # count_row skipped: d_count stayed 0 (zeroed by _excl_get),
        # so d_offset = cumsum(zeros) = all zeros. If count_row had run,
        # d_count would be nonzero and d_offset would reflect row counts.
        offset_np = cp.asnumpy(d_offset)
        assert np.all(offset_np == 0), (
            "flag=0 count_row wrote to d_count — guard failed "
            f"(d_offset={offset_np.tolist()})"
        )

    def test_flag_one_produces_correct_output(self):
        """Sanity check proving the flag=0 test is non-vacuous: with
        flag=1 all three kernels run and produce the correctly-permuted,
        CSR-sorted exclusion layout."""
        import cupy as cp

        num_particles = 5
        cached_i = cp.array([0, 1, 2], dtype=cp.int32)
        cached_j = cp.array([1, 2, 3], dtype=cp.int32)
        cached_scale = cp.array([0.0, 0.0, 0.0], dtype=cp.float32)
        composed_perm = cp.array([2, 0, 1, 3, 4], dtype=cp.int32)

        pool = {}
        flag = cp.array([1], dtype=cp.int32)
        d_offset, d_neighbors, d_scale_out, d_new_i, d_new_j, d_new_scale = \
            permute_exclusion_pairs_gpu(
                cached_i, cached_j, cached_scale,
                composed_perm, num_particles, pool, flag,
            )

        # permute_pairs: new_i[k] = perm[cached_i[k]]
        np.testing.assert_array_equal(cp.asnumpy(d_new_i), [2, 0, 1])
        np.testing.assert_array_equal(cp.asnumpy(d_new_j), [0, 1, 3])
        # count_row: rows 0,1,2 each have 1 pair
        # offset = cumsum([0,1,1,1,0,0]) = [0,1,2,3,3,3]
        np.testing.assert_array_equal(
            cp.asnumpy(d_offset), [0, 1, 2, 3, 3, 3])
        # scatter_pairs: neighbors sorted by row (row0->j=1, row1->j=3, row2->j=0)
        np.testing.assert_array_equal(cp.asnumpy(d_neighbors), [1, 3, 0])


class TestGatherSortedFlagGuard:
    """gather_sorted_kernel (used by NonbondedForce._gather_per_particle)
    must skip entirely when d_rebuild_flag=0.

    This kernel is only called from bind_sorted() which only runs during
    rebuilds, so guarding it is safe. (pack_sorted_posq is intentionally
    NOT guarded -- it is shared with the per-step _refresh_posq refresh.)
    """

    def test_flag_zero_skips_gather(self):
        """When flag=0 the kernel must not write to dst. Verified by
        pre-filling dst with a sentinel: if the kernel ran it would
        overwrite the sentinel with src[block_atoms[idx]]."""
        import cupy as cp
        from mdpy.force.nonbonded_force import _GATHER_SORTED_KERNEL_SRC

        num_particles = 8
        total_slots = 32
        src = cp.arange(num_particles, dtype=cp.float32)
        block_atoms = cp.zeros(total_slots, dtype=cp.int32)
        sentinel = -777.0
        dst = cp.full(total_slots, sentinel, dtype=cp.float32)
        flag = cp.array([0], dtype=cp.int32)

        kernel = cp.RawKernel(_GATHER_SORTED_KERNEL_SRC, "gather_sorted_kernel")
        tpb = 256
        grid = ((total_slots + tpb - 1) // tpb,)
        kernel(grid, (tpb,), (src, block_atoms, np.int32(total_slots),
                              np.int32(num_particles), dst, flag))
        cp.cuda.Device().synchronize()

        dst_np = cp.asnumpy(dst)
        assert np.all(dst_np == sentinel), (
            "flag=0 gather_sorted wrote to dst -- guard failed"
        )

    def test_flag_one_actually_gathers(self):
        """Sanity check proving the flag=0 test above is non-vacuous:
        with flag=1 the kernel runs and dst[idx] = src[block_atoms[idx]]."""
        import cupy as cp
        from mdpy.force.nonbonded_force import _GATHER_SORTED_KERNEL_SRC

        num_particles = 8
        total_slots = 32
        src = cp.arange(num_particles, dtype=cp.float32)
        block_atoms_np = np.full(total_slots, -1, dtype=np.int32)
        block_atoms_np[:num_particles] = np.arange(num_particles)[::-1]
        block_atoms = cp.asarray(block_atoms_np)
        dst = cp.full(total_slots, -777.0, dtype=cp.float32)
        flag = cp.array([1], dtype=cp.int32)

        kernel = cp.RawKernel(_GATHER_SORTED_KERNEL_SRC, "gather_sorted_kernel")
        tpb = 256
        grid = ((total_slots + tpb - 1) // tpb,)
        kernel(grid, (tpb,), (src, block_atoms, np.int32(total_slots),
                              np.int32(num_particles), dst, flag))
        cp.cuda.Device().synchronize()

        dst_np = cp.asnumpy(dst)
        expected = np.zeros(total_slots, dtype=np.float32)
        expected[:num_particles] = np.arange(num_particles)[::-1]
        np.testing.assert_array_equal(dst_np, expected)


class TestRemapIndicesFlagGuard:
    """remap_indices_kernel (file-local copies in force/_utils.py,
    constraint/settle.py, constraint/lincs.py) must skip entirely when
    d_rebuild_flag=0.

    Without the guard, calling remap_indices_gpu on a no-rebuild step
    would re-apply the old permutation to already-remapped index arrays,
    double-permuting the bonded/constraint atom indices.
    """

    @pytest.mark.parametrize("module_name", [
        "mdpy.force._utils",
        "mdpy.constraint.settle",
        "mdpy.constraint.lincs",
    ])
    def test_flag_zero_skips_remap(self, module_name):
        """When flag=0 the kernel must not write to d_indices. Verified
        by pre-filling d_indices with [0,1,...,N-1] and a reverse remap:
        an unguarded kernel would overwrite with [N-1,...,1,0]."""
        import cupy as cp
        import importlib

        mod = importlib.import_module(module_name)
        kernel = cp.RawKernel(mod._REMAP_INDICES_KERNEL, "remap_indices_kernel")

        N = 8
        src_np = np.arange(N, dtype=np.int32)
        d_remap = cp.asarray(src_np[::-1].copy())
        d_indices = cp.asarray(src_np.copy())
        flag = cp.array([0], dtype=cp.int32)

        tpb = 256
        grid = ((N + tpb - 1) // tpb,)
        kernel(grid, (tpb,), (d_remap, d_indices, np.int32(N), flag))
        cp.cuda.Device().synchronize()

        np.testing.assert_array_equal(cp.asnumpy(d_indices), src_np, (
            f"flag=0 {module_name} remap_indices_kernel wrote to d_indices "
            f"— guard failed (double-permute corruption)"
        ))

    @pytest.mark.parametrize("module_name", [
        "mdpy.force._utils",
        "mdpy.constraint.settle",
        "mdpy.constraint.lincs",
    ])
    def test_flag_one_actually_remaps(self, module_name):
        """Sanity check proving the flag=0 test above is non-vacuous:
        with flag=1 the kernel runs and d_indices[i] = d_remap[d_indices[i]]."""
        import cupy as cp
        import importlib

        mod = importlib.import_module(module_name)
        kernel = cp.RawKernel(mod._REMAP_INDICES_KERNEL, "remap_indices_kernel")

        N = 8
        src_np = np.arange(N, dtype=np.int32)
        d_remap = cp.asarray(src_np[::-1].copy())
        d_indices = cp.asarray(src_np.copy())
        flag = cp.array([1], dtype=cp.int32)

        tpb = 256
        grid = ((N + tpb - 1) // tpb,)
        kernel(grid, (tpb,), (d_remap, d_indices, np.int32(N), flag))
        cp.cuda.Device().synchronize()

        np.testing.assert_array_equal(cp.asnumpy(d_indices), src_np[::-1], (
            f"flag=1 {module_name} remap_indices_kernel did not apply the remap"
        ))
