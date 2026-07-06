import numpy as np
import cupy as cp
import pytest
from mdpy.core.topology import Builder
from mdpy.core.block_list import (
    BlockList, BLOCK_SIZE, NUM_ATOMS_SENTINEL, SCAN_BLOCK,
    _COMPOSITE_PREFIX_SUM_KERNEL, _CELL_PREFIX_SUM_KERNEL,
)


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


def _make_pbc(box):
    return np.eye(3, dtype=np.float32) * box


def _rebuild_and_build_block_pairs(n, box=50.0, cutoff=10.0, skin=2.0, seed=42, positions=None):
    if positions is None:
        positions = _make_positions(n, box, seed)
    topology = _make_topology(n)
    pbc_matrix = _make_pbc(box)
    pbc_inv = np.linalg.inv(pbc_matrix)
    bl = BlockList(cutoff=cutoff, skin=skin)
    bl.rebuild(positions, topology, pbc_matrix, pbc_inv, force=True)
    bl.build_block_pairs(topology, pbc_matrix)
    # Read actual counts for test assertions (syncs — acceptable in tests,
    # NOT in the hot path where kernels read from device directly).
    bl.num_blocks = int(bl._d_num_blocks[0].get())
    bl.num_block_pairs = int(bl._d_counters[0].get())
    return bl, positions, pbc_matrix, pbc_inv, topology


class TestCellAssignment:

    def test_cell_grid_dimensions(self):
        n, box = 100, 50.0
        cutoff, skin = 10.0, 2.0
        bl, *_ = _rebuild_and_build_block_pairs(n, box, cutoff, skin)
        expected_nc = int(box / (cutoff + skin))
        assert bl.nc_x == expected_nc
        assert bl.nc_y == expected_nc
        assert bl.nc_z == expected_nc
        assert bl.nc_total == expected_nc ** 3

    def test_block_coverage(self):
        n, box = 100, 50.0
        bl, *_ = _rebuild_and_build_block_pairs(n, box)
        ba = bl.block_atoms.ravel()
        real_atoms = ba[ba >= 0]
        unique, counts = np.unique(real_atoms, return_counts=True)
        assert len(unique) == n
        assert np.all(counts == 1)

    def test_blocks_are_cell_aligned(self):
        n, box = 100, 50.0
        bl, *_ = _rebuild_and_build_block_pairs(n, box)
        atom_to_block = cp.asnumpy(bl.d_atom_to_block)
        ba = bl.block_atoms
        for bi in range(bl.num_blocks):
            for slot in range(BLOCK_SIZE):
                atom_id = ba[bi, slot]
                if atom_id >= 0:
                    assert atom_to_block[atom_id] == bi, (
                        f"atom {atom_id} in block {bi} slot {slot} "
                        f"but d_atom_to_block[{atom_id}]={atom_to_block[atom_id]}"
                    )

    def test_atom_to_block_mapping(self):
        n, box = 100, 50.0
        bl, *_ = _rebuild_and_build_block_pairs(n, box)
        atom_to_block = cp.asnumpy(bl.d_atom_to_block)
        atom_to_slot = cp.asnumpy(bl.d_atom_to_slot)
        ba = bl.block_atoms
        for bi in range(bl.num_blocks):
            for slot in range(BLOCK_SIZE):
                atom_id = ba[bi, slot]
                if atom_id >= 0:
                    assert atom_to_block[atom_id] == bi
                    assert atom_to_slot[atom_id] == slot

    def test_sort_order_is_correct(self):
        n, box = 100, 50.0
        bl, *_ = _rebuild_and_build_block_pairs(n, box)
        raw_order = cp.asnumpy(bl.d_raw_order)
        pdb_to_sorted = cp.asnumpy(bl.d_pdb_to_sorted)
        sorted_to_pdb = cp.asnumpy(bl.d_sorted_to_pdb)
        assert len(raw_order) == n
        assert len(pdb_to_sorted) == n
        assert len(sorted_to_pdb) == n
        for pdb_i in range(n):
            sorted_i = pdb_to_sorted[pdb_i]
            assert sorted_to_pdb[sorted_i] == pdb_i
        for sorted_i in range(n):
            pdb_i = sorted_to_pdb[sorted_i]
            assert raw_order[sorted_i] == pdb_i


class TestInteractingBlocks:

    def test_block_pairs_not_empty(self):
        n, box = 100, 50.0
        bl, *_ = _rebuild_and_build_block_pairs(n, box, cutoff=10.0, skin=2.0)
        assert bl.num_block_pairs > 0
        block_pairs = bl.block_pairs
        interacting = bl.interacting_atoms
        assert block_pairs.shape == (bl.num_block_pairs,)
        assert interacting.shape == (bl.num_block_pairs, BLOCK_SIZE)

    def test_interacting_atoms_within_cutoff(self):
        n, box = 100, 50.0
        cutoff, skin = 10.0, 2.0
        bl, positions, pbc_matrix, _, _ = _rebuild_and_build_block_pairs(n, box, cutoff, skin)
        sorted_to_pdb = cp.asnumpy(bl.d_sorted_to_pdb)
        block_pairs = bl.block_pairs
        interacting = bl.interacting_atoms
        ba = bl.block_atoms
        build_radius_sq = bl.build_radius ** 2
        pbc_2d = pbc_matrix.reshape(3, 3)
        box_diag = np.array([pbc_2d[0, 0], pbc_2d[1, 1], pbc_2d[2, 2]])

        for ti in range(bl.num_block_pairs):
            source_block = block_pairs[ti]
            interacting_row = interacting[ti]
            source_atoms = ba[source_block]
            for slot in range(BLOCK_SIZE):
                aj = interacting_row[slot]
                if aj < 0 or aj == NUM_ATOMS_SENTINEL:
                    continue
                pdb_j = sorted_to_pdb[aj]
                pos_j = positions[pdb_j]
                found_close = False
                for si in range(BLOCK_SIZE):
                    ak = source_atoms[si]
                    if ak < 0:
                        continue
                    pdb_k = sorted_to_pdb[ak]
                    pos_k = positions[pdb_k]
                    dx = pos_j - pos_k
                    dx -= box_diag * np.round(dx / box_diag)
                    dist_sq = np.sum(dx ** 2)
                    if dist_sq <= build_radius_sq:
                        found_close = True
                        break
                assert found_close, (
                    f"Block-pair {ti}: interacting atom sorted_idx={aj} pdb={pdb_j} "
                    f"is not within build_radius of any atom in source block {source_block}"
                )

    def test_pair_completeness(self):
        n, box = 100, 50.0
        cutoff, skin = 10.0, 2.0
        bl, positions, pbc_matrix, _, _ = _rebuild_and_build_block_pairs(n, box, cutoff, skin)
        sorted_to_pdb = cp.asnumpy(bl.d_sorted_to_pdb)
        block_pairs = bl.block_pairs
        interacting = bl.interacting_atoms
        ba = bl.block_atoms
        build_radius_sq = bl.build_radius ** 2
        pbc_2d = pbc_matrix.reshape(3, 3)
        box_diag = np.array([pbc_2d[0, 0], pbc_2d[1, 1], pbc_2d[2, 2]])

        found_pairs = set()
        for ti in range(bl.num_block_pairs):
            source_block = block_pairs[ti]
            source_atoms = ba[source_block]
            interacting_row = interacting[ti]
            for si in range(BLOCK_SIZE):
                ak = source_atoms[si]
                if ak < 0:
                    continue
                for sj in range(BLOCK_SIZE):
                    aj = interacting_row[sj]
                    if aj < 0 or aj == NUM_ATOMS_SENTINEL:
                        continue
                    if ak != aj:
                        found_pairs.add((min(ak, aj), max(ak, aj)))

        sorted_pos_x, sorted_pos_y, sorted_pos_z = bl._sorted_positions
        spx = cp.asnumpy(sorted_pos_x)
        spy = cp.asnumpy(sorted_pos_y)
        spz = cp.asnumpy(sorted_pos_z)

        missing = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = spx[j] - spx[i]
                dy = spy[j] - spy[i]
                dz = spz[j] - spz[i]
                dx -= box_diag[0] * round(dx / box_diag[0])
                dy -= box_diag[1] * round(dy / box_diag[1])
                dz -= box_diag[2] * round(dz / box_diag[2])
                dist_sq = dx * dx + dy * dy + dz * dz
                if dist_sq <= build_radius_sq:
                    assert (i, j) in found_pairs, (
                        f"Pair (sorted {i}, sorted {j}) within build_radius but not found in block_pairs"
                    )

    def test_newton_third_law_no_duplicates(self):
        n, box = 100, 50.0
        bl, *_ = _rebuild_and_build_block_pairs(n, box, cutoff=10.0, skin=2.0)
        block_pairs = bl.block_pairs
        interacting = bl.interacting_atoms
        seen = set()
        for ti in range(bl.num_block_pairs):
            source_block = block_pairs[ti]
            for sj in range(BLOCK_SIZE):
                aj = interacting[ti, sj]
                if aj < 0 or aj == NUM_ATOMS_SENTINEL:
                    continue
                pair = (source_block, aj)
                assert pair not in seen, (
                    f"Duplicate (block={source_block}, sorted_atom={aj}) in block_pairs"
                )
                seen.add(pair)

    def test_self_block_pair_small(self):
        n = 4
        box = 50.0
        cutoff, skin = 100.0, 10.0
        positions = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
        ], dtype=np.float32)
        bl, *_ = _rebuild_and_build_block_pairs(n, box, cutoff, skin, positions=positions)
        assert bl.num_blocks == 1
        assert bl.num_block_pairs > 0

    def test_interaction_block_pairs_cull(self):
        group_size = 32
        n = group_size * 4
        box = 500.0
        cutoff, skin = 5.0, 1.0
        positions = np.zeros((n, 3), dtype=np.float32)
        centers = [
            [50.0, 50.0, 50.0],
            [200.0, 200.0, 200.0],
            [350.0, 50.0, 350.0],
            [50.0, 350.0, 200.0],
        ]
        rng = np.random.RandomState(123)
        for g in range(4):
            for i in range(group_size):
                idx = g * group_size + i
                positions[idx] = np.array(centers[g]) + rng.uniform(-1, 1, 3)
        bl, *_ = _rebuild_and_build_block_pairs(n, box, cutoff, skin, positions=positions)
        assert bl.num_block_pairs < bl.num_blocks ** 2

    def test_shift_nonzero_for_cross_boundary_block_pairs(self):
        n = 64
        box = 20.0
        cutoff, skin = 4.0, 1.0
        rng = np.random.RandomState(42)
        positions = np.zeros((n, 3), dtype=np.float32)
        positions[:32, 0] = 1.0
        positions[32:, 0] = box - 1.0
        positions[:, 1] = 12.5 + rng.uniform(-0.1, 0.1, n)
        positions[:, 2] = 12.5 + rng.uniform(-0.1, 0.1, n)
        bl, *_ = _rebuild_and_build_block_pairs(n, box, cutoff, skin, positions=positions)

        shift_x = cp.asnumpy(bl.d_block_pair_shift_x[:bl.num_block_pairs])
        shift_y = cp.asnumpy(bl.d_block_pair_shift_y[:bl.num_block_pairs])
        shift_z = cp.asnumpy(bl.d_block_pair_shift_z[:bl.num_block_pairs])

        total_shift = np.sum(np.abs(shift_x)) + np.sum(np.abs(shift_y)) + np.sum(np.abs(shift_z))
        assert total_shift > 0, "Expected at least some block pairs with nonzero PBC shift"

        block_pairs = cp.asnumpy(bl.d_block_pairs[:bl.num_block_pairs])
        int_atoms = cp.asnumpy(bl.d_interacting_atoms[:bl.num_block_pairs * 32]).reshape(bl.num_block_pairs, 32)
        block_atoms_np = cp.asnumpy(bl.d_block_atoms).reshape(-1, 32)

        for t in range(bl.num_block_pairs):
            bx = block_pairs[t]
            j_atoms = int_atoms[t]
            all_from_self = True
            for a in j_atoms:
                if a < 0 or a == NUM_ATOMS_SENTINEL:
                    continue
                found_in_self = False
                for s in range(32):
                    if block_atoms_np[bx, s] == a:
                        found_in_self = True
                        break
                if not found_in_self:
                    all_from_self = False
                    break
            if all_from_self:
                assert shift_x[t] == 0.0 and shift_y[t] == 0.0 and shift_z[t] == 0.0, \
                    f"Self-block-pair {t} should have zero shift"

    def test_shift_distance_matches_roundf(self):
        n = 64
        box = 20.0
        cutoff, skin = 4.0, 1.0
        rng = np.random.RandomState(42)
        positions = np.zeros((n, 3), dtype=np.float32)
        positions[:32, 0] = 1.0
        positions[32:, 0] = box - 1.0
        positions[:, 1] = 12.5 + rng.uniform(-0.1, 0.1, n)
        positions[:, 2] = 12.5 + rng.uniform(-0.1, 0.1, n)
        bl, *_ = _rebuild_and_build_block_pairs(n, box, cutoff, skin, positions=positions)

        pos_x = cp.asnumpy(bl._sorted_positions[0])
        pos_y = cp.asnumpy(bl._sorted_positions[1])
        pos_z = cp.asnumpy(bl._sorted_positions[2])
        pbc_2d = cp.asnumpy(bl._d_pbc_matrix).reshape(3, 3)
        box_x = float(pbc_2d[0, 0])
        box_y = float(pbc_2d[1, 1])
        box_z = float(pbc_2d[2, 2])

        block_pairs = cp.asnumpy(bl.d_block_pairs[:bl.num_block_pairs])
        shift_x = cp.asnumpy(bl.d_block_pair_shift_x[:bl.num_block_pairs])
        shift_y = cp.asnumpy(bl.d_block_pair_shift_y[:bl.num_block_pairs])
        shift_z = cp.asnumpy(bl.d_block_pair_shift_z[:bl.num_block_pairs])
        int_atoms = cp.asnumpy(bl.d_interacting_atoms[:bl.num_block_pairs * 32]).reshape(bl.num_block_pairs, 32)
        block_atoms_np = cp.asnumpy(bl.d_block_atoms).reshape(-1, 32)

        max_err = 0.0
        for t in range(bl.num_block_pairs):
            bx = block_pairs[t]
            sx, sy, sz = shift_x[t], shift_y[t], shift_z[t]
            for lane in range(32):
                gj = int_atoms[t, lane]
                if gj < 0 or gj == NUM_ATOMS_SENTINEL:
                    continue
                xj, yj, zj = pos_x[gj], pos_y[gj], pos_z[gj]
                for k in range(32):
                    gi = block_atoms_np[bx, k]
                    if gi < 0:
                        continue
                    xi, yi, zi = pos_x[gi], pos_y[gi], pos_z[gi]

                    dx_r = xj - xi
                    dx_r -= box_x * round(dx_r / box_x)
                    dy_r = yj - yi
                    dy_r -= box_y * round(dy_r / box_y)
                    dz_r = zj - zi
                    dz_r -= box_z * round(dz_r / box_z)

                    dx_s = (xj + sx) - xi
                    dy_s = (yj + sy) - yi
                    dz_s = (zj + sz) - zi

                    err = max(abs(dx_r - dx_s), abs(dy_r - dy_s), abs(dz_r - dz_s))
                    max_err = max(max_err, err)

        assert max_err < 1e-5, f"Shift distance error {max_err} exceeds tolerance"


class TestPBCHandling:

    def test_cross_boundary_block_pairs(self):
        n = 64
        box = 50.0
        cutoff, skin = 12.0, 2.0
        positions = np.zeros((n, 3), dtype=np.float32)
        positions[:32, 0] = 1.0
        positions[32:, 0] = box - 1.0
        rng = np.random.RandomState(99)
        positions[:, 1] = rng.uniform(0, box, n)
        positions[:, 2] = rng.uniform(0, box, n)
        bl, *_ = _rebuild_and_build_block_pairs(n, box, cutoff, skin, positions=positions)
        assert bl.num_block_pairs > 0
        sorted_to_pdb = cp.asnumpy(bl.d_sorted_to_pdb)
        ba = bl.block_atoms
        block_pairs = bl.block_pairs
        interacting = bl.interacting_atoms
        has_cross = False
        for ti in range(bl.num_block_pairs):
            source_block = block_pairs[ti]
            source_pdbs = set()
            for si in range(BLOCK_SIZE):
                ak = ba[source_block, si]
                if ak >= 0:
                    source_pdbs.add(sorted_to_pdb[ak])
            source_left = any(positions[p, 0] < box / 2 for p in source_pdbs)
            source_right = any(positions[p, 0] > box / 2 for p in source_pdbs)
            for sj in range(BLOCK_SIZE):
                aj = interacting[ti, sj]
                if aj < 0 or aj == NUM_ATOMS_SENTINEL:
                    continue
                pdb_j = sorted_to_pdb[aj]
                inter_left = positions[pdb_j, 0] < box / 2
                inter_right = positions[pdb_j, 0] > box / 2
                if (source_left and inter_right) or (source_right and inter_left):
                    raw_xdiff = abs(positions[pdb_j, 0] - positions[next(iter(source_pdbs)), 0])
                    if raw_xdiff > box / 2:
                        has_cross = True
                        break
            if has_cross:
                break
        assert has_cross, "No cross-boundary block pairs found when expected"


class TestExclusionMasks:

    def test_bond_excluded(self):
        n = 10
        builder = Builder()
        builder.set_particles(
            masses=np.ones(n, dtype=np.float32),
            charges=np.zeros(n, dtype=np.float32),
            particle_types=np.zeros(n, dtype=np.int32),
        )
        for i in range(n - 1):
            builder.add_bond(i, i + 1, k=300.0, r0=1.5)
        builder.build_exclusion_map()
        topology, _ = builder.build()
        positions = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            positions[i] = [i * 1.0, 0, 0]
        box = 20.0
        pbc_matrix = _make_pbc(box)
        pbc_inv = np.linalg.inv(pbc_matrix)
        bl = BlockList(cutoff=10.0, skin=2.0)
        bl.rebuild(positions, topology, pbc_matrix, pbc_inv, force=True)
        bl.build_block_pairs(topology, pbc_matrix)
        bl.num_blocks = int(bl._d_num_blocks[0].get())
        bl.num_block_pairs = int(bl._d_counters[0].get())

        excl = bl.exclusion_masks
        ia = bl.interacting_atoms
        atb = cp.asnumpy(bl.d_atom_to_block)
        ats = cp.asnumpy(bl.d_atom_to_slot)
        for t in range(bl.num_block_pairs):
            bx = bl.block_pairs[t]
            for sj in range(BLOCK_SIZE):
                aj = ia[t, sj]
                if aj < 0 or aj >= n:
                    continue
                if atb[aj] != bx:
                    continue
                mask = excl[t, sj]
                slot_in_block = ats[aj]
                for s in range(slot_in_block + 1):
                    assert (mask >> s) & 1, f"triangle bit {s} missing"

    def test_dihedral_14_fully_excluded(self):
        n = 10
        builder = Builder()
        builder.set_particles(
            masses=np.ones(n, dtype=np.float32),
            charges=np.zeros(n, dtype=np.float32),
            particle_types=np.zeros(n, dtype=np.int32),
        )
        for i in range(n - 1):
            builder.add_bond(i, i + 1, k=300.0, r0=1.5)
        for i in range(n - 2):
            builder.add_angle(i, i + 1, i + 2, force_constant=50.0, equilibrium_angle=1.9)
        for i in range(n - 3):
            builder.add_dihedral(i, i + 1, i + 2, i + 3, force_constant=0.5, periodicity=3, phase=0.0)
        builder.build_exclusion_map()
        topology, _ = builder.build()
        positions = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            positions[i] = [i * 1.0, 0, 0]
        box = 20.0
        pbc_matrix = _make_pbc(box)
        pbc_inv = np.linalg.inv(pbc_matrix)
        bl = BlockList(cutoff=10.0, skin=2.0)
        bl.rebuild(positions, topology, pbc_matrix, pbc_inv, force=True)
        bl.build_block_pairs(topology, pbc_matrix)
        bl.num_blocks = int(bl._d_num_blocks[0].get())
        bl.num_block_pairs = int(bl._d_counters[0].get())

        excl = bl.exclusion_masks
        ia = bl.interacting_atoms
        atb = cp.asnumpy(bl.d_atom_to_block)
        ats = cp.asnumpy(bl.d_atom_to_slot)
        for t in range(bl.num_block_pairs):
            bx = bl.block_pairs[t]
            for sj in range(BLOCK_SIZE):
                aj = ia[t, sj]
                if aj < 0 or aj >= n:
                    continue
                if atb[aj] != bx:
                    continue
                slot_aj = ats[aj]
                for pair_a, pair_b in [(0, 3), (1, 4), (2, 5)]:
                    if aj == pair_a and atb[pair_b] == bx:
                        slot_b = ats[pair_b]
                        assert (excl[t, sj] >> slot_b) & 1, (
                            f"1-4 pair ({pair_a},{pair_b}) should be fully excluded"
                        )


class TestBlockPairClassification:

    def test_classify_splits_block_pairs(self):
        n = 10
        builder = Builder()
        builder.set_particles(
            masses=np.ones(n, dtype=np.float32),
            charges=np.zeros(n, dtype=np.float32),
            particle_types=np.zeros(n, dtype=np.int32),
        )
        for i in range(n - 1):
            builder.add_bond(i, i + 1, k=300.0, r0=1.5)
        builder.build_exclusion_map()
        topology, _ = builder.build()
        positions = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            positions[i] = [i * 1.0, 0, 0]
        box = 20.0
        pbc_matrix = _make_pbc(box)
        pbc_inv = np.linalg.inv(pbc_matrix)
        bl = BlockList(cutoff=10.0, skin=2.0)
        bl.rebuild(positions, topology, pbc_matrix, pbc_inv, force=True)
        bl.build_block_pairs(topology, pbc_matrix)

        assert bl.num_main_block_pairs + bl.num_exclusion_block_pairs == bl.num_block_pairs
        if bl.num_exclusion_block_pairs > 0:
            assert bl.d_excl_exclusion_masks.size >= bl.num_exclusion_block_pairs * BLOCK_SIZE

    def test_no_exclusion_all_main(self):
        n = 20
        topology = _make_topology(n)
        positions = _make_positions(n, 30.0, seed=11)
        pbc_matrix = _make_pbc(30.0)
        pbc_inv = np.linalg.inv(pbc_matrix)
        bl = BlockList(cutoff=8.0, skin=2.0)
        bl.rebuild(positions, topology, pbc_matrix, pbc_inv, force=True)
        bl.build_block_pairs(topology, pbc_matrix)

        assert bl.num_main_block_pairs + bl.num_exclusion_block_pairs == bl.num_block_pairs


class TestCheckRebuild:

    def test_check_rebuild(self):
        topology = _make_topology(4)
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ], dtype=np.float32)
        pbc_matrix = _make_pbc(50.0)
        pbc_inv = np.linalg.inv(pbc_matrix)
        bl = BlockList(cutoff=10.0, skin=2.0)
        bl.rebuild(positions, topology, pbc_matrix, pbc_inv, force=True)
        bl.capture_snapshot((
            cp.asarray(np.ascontiguousarray(positions[:, 0], dtype=np.float32)),
            cp.asarray(np.ascontiguousarray(positions[:, 1], dtype=np.float32)),
            cp.asarray(np.ascontiguousarray(positions[:, 2], dtype=np.float32)),
        ))

        for _ in range(19):
            assert not bl.check_rebuild(positions)

        moved = positions.copy()
        moved[0, 0] += 2.0
        assert bl.check_rebuild(moved)

    def test_check_rebuild_uninitialized(self):
        topology = _make_topology(4)
        positions = np.zeros((4, 3), dtype=np.float32)
        bl = BlockList(cutoff=10.0, skin=2.0)
        assert bl.check_rebuild(positions)

    def test_async_check_sticky_flag(self):
        n, box = 50, 50.0
        positions = _make_positions(n, box)
        topology = _make_topology(n)
        pbc_matrix = _make_pbc(box)
        pbc_inv = np.linalg.inv(pbc_matrix)
        bl = BlockList(cutoff=10.0, skin=2.0)
        bl.rebuild(positions, topology, pbc_matrix, pbc_inv, force=True)
        bl.capture_snapshot((
            cp.asarray(np.ascontiguousarray(positions[:, 0], dtype=np.float32)),
            cp.asarray(np.ascontiguousarray(positions[:, 1], dtype=np.float32)),
            cp.asarray(np.ascontiguousarray(positions[:, 2], dtype=np.float32)),
        ))

        pos_x = bl.d_positions_at_rebuild_x.copy()
        pos_y = bl.d_positions_at_rebuild_y.copy()
        pos_z = bl.d_positions_at_rebuild_z.copy()

        bl.d_rebuild_flag[0] = 0
        pos_x[:5] += 3.0
        bl.check_rebuild_async((pos_x, pos_y, pos_z))
        cp.cuda.Stream.null.synchronize()
        assert int(bl.d_rebuild_flag[0]) == 1

        pos_x[:5] -= 3.0
        bl.check_rebuild_async((pos_x, pos_y, pos_z))
        cp.cuda.Stream.null.synchronize()
        assert int(bl.d_rebuild_flag[0]) == 1, "flag must stay sticky"

    def test_capture_snapshot_stores_provided_positions(self):
        n, box = 50, 50.0
        positions = _make_positions(n, box)
        topology = _make_topology(n)
        pbc_matrix = _make_pbc(box)
        pbc_inv = np.linalg.inv(pbc_matrix)
        bl = BlockList(cutoff=10.0, skin=2.0)
        bl.rebuild(positions, topology, pbc_matrix, pbc_inv, force=True)

        wrapped = positions.copy()
        wrapped[0, 0] = 0.3
        wrapped[1, 1] = 0.7
        pos_x = cp.asarray(np.ascontiguousarray(wrapped[:, 0], dtype=np.float32))
        pos_y = cp.asarray(np.ascontiguousarray(wrapped[:, 1], dtype=np.float32))
        pos_z = cp.asarray(np.ascontiguousarray(wrapped[:, 2], dtype=np.float32))
        bl.capture_snapshot((pos_x, pos_y, pos_z))

        snap_x = cp.asnumpy(bl.d_positions_at_rebuild_x)
        snap_y = cp.asnumpy(bl.d_positions_at_rebuild_y)
        snap_z = cp.asnumpy(bl.d_positions_at_rebuild_z)
        assert snap_x[0] == pytest.approx(0.3, abs=1e-5)
        assert snap_y[1] == pytest.approx(0.7, abs=1e-5)
        assert snap_z.shape == (n,)

    def test_post_wrap_snapshot_no_false_trigger(self):
        """Regression: snapshot captured AFTER wrap must not false-trigger.

        Before fix: snapshot held pre-wrap positions (counting_scatter ran
        before wrap). A particle at L+0.2 wrapped to 0.2, but snapshot
        stayed at L+0.2. Next check_rebuild saw displacement ~= L.
        After fix: capture_snapshot is called AFTER wrap, snapshot holds 0.2.
        """
        n = 4
        box = 50.0
        cutoff, skin = 10.0, 2.0
        positions = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [3.0, 1.0, 1.0],
            [4.0, 1.0, 1.0],
        ], dtype=np.float32)
        topology = _make_topology(n)
        pbc_matrix = _make_pbc(box)
        pbc_inv = np.linalg.inv(pbc_matrix)
        bl = BlockList(cutoff=cutoff, skin=skin)
        bl.rebuild(positions, topology, pbc_matrix, pbc_inv, force=True)
        bl.capture_snapshot((
            cp.asarray(positions[:, 0].astype(np.float32)),
            cp.asarray(positions[:, 1].astype(np.float32)),
            cp.asarray(positions[:, 2].astype(np.float32)),
        ))

        # Simulate: particle 0 drifts to box+0.2 (pre-wrap), rebuild, wrap to 0.2
        drifted = positions.copy()
        drifted[0, 0] = box + 0.2
        bl.rebuild(drifted, topology, pbc_matrix, pbc_inv, force=True)
        wrapped = drifted.copy()
        wrapped[0, 0] = 0.2
        bl.capture_snapshot((
            cp.asarray(np.ascontiguousarray(wrapped[:, 0], dtype=np.float32)),
            cp.asarray(np.ascontiguousarray(wrapped[:, 1], dtype=np.float32)),
            cp.asarray(np.ascontiguousarray(wrapped[:, 2], dtype=np.float32)),
        ))

        # Next step: particle 0 drifts 0.05 from wrapped position
        next_pos = wrapped.copy()
        next_pos[0, 0] = 0.25
        result = bl.check_rebuild((
            cp.asarray(np.ascontiguousarray(next_pos[:, 0], dtype=np.float32)),
            cp.asarray(np.ascontiguousarray(next_pos[:, 1], dtype=np.float32)),
            cp.asarray(np.ascontiguousarray(next_pos[:, 2], dtype=np.float32)),
        ))
        assert not result, (
            "check_rebuild false-triggered: snapshot likely holds pre-wrap "
            "position (box+0.2) instead of post-wrap (0.2)"
        )

    def test_read_flag_sync_returns_flag_value(self):
        n, box = 50, 50.0
        positions = _make_positions(n, box)
        topology = _make_topology(n)
        pbc_matrix = _make_pbc(box)
        pbc_inv = np.linalg.inv(pbc_matrix)
        bl = BlockList(cutoff=10.0, skin=2.0)
        bl.rebuild(positions, topology, pbc_matrix, pbc_inv, force=True)

        bl.d_rebuild_flag[0] = 0
        assert bl.read_flag_sync() == 0

        bl.d_rebuild_flag[0] = 1
        assert bl.read_flag_sync() == 1

    def test_reset_flag_clears_flag(self):
        n, box = 50, 50.0
        positions = _make_positions(n, box)
        topology = _make_topology(n)
        pbc_matrix = _make_pbc(box)
        pbc_inv = np.linalg.inv(pbc_matrix)
        bl = BlockList(cutoff=10.0, skin=2.0)
        bl.rebuild(positions, topology, pbc_matrix, pbc_inv, force=True)

        bl.d_rebuild_flag[0] = 1
        bl.reset_flag()
        cp.cuda.Stream.null.synchronize()
        assert int(bl.d_rebuild_flag[0]) == 0


class TestPostArgsortFusion:

    def test_post_argsort_matches_cupy(self):
        n, box = 1000, 50.0
        rng = np.random.RandomState(123)
        positions = rng.uniform(0, box, (n, 3)).astype(np.float32)
        topology = _make_topology(n)
        pbc_matrix = _make_pbc(box)
        pbc_inv = np.linalg.inv(pbc_matrix)
        bl = BlockList(cutoff=10.0, skin=2.0)
        bl.rebuild(positions, topology, pbc_matrix, pbc_inv, force=True)

        assert bl.d_raw_order.shape == (n,)
        assert bl.d_pdb_to_sorted.shape == (n,)
        assert bl.d_sorted_to_pdb.shape == (n,)

        raw = cp.asnumpy(bl.d_raw_order)
        p2s = cp.asnumpy(bl.d_pdb_to_sorted)
        s2p = cp.asnumpy(bl.d_sorted_to_pdb)

        assert np.all(p2s[raw] == np.arange(n))

        pos_x = cp.asnumpy(bl._sorted_positions[0])
        pos_y = cp.asnumpy(bl._sorted_positions[1])
        pos_z = cp.asnumpy(bl._sorted_positions[2])
        assert np.allclose(pos_x, positions[:, 0][raw], atol=1e-6)
        assert np.allclose(pos_y, positions[:, 1][raw], atol=1e-6)
        assert np.allclose(pos_z, positions[:, 2][raw], atol=1e-6)


class TestCellProcessingBatch:

    def test_cell_arrays_match_after_batch(self):
        n, box = 5000, 50.0
        rng = np.random.RandomState(456)
        positions = rng.uniform(0, box, (n, 3)).astype(np.float32)
        topology = _make_topology(n)
        pbc_matrix = _make_pbc(box)
        pbc_inv = np.linalg.inv(pbc_matrix)
        bl = BlockList(cutoff=10.0, skin=2.0)
        bl.rebuild(positions, topology, pbc_matrix, pbc_inv, force=True)

        assert bl.num_blocks > 0
        assert bl.d_cell_block_offset is not None
        assert bl.d_cell_block_count is not None
        assert bl.d_block_atoms is not None

        block_atoms_np = cp.asnumpy(bl.d_block_atoms)
        block_atoms_np = block_atoms_np.reshape(-1, 32)
        real_atoms = block_atoms_np[block_atoms_np >= 0]
        assert len(np.unique(real_atoms)) == n

        bl.build_block_pairs(topology, pbc_matrix)
        assert bl.num_block_pairs > 0


class TestBlockToCellExpand:

    def test_block_to_cell_correct(self):
        n, box = 5000, 50.0
        rng = np.random.RandomState(789)
        positions = rng.uniform(0, box, (n, 3)).astype(np.float32)
        topology = _make_topology(n)
        pbc_matrix = _make_pbc(box)
        pbc_inv = np.linalg.inv(pbc_matrix)
        bl = BlockList(cutoff=10.0, skin=2.0)
        bl.rebuild(positions, topology, pbc_matrix, pbc_inv, force=True)

        block_to_cell = cp.asnumpy(bl.d_block_to_cell)
        block_count = cp.asnumpy(bl.d_cell_block_count)
        block_offset = cp.asnumpy(bl.d_cell_block_offset)

        for c in range(bl.nc_total):
            for b in range(block_count[c]):
                bi = block_offset[c] + b
                assert block_to_cell[bi] == c, (
                    f"block {bi} should be in cell {c}, got {block_to_cell[bi]}"
                )


class TestCellSubsetDecomposition:

    def test_cell_subsets_gt1_for_small_system(self):
        bl, *_ = _rebuild_and_build_block_pairs(100, 50.0, cutoff=10.0, skin=2.0)
        assert bl.num_cell_subsets > 1, (
            f"Expected cell_subsets > 1 for small system, got {bl.num_cell_subsets}"
        )

    def test_interacting_atoms_within_cutoff_with_subset(self):
        n, box = 200, 50.0
        cutoff, skin = 10.0, 2.0
        bl, positions, pbc_matrix, _, _ = _rebuild_and_build_block_pairs(n, box, cutoff, skin)
        assert bl.num_cell_subsets > 1, "Test requires K > 1 to be meaningful"

        sorted_to_pdb = cp.asnumpy(bl.d_sorted_to_pdb)
        block_pairs = bl.block_pairs
        interacting = bl.interacting_atoms
        ba = bl.block_atoms
        build_radius_sq = bl.build_radius ** 2
        pbc_2d = pbc_matrix.reshape(3, 3)
        box_diag = np.array([pbc_2d[0, 0], pbc_2d[1, 1], pbc_2d[2, 2]])

        for ti in range(bl.num_block_pairs):
            source_block = block_pairs[ti]
            interacting_row = interacting[ti]
            source_atoms = ba[source_block]
            for slot in range(BLOCK_SIZE):
                aj = interacting_row[slot]
                if aj < 0 or aj == NUM_ATOMS_SENTINEL:
                    continue
                pdb_j = sorted_to_pdb[aj]
                pos_j = positions[pdb_j]
                found_close = False
                for si in range(BLOCK_SIZE):
                    ak = source_atoms[si]
                    if ak < 0:
                        continue
                    pdb_k = sorted_to_pdb[ak]
                    pos_k = positions[pdb_k]
                    dx = pos_j - pos_k
                    dx -= box_diag * np.round(dx / box_diag)
                    dist_sq = np.sum(dx ** 2)
                    if dist_sq <= build_radius_sq:
                        found_close = True
                        break
                assert found_close, (
                    f"Block-pair {ti}: interacting atom sorted_idx={aj} pdb={pdb_j} "
                    f"is not within build_radius of any atom in source block {source_block}"
                )

    def test_no_duplicate_self_pairs(self):
        n, box = 100, 50.0
        cutoff, skin = 10.0, 2.0
        bl, *_ = _rebuild_and_build_block_pairs(n, box, cutoff, skin)
        assert bl.num_cell_subsets > 1

        block_pairs = bl.block_pairs
        interacting = bl.interacting_atoms

        self_pair_count = 0
        for ti in range(bl.num_block_pairs):
            source_block = block_pairs[ti]
            source_atoms = set(int(a) for a in bl.block_atoms[source_block] if a >= 0)
            interacting_row = interacting[ti]
            interacting_set = set(
                int(a) for a in interacting_row
                if a >= 0 and a != NUM_ATOMS_SENTINEL
            )
            if source_atoms == interacting_set:
                self_pair_count += 1

        assert self_pair_count <= bl.num_blocks, (
            f"Found {self_pair_count} self-pairs but only {bl.num_blocks} blocks exist. "
            f"Self-interaction may be duplicated by cell-subset decomposition."
        )


class TestParallelPrefixSumKernels:
    """Direct RawKernel unit tests for the parallel prefix-sum kernels,
    isolated from the full rebuild pipeline."""

    def _run_composite(self, counts):
        import cupy as cp
        K = len(counts)
        d_counts = cp.asarray(counts)
        d_offset = cp.empty(K + 1, dtype=cp.int32)
        kernel = cp.RawKernel(_COMPOSITE_PREFIX_SUM_KERNEL, "composite_prefix_sum_kernel")
        kernel((1,), (SCAN_BLOCK,), (d_counts, np.int32(K), d_offset))
        return cp.asnumpy(d_offset)

    def test_composite_matches_numpy_large(self):
        rng = np.random.RandomState(0)
        counts = rng.randint(0, 8, size=32768).astype(np.int32)
        out = self._run_composite(counts)
        ref = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
        assert np.array_equal(out, ref)

    def test_composite_matches_numpy_small(self):
        # n < SCAN_BLOCK: many threads idle, exercises tile boundary
        rng = np.random.RandomState(1)
        counts = rng.randint(0, 5, size=5).astype(np.int32)
        out = self._run_composite(counts)
        ref = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
        assert np.array_equal(out, ref)

    def test_composite_single_element(self):
        counts = np.array([7], dtype=np.int32)
        out = self._run_composite(counts)
        assert np.array_equal(out, np.array([0, 7], dtype=np.int32))

    def test_composite_all_zero(self):
        counts = np.zeros(2048, dtype=np.int32)
        out = self._run_composite(counts)
        assert np.array_equal(out, np.zeros(2049, dtype=np.int32))

    def _run_cell(self, cell_counts):
        import cupy as cp
        nc_total = len(cell_counts)
        d_counts = cp.asarray(cell_counts)
        d_cell_offset = cp.empty(nc_total + 1, dtype=cp.int32)
        d_cell_block_offset = cp.empty(nc_total + 1, dtype=cp.int32)
        d_cell_block_count = cp.empty(nc_total, dtype=cp.int32)
        d_cell_offset_padded = cp.empty(nc_total + 1, dtype=cp.int32)
        cbc_ref = ((cell_counts + 31) // 32).astype(np.int32)
        total_blocks_ref = int(cbc_ref.sum())
        d_block_to_cell = cp.empty(max(1, total_blocks_ref), dtype=cp.int32)
        d_num_blocks = cp.zeros(1, dtype=cp.int32)
        d_total_padded = cp.zeros(1, dtype=cp.int32)
        kernel = cp.RawKernel(_CELL_PREFIX_SUM_KERNEL, "cell_prefix_sum_kernel")
        kernel(
            (1,), (SCAN_BLOCK,),
            (
                d_counts, np.int32(nc_total),
                d_cell_offset, d_cell_block_offset, d_cell_block_count,
                d_cell_offset_padded, d_block_to_cell, d_num_blocks, d_total_padded,
            ),
        )
        return {
            "cell_offset": cp.asnumpy(d_cell_offset),
            "cell_block_offset": cp.asnumpy(d_cell_block_offset),
            "cell_block_count": cp.asnumpy(d_cell_block_count),
            "cell_offset_padded": cp.asnumpy(d_cell_offset_padded),
            "block_to_cell": cp.asnumpy(d_block_to_cell)[:total_blocks_ref],
            "num_blocks": int(d_num_blocks[0].get()),
            "total_padded": int(d_total_padded[0].get()),
        }

    def _assert_cell_against_reference(self, cell_counts):
        res = self._run_cell(cell_counts)
        cbc = ((cell_counts + 31) // 32).astype(np.int32)
        ref = {
            "cell_offset": np.concatenate([[0], np.cumsum(cell_counts)]).astype(np.int32),
            "cell_block_offset": np.concatenate([[0], np.cumsum(cbc)]).astype(np.int32),
            "cell_block_count": cbc,
            "cell_offset_padded": (np.concatenate([[0], np.cumsum(cbc)]) * 32).astype(np.int32),
            "block_to_cell": np.repeat(np.arange(len(cell_counts)), cbc).astype(np.int32),
            "num_blocks": int(cbc.sum()),
            "total_padded": int(cbc.sum()) * 32,
        }
        for key in ref:
            assert np.array_equal(res[key], ref[key]), (
                f"mismatch in {key}:\n got={res[key]}\n ref={ref[key]}"
            )

    def test_cell_matches_reference_large(self):
        rng = np.random.RandomState(2)
        cell_counts = rng.randint(0, 200, size=5000).astype(np.int32)
        self._assert_cell_against_reference(cell_counts)

    def test_cell_matches_reference_small(self):
        rng = np.random.RandomState(3)
        cell_counts = rng.randint(0, 64, size=8).astype(np.int32)
        self._assert_cell_against_reference(cell_counts)

    def test_cell_matches_reference_empty_cells(self):
        # Many empty cells -> bc=0, exercises scatter binary search gaps.
        cell_counts = np.zeros(3000, dtype=np.int32)
        cell_counts[::17] = 5
        self._assert_cell_against_reference(cell_counts)

    def test_cell_single_block(self):
        # One non-empty cell with < 32 atoms -> exactly one block.
        cell_counts = np.zeros(10, dtype=np.int32)
        cell_counts[3] = 10
        self._assert_cell_against_reference(cell_counts)


class TestShiftGroupedPacking:
    """Verify that neighbor cells with the same PBC shift pack their
    j-atoms into shared tiles, producing fewer tiles with higher fill."""

    def test_interior_cells_merge_into_fewer_tiles(self):
        """Place atoms in a dense cluster at box center — all 27 neighbor
        cells share shift (0,0,0). Under shift-grouped packing, at least
        one tile should be nearly full (>=28)."""
        n = 128
        box = 60.0
        cutoff, skin = 4.0, 1.0
        rng = np.random.RandomState(42)
        positions = np.zeros((n, 3), dtype=np.float32)
        # Cluster around (30, 30, 30) — center of a 60 box,
        # far from any PBC boundary so all shifts are zero.
        positions[:] = 30.0 + rng.uniform(-3.0, 3.0, (n, 3))
        bl, *_ = _rebuild_and_build_block_pairs(n, box, cutoff, skin, positions=positions)

        int_atoms = cp.asnumpy(
            bl.d_interacting_atoms[:bl.num_block_pairs * 32]
        ).reshape(bl.num_block_pairs, 32)
        fills = np.sum(
            (int_atoms >= 0) & (int_atoms != NUM_ATOMS_SENTINEL),
            axis=1
        )
        max_fill = int(np.max(fills))
        assert max_fill >= 28, (
            f"Expected at least one tile with fill >= 28 "
            f"(shift-grouped packing of same-shift cells), "
            f"but max fill is {max_fill}. "
            f"Fills: {dict(zip(*np.unique(fills, return_counts=True)))}"
        )

    def test_all_atoms_in_tile_share_shift(self):
        """Every j-atom in a tile must be shift-compatible: applying the
        tile's shift must place it within build_radius of some i-atom."""
        n = 200
        box = 50.0
        cutoff, skin = 10.0, 2.0
        bl, positions, pbc_matrix, _, _ = _rebuild_and_build_block_pairs(
            n, box, cutoff, skin
        )

        sorted_to_pdb = cp.asnumpy(bl.d_sorted_to_pdb)
        block_pairs = bl.block_pairs
        interacting = bl.interacting_atoms
        ba = bl.block_atoms
        shift_x = cp.asnumpy(bl.d_block_pair_shift_x[:bl.num_block_pairs])
        shift_y = cp.asnumpy(bl.d_block_pair_shift_y[:bl.num_block_pairs])
        shift_z = cp.asnumpy(bl.d_block_pair_shift_z[:bl.num_block_pairs])
        build_radius_sq = bl.build_radius ** 2

        for ti in range(bl.num_block_pairs):
            bx = block_pairs[ti]
            sx = float(shift_x[ti])
            sy = float(shift_y[ti])
            sz = float(shift_z[ti])
            source_atoms = ba[bx]
            interacting_row = interacting[ti]
            for slot in range(BLOCK_SIZE):
                aj = int(interacting_row[slot])
                if aj < 0 or aj == NUM_ATOMS_SENTINEL:
                    continue
                pdb_j = sorted_to_pdb[aj]
                pos_j = positions[pdb_j]
                found_close = False
                for si in range(BLOCK_SIZE):
                    ak = source_atoms[si]
                    if ak < 0:
                        continue
                    pdb_k = sorted_to_pdb[ak]
                    pos_k = positions[pdb_k]
                    dx = (pos_j[0] + sx) - pos_k[0]
                    dy = (pos_j[1] + sy) - pos_k[1]
                    dz = (pos_j[2] + sz) - pos_k[2]
                    if dx * dx + dy * dy + dz * dz <= build_radius_sq:
                        found_close = True
                        break
                assert found_close, (
                    f"Tile {ti}: j-atom sorted={aj} (pdb={pdb_j}) "
                    f"with shift ({sx},{sy},{sz}) not within build_radius "
                    f"of any i-atom in block {bx}. "
                    f"Incompatible shifts may have been packed together."
                )

    def test_pair_completeness_shift_grouped(self):
        """Every atom pair within build_radius must appear in some tile."""
        n = 200
        box = 50.0
        cutoff, skin = 10.0, 2.0
        bl, positions, pbc_matrix, _, _ = _rebuild_and_build_block_pairs(
            n, box, cutoff, skin
        )

        block_pairs = bl.block_pairs
        interacting = bl.interacting_atoms
        ba = bl.block_atoms
        build_radius_sq = bl.build_radius ** 2
        pbc_2d = pbc_matrix.reshape(3, 3)
        box_diag = np.array([pbc_2d[0, 0], pbc_2d[1, 1], pbc_2d[2, 2]])

        found_pairs = set()
        for ti in range(bl.num_block_pairs):
            source_block = block_pairs[ti]
            source_atoms = ba[source_block]
            interacting_row = interacting[ti]
            for si in range(BLOCK_SIZE):
                ak = source_atoms[si]
                if ak < 0:
                    continue
                for sj in range(BLOCK_SIZE):
                    aj = int(interacting_row[sj])
                    if aj < 0 or aj == NUM_ATOMS_SENTINEL:
                        continue
                    if ak != aj:
                        found_pairs.add(
                            (min(int(ak), aj), max(int(ak), aj))
                        )

        spx = cp.asnumpy(bl._sorted_positions[0])
        spy = cp.asnumpy(bl._sorted_positions[1])
        spz = cp.asnumpy(bl._sorted_positions[2])

        for i in range(n):
            for j in range(i + 1, n):
                dx = spx[j] - spx[i]
                dy = spy[j] - spy[i]
                dz = spz[j] - spz[i]
                dx -= box_diag[0] * round(dx / box_diag[0])
                dy -= box_diag[1] * round(dy / box_diag[1])
                dz -= box_diag[2] * round(dz / box_diag[2])
                if dx * dx + dy * dy + dz * dz <= build_radius_sq:
                    assert (i, j) in found_pairs, (
                        f"Pair (sorted {i}, {j}) within build_radius "
                        f"but missing from block_pairs — "
                        f"cross-cell packing dropped it."
                    )

    def test_no_duplicate_pairs_shift_grouped(self):
        """No duplicate (source_block, j_atom) pairs."""
        n = 200
        box = 50.0
        cutoff, skin = 10.0, 2.0
        bl, *_ = _rebuild_and_build_block_pairs(n, box, cutoff, skin)
        block_pairs = bl.block_pairs
        interacting = bl.interacting_atoms
        seen = set()
        for ti in range(bl.num_block_pairs):
            source_block = int(block_pairs[ti])
            for sj in range(BLOCK_SIZE):
                aj = int(interacting[ti, sj])
                if aj < 0 or aj == NUM_ATOMS_SENTINEL:
                    continue
                pair = (source_block, aj)
                assert pair not in seen, (
                    f"Duplicate (block={source_block}, atom={aj}) — "
                    f"cross-cell packing created a duplicate."
                )
                seen.add(pair)

    def test_pbc_boundary_incompatible_shifts_separated(self):
        """Atoms near a PBC boundary: tiles spanning the boundary must
        have correct shifts. No tile should mix incompatible images."""
        n = 128
        box = 20.0
        cutoff, skin = 4.0, 1.0
        rng = np.random.RandomState(42)
        positions = np.zeros((n, 3), dtype=np.float32)
        positions[:64, 0] = 1.0 + rng.uniform(-0.5, 0.5, 64)
        positions[64:, 0] = box - 1.0 + rng.uniform(-0.5, 0.5, 64)
        positions[:, 1] = 10.0 + rng.uniform(-1.0, 1.0, n)
        positions[:, 2] = 10.0 + rng.uniform(-1.0, 1.0, n)
        bl, positions, pbc_matrix, _, _ = _rebuild_and_build_block_pairs(
            n, box, cutoff, skin, positions=positions
        )

        sorted_to_pdb = cp.asnumpy(bl.d_sorted_to_pdb)
        block_pairs = bl.block_pairs
        interacting = bl.interacting_atoms
        ba = bl.block_atoms
        shift_x = cp.asnumpy(bl.d_block_pair_shift_x[:bl.num_block_pairs])
        shift_y = cp.asnumpy(bl.d_block_pair_shift_y[:bl.num_block_pairs])
        shift_z = cp.asnumpy(bl.d_block_pair_shift_z[:bl.num_block_pairs])
        build_radius_sq = bl.build_radius ** 2

        for ti in range(bl.num_block_pairs):
            bx = block_pairs[ti]
            sx = float(shift_x[ti])
            sy = float(shift_y[ti])
            sz = float(shift_z[ti])
            source_atoms = ba[bx]
            interacting_row = interacting[ti]
            for slot in range(BLOCK_SIZE):
                aj = int(interacting_row[slot])
                if aj < 0 or aj == NUM_ATOMS_SENTINEL:
                    continue
                pdb_j = sorted_to_pdb[aj]
                pos_j = positions[pdb_j]
                found_close = False
                for si in range(BLOCK_SIZE):
                    ak = source_atoms[si]
                    if ak < 0:
                        continue
                    pdb_k = sorted_to_pdb[ak]
                    pos_k = positions[pdb_k]
                    dx = (pos_j[0] + sx) - pos_k[0]
                    dy = (pos_j[1] + sy) - pos_k[1]
                    dz = (pos_j[2] + sz) - pos_k[2]
                    if dx * dx + dy * dy + dz * dz <= build_radius_sq:
                        found_close = True
                        break
                assert found_close, (
                    f"PBC tile {ti}: j-atom {aj} (pdb={pdb_j}) with shift "
                    f"({sx},{sy},{sz}) not within build_radius of block {bx}. "
                    f"Incompatible shifts may have been packed together."
                )


class TestSnapshotPostWrapIntegration:
    """Integration test: System._do_rebuild captures snapshot AFTER wrap."""

    def test_snapshot_holds_wrapped_positions(self):
        """After _do_rebuild, snapshot must hold wrapped (in-box) positions.

        Upload a particle outside the box, rebuild, and verify the snapshot
        holds the wrapped value — not the raw uploaded value.
        """
        from mdpy.system import System

        n = 4
        builder = Builder()
        builder.set_particles(
            masses=np.ones(n, dtype=np.float32),
            charges=np.zeros(n, dtype=np.float32),
            particle_types=np.zeros(n, dtype=np.int32),
        )
        builder.build_exclusion_map()
        topology, _ = builder.build()

        box = 50.0
        pbc_matrix = np.eye(3, dtype=np.float32) * box

        system = System(topology)
        system.upload_pbc(pbc_matrix)
        system._cutoff = 10.0
        system._skin = 2.0

        positions = np.array([
            [box + 0.3, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ], dtype=np.float32)
        velocities = np.zeros((n, 3), dtype=np.float32)
        system.upload_positions(positions)
        system.upload_velocities(velocities)

        system.update_neighbor_list(force_rebuild=True)

        snap_x = cp.asnumpy(system._block_list.d_positions_at_rebuild_x)
        gpu_x = cp.asnumpy(system.gpu.d_positions_x)

        # Snapshot must be populated (not empty/stale)
        assert system._block_list.d_positions_at_rebuild_x.size == n
        # Snapshot must match wrapped GPU positions (both in sorted order)
        assert np.allclose(snap_x, gpu_x, atol=1e-5)

        max_snap = float(np.max(np.abs(snap_x)))
        max_gpu = float(np.max(np.abs(gpu_x)))
        assert max_snap < box, (
            f"Snapshot holds out-of-box position (max={max_snap}), "
            f"expected < {box}. Snapshot was captured BEFORE wrap."
        )
        assert max_gpu < box, (
            f"GPU positions out of box (max={max_gpu})"
        )

    def test_conditional_rebuild_skips_when_displacement_small(self):
        """update_neighbor_list should NOT rebuild when no particle moved.

        After the fix: rebuild is triggered by the displacement flag, not
        by an unconditional timer. If positions don't change, the flag stays
        0 and rebuild is skipped indefinitely.
        """
        from mdpy.system import System

        n = 4
        builder = Builder()
        builder.set_particles(
            masses=np.ones(n, dtype=np.float32),
            charges=np.zeros(n, dtype=np.float32),
            particle_types=np.zeros(n, dtype=np.int32),
        )
        builder.build_exclusion_map()
        topology, _ = builder.build()

        box = 50.0
        pbc_matrix = np.eye(3, dtype=np.float32) * box
        system = System(topology)
        system.upload_pbc(pbc_matrix)
        system._cutoff = 10.0
        system._skin = 2.0

        positions = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [3.0, 1.0, 1.0],
            [4.0, 1.0, 1.0],
        ], dtype=np.float32)
        velocities = np.zeros((n, 3), dtype=np.float32)
        system.upload_positions(positions)
        system.upload_velocities(velocities)

        system.update_neighbor_list(force_rebuild=True)

        # Monkey-patch to count _do_rebuild calls
        original_do_rebuild = system._do_rebuild
        rebuild_calls = [0]
        def counting_do_rebuild(*args, **kwargs):
            rebuild_calls[0] += 1
            return original_do_rebuild(*args, **kwargs)
        system._do_rebuild = counting_do_rebuild

        # Call update_neighbor_list many times without moving positions.
        # sync_interval=3 so the flag-read path triggers frequently.
        for _ in range(30):
            system.update_neighbor_list(sync_interval=3)

        # Old code rebuilds ~10 times (every sync_interval steps, unconditionally).
        # New code should rebuild 0 times (no displacement, flag stays 0).
        assert rebuild_calls[0] == 0, (
            f"Expected 0 rebuilds (no displacement), got {rebuild_calls[0]}"
        )

    def test_conditional_rebuild_fires_on_displacement(self):
        """update_neighbor_list MUST rebuild when a particle exceeds skin/2.

        Complement to test_conditional_rebuild_skips_when_displacement_small:
        verifies the positive case — displacement > skin/2 triggers rebuild
        at the next sync_interval boundary.
        """
        from mdpy.core.topology import Builder
        from mdpy.system import System

        n = 4
        builder = Builder()
        builder.set_particles(
            masses=np.ones(n, dtype=np.float32),
            charges=np.zeros(n, dtype=np.float32),
            particle_types=np.zeros(n, dtype=np.int32),
        )
        builder.build_exclusion_map()
        topology, _ = builder.build()

        box = 50.0
        pbc_matrix = np.eye(3, dtype=np.float32) * box
        system = System(topology)
        system.upload_pbc(pbc_matrix)
        system._cutoff = 10.0
        system._skin = 2.0

        positions = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [3.0, 1.0, 1.0],
            [4.0, 1.0, 1.0],
        ], dtype=np.float32)
        velocities = np.zeros((n, 3), dtype=np.float32)
        system.upload_positions(positions)
        system.upload_velocities(velocities)

        system.update_neighbor_list(force_rebuild=True)

        # Monkey-patch to count _do_rebuild calls
        original_do_rebuild = system._do_rebuild
        rebuild_calls = [0]
        def counting_do_rebuild(*args, **kwargs):
            rebuild_calls[0] += 1
            return original_do_rebuild(*args, **kwargs)
        system._do_rebuild = counting_do_rebuild

        # Move particle 0 by 1.5 Angstrom — exceeds skin/2 = 1.0
        moved = positions.copy()
        moved[0, 0] += 1.5
        system.upload_positions(moved)

        sync_interval = 3
        for _ in range(sync_interval):
            system.update_neighbor_list(sync_interval=sync_interval)

        assert rebuild_calls[0] == 1, (
            f"Expected 1 rebuild (particle moved {1.5} > skin/2={system._skin/2}), "
            f"got {rebuild_calls[0]}"
        )
