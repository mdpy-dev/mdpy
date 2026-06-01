import numpy as np
import cupy as cp
import pytest
from mdpy.core.topology import Builder
from mdpy.core.block_list import BlockList, W, NUM_ATOMS_SENTINEL


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


def _rebuild_and_build_tiles(n, box=50.0, cutoff=10.0, skin=2.0, seed=42, positions=None):
    if positions is None:
        positions = _make_positions(n, box, seed)
    topology = _make_topology(n)
    pbc_matrix = _make_pbc(box)
    pbc_inv = np.linalg.inv(pbc_matrix)
    bl = BlockList(cutoff=cutoff, skin=skin)
    bl.rebuild(positions, topology, pbc_matrix, pbc_inv)
    bl.build_tiles(topology, pbc_matrix)
    return bl, positions, pbc_matrix, pbc_inv, topology


class TestCellAssignment:

    def test_cell_grid_dimensions(self):
        n, box = 100, 50.0
        cutoff, skin = 10.0, 2.0
        bl, *_ = _rebuild_and_build_tiles(n, box, cutoff, skin)
        expected_nc = int(box / (cutoff + skin))
        assert bl.nc_x == expected_nc
        assert bl.nc_y == expected_nc
        assert bl.nc_z == expected_nc
        assert bl.nc_total == expected_nc ** 3

    def test_block_coverage(self):
        n, box = 100, 50.0
        bl, *_ = _rebuild_and_build_tiles(n, box)
        ba = bl.block_atoms.ravel()
        real_atoms = ba[ba >= 0]
        unique, counts = np.unique(real_atoms, return_counts=True)
        assert len(unique) == n
        assert np.all(counts == 1)

    def test_blocks_are_cell_aligned(self):
        n, box = 100, 50.0
        bl, *_ = _rebuild_and_build_tiles(n, box)
        atom_to_block = cp.asnumpy(bl.d_atom_to_block)
        ba = bl.block_atoms
        for bi in range(bl.num_blocks):
            for slot in range(W):
                atom_id = ba[bi, slot]
                if atom_id >= 0:
                    assert atom_to_block[atom_id] == bi, (
                        f"atom {atom_id} in block {bi} slot {slot} "
                        f"but d_atom_to_block[{atom_id}]={atom_to_block[atom_id]}"
                    )

    def test_atom_to_block_mapping(self):
        n, box = 100, 50.0
        bl, *_ = _rebuild_and_build_tiles(n, box)
        atom_to_block = cp.asnumpy(bl.d_atom_to_block)
        atom_to_slot = cp.asnumpy(bl.d_atom_to_slot)
        ba = bl.block_atoms
        for bi in range(bl.num_blocks):
            for slot in range(W):
                atom_id = ba[bi, slot]
                if atom_id >= 0:
                    assert atom_to_block[atom_id] == bi
                    assert atom_to_slot[atom_id] == slot

    def test_sort_order_is_correct(self):
        n, box = 100, 50.0
        bl, *_ = _rebuild_and_build_tiles(n, box)
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

    def test_tiles_not_empty(self):
        n, box = 100, 50.0
        bl, *_ = _rebuild_and_build_tiles(n, box, cutoff=10.0, skin=2.0)
        assert bl.num_tiles > 0
        tiles = bl.tiles
        interacting = bl.interacting_atoms
        assert tiles.shape == (bl.num_tiles,)
        assert interacting.shape == (bl.num_tiles, W)

    def test_interacting_atoms_within_cutoff(self):
        n, box = 100, 50.0
        cutoff, skin = 10.0, 2.0
        bl, positions, pbc_matrix, _, _ = _rebuild_and_build_tiles(n, box, cutoff, skin)
        sorted_to_pdb = cp.asnumpy(bl.d_sorted_to_pdb)
        tiles = bl.tiles
        interacting = bl.interacting_atoms
        ba = bl.block_atoms
        build_radius_sq = bl.build_radius ** 2
        pbc_2d = pbc_matrix.reshape(3, 3)
        box_diag = np.array([pbc_2d[0, 0], pbc_2d[1, 1], pbc_2d[2, 2]])

        for ti in range(bl.num_tiles):
            source_block = tiles[ti]
            interacting_row = interacting[ti]
            source_atoms = ba[source_block]
            for slot in range(W):
                aj = interacting_row[slot]
                if aj < 0 or aj == NUM_ATOMS_SENTINEL:
                    continue
                pdb_j = sorted_to_pdb[aj]
                pos_j = positions[pdb_j]
                found_close = False
                for si in range(W):
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
                    f"Tile {ti}: interacting atom sorted_idx={aj} pdb={pdb_j} "
                    f"is not within build_radius of any atom in source block {source_block}"
                )

    def test_pair_completeness(self):
        n, box = 100, 50.0
        cutoff, skin = 10.0, 2.0
        bl, positions, pbc_matrix, _, _ = _rebuild_and_build_tiles(n, box, cutoff, skin)
        sorted_to_pdb = cp.asnumpy(bl.d_sorted_to_pdb)
        tiles = bl.tiles
        interacting = bl.interacting_atoms
        ba = bl.block_atoms
        build_radius_sq = bl.build_radius ** 2
        pbc_2d = pbc_matrix.reshape(3, 3)
        box_diag = np.array([pbc_2d[0, 0], pbc_2d[1, 1], pbc_2d[2, 2]])

        found_pairs = set()
        for ti in range(bl.num_tiles):
            source_block = tiles[ti]
            source_atoms = ba[source_block]
            interacting_row = interacting[ti]
            for si in range(W):
                ak = source_atoms[si]
                if ak < 0:
                    continue
                for sj in range(W):
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
                        f"Pair (sorted {i}, sorted {j}) within build_radius but not found in tiles"
                    )

    def test_newton_third_law_no_duplicates(self):
        n, box = 100, 50.0
        bl, *_ = _rebuild_and_build_tiles(n, box, cutoff=10.0, skin=2.0)
        tiles = bl.tiles
        interacting = bl.interacting_atoms
        seen = set()
        for ti in range(bl.num_tiles):
            source_block = tiles[ti]
            for sj in range(W):
                aj = interacting[ti, sj]
                if aj < 0 or aj == NUM_ATOMS_SENTINEL:
                    continue
                pair = (source_block, aj)
                assert pair not in seen, (
                    f"Duplicate (block={source_block}, sorted_atom={aj}) in tiles"
                )
                seen.add(pair)

    def test_self_tile_small(self):
        n = 4
        box = 50.0
        cutoff, skin = 100.0, 10.0
        positions = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
        ], dtype=np.float32)
        bl, *_ = _rebuild_and_build_tiles(n, box, cutoff, skin, positions=positions)
        assert bl.num_blocks == 1
        assert bl.num_tiles > 0

    def test_interaction_tiles_cull(self):
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
        bl, *_ = _rebuild_and_build_tiles(n, box, cutoff, skin, positions=positions)
        assert bl.num_tiles < bl.num_blocks ** 2


class TestPBCHandling:

    def test_cross_boundary_tiles(self):
        n = 64
        box = 50.0
        cutoff, skin = 12.0, 2.0
        positions = np.zeros((n, 3), dtype=np.float32)
        positions[:32, 0] = 1.0
        positions[32:, 0] = box - 1.0
        rng = np.random.RandomState(99)
        positions[:, 1] = rng.uniform(0, box, n)
        positions[:, 2] = rng.uniform(0, box, n)
        bl, *_ = _rebuild_and_build_tiles(n, box, cutoff, skin, positions=positions)
        assert bl.num_tiles > 0
        sorted_to_pdb = cp.asnumpy(bl.d_sorted_to_pdb)
        ba = bl.block_atoms
        tiles = bl.tiles
        interacting = bl.interacting_atoms
        has_cross = False
        for ti in range(bl.num_tiles):
            source_block = tiles[ti]
            source_pdbs = set()
            for si in range(W):
                ak = ba[source_block, si]
                if ak >= 0:
                    source_pdbs.add(sorted_to_pdb[ak])
            source_left = any(positions[p, 0] < box / 2 for p in source_pdbs)
            source_right = any(positions[p, 0] > box / 2 for p in source_pdbs)
            for sj in range(W):
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
        assert has_cross, "No cross-boundary tiles found when expected"


from mdpy.core.tile_list import TileList as OldTileList


def _brute_force_pairs(positions, box, build_radius):
    n = len(positions)
    box_diag = np.array([box, box, box], dtype=np.float32)
    build_radius_sq = build_radius ** 2
    pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[j] - positions[i]
            dx -= box_diag * np.round(dx / box_diag)
            dist_sq = np.sum(dx ** 2)
            if dist_sq <= build_radius_sq:
                pairs.add((i, j))
    return pairs


def _collect_blocklist_pairs(bl):
    sorted_to_pdb = cp.asnumpy(bl.d_sorted_to_pdb)
    ba = bl.block_atoms
    ia = bl.interacting_atoms
    pairs = set()
    for t in range(bl.num_tiles):
        bx = bl.tiles[t]
        for sj in range(W):
            aj = ia[t, sj]
            if aj < 0 or aj >= bl.num_particles or aj == NUM_ATOMS_SENTINEL:
                continue
            aj_pdb = sorted_to_pdb[aj]
            for sk in range(W):
                ak = ba[bx, sk]
                if ak < 0:
                    continue
                ak_pdb = sorted_to_pdb[ak]
                pair = tuple(sorted([aj_pdb, ak_pdb]))
                if pair[0] != pair[1]:
                    pairs.add(pair)
    return pairs


def _collect_tilelist_pairs(tl):
    sorted_to_pdb = cp.asnumpy(tl.d_sorted_to_pdb)
    ba = tl.block_atoms
    ia = tl.interacting_atoms
    pairs = set()
    for t in range(tl.num_tiles):
        bx = tl.tiles[t]
        for sj in range(W):
            aj = ia[t, sj]
            if aj < 0 or aj >= tl.num_particles:
                continue
            aj_pdb = sorted_to_pdb[aj]
            for sk in range(W):
                ak = ba[bx, sk]
                if ak < 0:
                    continue
                ak_pdb = sorted_to_pdb[ak]
                pair = tuple(sorted([aj_pdb, ak_pdb]))
                if pair[0] != pair[1]:
                    pairs.add(pair)
    return pairs


class TestComparisonWithTileList:

    def test_same_pair_coverage_random(self):
        n, box, seed, cutoff, skin = 200, 50.0, 99, 8.0, 2.0
        positions = _make_positions(n, box, seed)
        topology = _make_topology(n)
        pbc_matrix = _make_pbc(box)
        pbc_inv = np.linalg.inv(pbc_matrix)

        bl = BlockList(cutoff=cutoff, skin=skin)
        bl.rebuild(positions, topology, pbc_matrix, pbc_inv)
        bl.build_tiles(topology, pbc_matrix)

        brute = _brute_force_pairs(positions, box, bl.build_radius)
        found = _collect_blocklist_pairs(bl)
        missing = brute - found
        assert not missing, f"Missing {len(missing)} pairs (out of {len(brute)})"

    def test_same_pair_coverage_dense(self):
        n, box, seed, cutoff, skin = 100, 20.0, 55, 10.0, 2.0
        positions = _make_positions(n, box, seed)
        topology = _make_topology(n)
        pbc_matrix = _make_pbc(box)
        pbc_inv = np.linalg.inv(pbc_matrix)

        bl = BlockList(cutoff=cutoff, skin=skin)
        bl.rebuild(positions, topology, pbc_matrix, pbc_inv)
        bl.build_tiles(topology, pbc_matrix)

        brute = _brute_force_pairs(positions, box, bl.build_radius)
        found = _collect_blocklist_pairs(bl)
        missing = brute - found
        assert not missing, f"Missing {len(missing)} pairs (out of {len(brute)})"

    def test_consistency_with_old_tilelist(self):
        n, box, seed, cutoff, skin = 200, 50.0, 99, 8.0, 2.0
        positions = _make_positions(n, box, seed)
        topology = _make_topology(n)
        pbc_matrix = _make_pbc(box)
        pbc_inv = np.linalg.inv(pbc_matrix)

        bl = BlockList(cutoff=cutoff, skin=skin)
        bl.rebuild(positions, topology, pbc_matrix, pbc_inv)
        bl.build_tiles(topology, pbc_matrix)

        tl = OldTileList(cutoff=cutoff, skin=skin)
        tl.rebuild(positions, topology, pbc_matrix, pbc_inv)
        tl.build_tiles(topology, pbc_matrix)

        brute = _brute_force_pairs(positions, box, bl.build_radius)
        bl_pairs = _collect_blocklist_pairs(bl)
        tl_pairs = _collect_tilelist_pairs(tl)

        bl_missing = brute - bl_pairs
        tl_missing = brute - tl_pairs
        assert not bl_missing, f"BlockList missing {len(bl_missing)}/{len(brute)} pairs"
        assert not tl_missing, f"TileList missing {len(tl_missing)}/{len(brute)} pairs"
