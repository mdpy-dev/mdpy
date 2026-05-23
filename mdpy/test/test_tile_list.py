import numpy as np
import cupy as cp
import pytest
from mdpy.core.topology import Builder
from mdpy.core.tile_list import TileList, W, NUM_ATOMS_SENTINEL


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


def _make_topology_4_particles_bond():
    builder = Builder()
    builder.set_particles(
        masses=np.ones(4, dtype=np.float32),
        charges=np.zeros(4, dtype=np.float32),
        particle_types=np.zeros(4, dtype=np.int32),
    )
    builder.add_bond(0, 1, k=305.0, r0=1.5)
    builder.add_bond(1, 2, k=310.0, r0=1.4)
    builder.add_bond(2, 3, k=300.0, r0=1.5)
    builder.add_angle(0, 1, 2, force_constant=50.0, equilibrium_angle=1.9)
    builder.add_angle(1, 2, 3, force_constant=50.0, equilibrium_angle=1.9)
    builder.build_exclusion_map()
    topology, _ = builder.build()
    return topology


class TestTileListFormat:

    def test_block_coverage(self):
        n, box = 50, 50.0
        positions = _make_positions(n, box)
        topology = _make_topology(n)
        pbc_matrix = _make_pbc(box)
        pbc_inv = np.linalg.inv(pbc_matrix)
        tl = TileList(cutoff=10.0, skin=2.0)
        tl.rebuild(positions, topology, pbc_matrix, pbc_inv)
        ba = tl.block_atoms.ravel()
        real_atoms = ba[ba >= 0]
        unique, counts = np.unique(real_atoms, return_counts=True)
        assert len(unique) == n
        assert np.all(counts == 1)

    def test_tiles_not_empty(self):
        n, box = 50, 50.0
        positions = _make_positions(n, box)
        topology = _make_topology(n)
        pbc_matrix = _make_pbc(box)
        pbc_inv = np.linalg.inv(pbc_matrix)
        tl = TileList(cutoff=10.0, skin=2.0)
        tl.rebuild(positions, topology, pbc_matrix, pbc_inv)
        assert tl.num_tiles > 0
        assert tl.tiles.shape == (tl.num_tiles,)
        assert tl.interacting_atoms.shape == (tl.num_tiles, W)

    def test_atom_to_block_mapping(self):
        n, box = 50, 50.0
        positions = _make_positions(n, box)
        topology = _make_topology(n)
        pbc_matrix = _make_pbc(box)
        pbc_inv = np.linalg.inv(pbc_matrix)
        tl = TileList(cutoff=10.0, skin=2.0)
        tl.rebuild(positions, topology, pbc_matrix, pbc_inv)

        atb = cp.asnumpy(tl.d_atom_to_block)
        ats = cp.asnumpy(tl.d_atom_to_slot)
        for block_index in range(tl.num_blocks):
            for slot in range(W):
                atom_id = int(tl.block_atoms[block_index, slot])
                if atom_id >= 0:
                    assert atb[atom_id] == block_index
                    assert ats[atom_id] == slot

    def test_morton_code_ordering(self):
        np.random.seed(123)
        n = 200
        positions = np.random.uniform(0, 40, (n, 3)).astype(np.float32)
        topology = _make_topology(n)
        pbc_matrix = _make_pbc(50.0)
        pbc_inv = np.linalg.inv(pbc_matrix)
        tl = TileList(cutoff=8.0, skin=2.0)
        tl.rebuild(positions, topology, pbc_matrix, pbc_inv)

        ba = tl.block_atoms
        for block_index in range(tl.num_blocks):
            for slot in range(W):
                atom_id = int(ba[block_index, slot])
                if atom_id < 0:
                    continue
                if slot > 0:
                    prev_id = int(ba[block_index, slot - 1])
                    if prev_id >= 0:
                        assert atom_id != prev_id


class TestInteractingBlocks:

    def test_interacting_atoms_within_cutoff(self):
        n, box = 100, 30.0
        positions = _make_positions(n, box)
        topology = _make_topology(n)
        pbc_matrix = _make_pbc(box)
        pbc_inv = np.linalg.inv(pbc_matrix)
        cutoff = 8.0
        tl = TileList(cutoff=cutoff, skin=2.0)
        tl.rebuild(positions, topology, pbc_matrix, pbc_inv)
        ba = tl.block_atoms
        ia = tl.interacting_atoms
        build_radius = cutoff + 2.0
        for t in range(tl.num_tiles):
            bx = tl.tiles[t]
            for sj in range(W):
                aj = ia[t, sj]
                if aj == NUM_ATOMS_SENTINEL or aj < 0:
                    continue
                pos_j = positions[aj]
                found = False
                for sk in range(W):
                    ak = ba[bx, sk]
                    if ak < 0:
                        continue
                    delta = pos_j - positions[ak]
                    delta -= box * np.round(delta / box)
                    if np.sqrt(np.sum(delta ** 2)) <= build_radius + 0.1:
                        found = True
                        break
                assert found, f"atom {aj} not within build_radius of block {bx}"

    def test_pair_completeness(self):
        n, box = 30, 30.0
        positions = _make_positions(n, box, seed=123)
        topology = _make_topology(n)
        pbc_matrix = _make_pbc(box)
        pbc_inv = np.linalg.inv(pbc_matrix)
        cutoff, skin = 10.0, 2.0
        build_radius = cutoff + skin
        tl = TileList(cutoff=cutoff, skin=skin)
        tl.rebuild(positions, topology, pbc_matrix, pbc_inv)
        close_pairs = set()
        for i in range(n):
            for j in range(i + 1, n):
                delta = positions[i] - positions[j]
                delta -= box * np.round(delta / box)
                if np.sqrt(np.sum(delta ** 2)) <= build_radius:
                    close_pairs.add((i, j))
        found_pairs = set()
        ba = tl.block_atoms
        ia = tl.interacting_atoms
        for t in range(tl.num_tiles):
            bx = tl.tiles[t]
            for sj in range(W):
                aj = ia[t, sj]
                if aj < 0 or aj >= n:
                    continue
                for sk in range(W):
                    ak = ba[bx, sk]
                    if ak < 0:
                        continue
                    pair = tuple(sorted([aj, ak]))
                    if pair[0] != pair[1]:
                        found_pairs.add(pair)
        missing = close_pairs - found_pairs
        assert len(missing) == 0, f"Missing {len(missing)} pairs"

    def test_interaction_tiles_cull(self):
        n = 100
        topology = _make_topology(n)
        np.random.seed(42)
        positions = np.zeros((n, 3), dtype=np.float32)
        positions[:32] = [0, 0, 0]
        positions[32:64] = [150, 0, 0]
        positions[64:96] = [0, 150, 0]
        positions[96:] = [150, 150, 0]
        positions += np.random.randn(n, 3).astype(np.float32) * 2
        positions -= positions.min(axis=0)
        pbc_matrix = _make_pbc(500.0)
        pbc_inv = np.linalg.inv(pbc_matrix)
        tl = TileList(cutoff=5.0, skin=2.0)
        tl.rebuild(positions, topology, pbc_matrix, pbc_inv)
        max_possible = tl.num_blocks * tl.num_blocks
        assert tl.num_tiles < max_possible


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
        tl = TileList(cutoff=10.0, skin=2.0)
        tl.rebuild(positions, topology, pbc_matrix, pbc_inv)
        excl = tl.exclusion_masks
        ia = tl.interacting_atoms
        atb = cp.asnumpy(tl.d_atom_to_block)
        ats = cp.asnumpy(tl.d_atom_to_slot)
        for t in range(tl.num_tiles):
            bx = tl.tiles[t]
            for sj in range(W):
                aj = ia[t, sj]
                if aj < 0 or aj >= n:
                    continue
                if atb[aj] != bx:
                    continue
                mask = excl[t, sj]
                slot_in_block = ats[aj]
                for s in range(slot_in_block + 1):
                    assert (mask >> s) & 1, f"triangle bit {s} missing"

    def test_dihedral_scaling(self):
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
        tl = TileList(cutoff=10.0, skin=2.0)
        tl.rebuild(positions, topology, pbc_matrix, pbc_inv)
        excl = tl.exclusion_masks
        scale = tl.scaling_masks
        ia = tl.interacting_atoms
        atb = cp.asnumpy(tl.d_atom_to_block)
        ats = cp.asnumpy(tl.d_atom_to_slot)
        found_14 = False
        for t in range(tl.num_tiles):
            bx = tl.tiles[t]
            for sj in range(W):
                aj = ia[t, sj]
                if aj < 0 or aj >= n:
                    continue
                if atb[aj] != bx:
                    continue
                slot_aj = ats[aj]
                for pair_a, pair_b in [(0, 3), (1, 4), (2, 5)]:
                    if aj == pair_a and atb[pair_b] == bx:
                        slot_b = ats[pair_b]
                        if not ((excl[t, sj] >> slot_b) & 1):
                            if (scale[t, sj] >> slot_b) & 1:
                                found_14 = True
        assert found_14, "expected 1-4 pair in scaling mask"

    def test_4_particle_bond_exclusion(self):
        topology = _make_topology_4_particles_bond()
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.5, 0.0, 0.0],
        ], dtype=np.float32)
        pbc_matrix = _make_pbc(50.0)
        pbc_inv = np.linalg.inv(pbc_matrix)
        tl = TileList(cutoff=100.0, skin=10.0)
        tl.rebuild(positions, topology, pbc_matrix, pbc_inv)

        excl = tl.exclusion_masks
        ia = tl.interacting_atoms
        atb = cp.asnumpy(tl.d_atom_to_block)
        ats = cp.asnumpy(tl.d_atom_to_slot)
        bx = tl.tiles[0]
        for sj in range(4):
            aj = ia[0, sj]
            mask = int(excl[0, sj])
            slot_aj = ats[aj]
            assert mask & (1 << slot_aj), "diagonal should be excluded"
            if aj == 0:
                assert mask & (1 << ats[1]), "1-2 pair (0,1) should be excluded"
                assert mask & (1 << ats[2]), "1-3 pair (0,2) should be excluded"
            if aj == 1:
                assert mask & (1 << ats[0]), "lower triangle"
                assert mask & (1 << ats[2]), "1-2 pair (1,2) excluded"
                assert mask & (1 << ats[3]), "1-3 pair (1,3) excluded"

    def test_4_particle_dihedral_scaling(self):
        builder = Builder()
        builder.set_particles(
            masses=np.ones(4, dtype=np.float32),
            charges=np.zeros(4, dtype=np.float32),
            particle_types=np.zeros(4, dtype=np.int32),
        )
        builder.add_bond(0, 1, k=305.0, r0=1.5)
        builder.add_bond(1, 2, k=310.0, r0=1.4)
        builder.add_bond(2, 3, k=300.0, r0=1.5)
        builder.add_angle(0, 1, 2, force_constant=50.0, equilibrium_angle=1.9)
        builder.add_angle(1, 2, 3, force_constant=50.0, equilibrium_angle=1.9)
        builder.add_dihedral(0, 1, 2, 3, force_constant=0.5, periodicity=3, phase=0.0)
        builder.build_exclusion_map()
        topology, _ = builder.build()
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.5, 0.0, 0.0],
        ], dtype=np.float32)
        pbc_matrix = _make_pbc(50.0)
        pbc_inv = np.linalg.inv(pbc_matrix)
        tl = TileList(cutoff=100.0, skin=10.0)
        tl.rebuild(positions, topology, pbc_matrix, pbc_inv)

        scale = tl.scaling_masks
        excl = tl.exclusion_masks
        ia = tl.interacting_atoms
        atb = cp.asnumpy(tl.d_atom_to_block)
        ats = cp.asnumpy(tl.d_atom_to_slot)
        bx = tl.tiles[0]
        for sj in range(4):
            aj = ia[0, sj]
            if aj == 0:
                slot_3 = ats[3]
                assert (int(scale[0, sj]) >> slot_3) & 1, "1-4 pair (0,3) should have scaling mask"
                assert not ((int(excl[0, sj]) >> slot_3) & 1), "1-4 pair should NOT be in exclusion"
            if aj == 3:
                slot_0 = ats[0]
                assert (int(scale[0, sj]) >> slot_0) & 1, "1-4 pair (3,0) should have scaling mask"


class TestPBCHandling:

    def test_cross_boundary_tiles(self):
        n, box = 64, 50.0
        positions = np.zeros((n, 3), dtype=np.float32)
        rng = np.random.RandomState(42)
        positions[:32, 0] = rng.uniform(0, 2, 32)
        positions[32:, 0] = rng.uniform(48, 50, 32)
        positions[:, 1] = rng.uniform(20, 30, n)
        positions[:, 2] = rng.uniform(20, 30, n)
        topology = _make_topology(n)
        pbc_matrix = _make_pbc(box)
        pbc_inv = np.linalg.inv(pbc_matrix)
        tl = TileList(cutoff=12.0, skin=2.0)
        tl.rebuild(positions, topology, pbc_matrix, pbc_inv)
        assert tl.num_tiles > 0
        ba = tl.block_atoms
        ia = tl.interacting_atoms
        found_cross = False
        for t in range(tl.num_tiles):
            bx = tl.tiles[t]
            has_g1 = any(0 <= ba[bx, s] < 32 for s in range(W))
            if not has_g1:
                continue
            for s in range(W):
                if 32 <= ia[t, s] < n:
                    found_cross = True
                    break
            if found_cross:
                break
        assert found_cross, "Expected cross-boundary tiles"


class TestRebuildMechanics:

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
        tl = TileList(cutoff=10.0, skin=2.0)
        tl.rebuild(positions, topology, pbc_matrix, pbc_inv)

        for _ in range(19):
            assert not tl.check_rebuild(positions)
        assert tl.check_rebuild(positions)

    def test_check_rebuild_uninitialized(self):
        topology = _make_topology(4)
        positions = np.zeros((4, 3), dtype=np.float32)
        tl = TileList(cutoff=10.0, skin=2.0)
        assert tl.check_rebuild(positions)

    def test_self_tile_small(self):
        topology = _make_topology(4)
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ], dtype=np.float32)
        pbc_matrix = _make_pbc(50.0)
        pbc_inv = np.linalg.inv(pbc_matrix)
        tl = TileList(cutoff=100.0, skin=10.0)
        tl.rebuild(positions, topology, pbc_matrix, pbc_inv)
        assert tl.num_blocks == 1
        assert tl.num_tiles > 0

    def test_small_grid_single_bin(self):
        topology = _make_topology(8)
        positions = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],
            [6.0, 6.0, 6.0],
            [7.0, 7.0, 7.0],
            [8.0, 8.0, 8.0],
        ], dtype=np.float32)
        pbc_matrix = _make_pbc(15.0)
        pbc_inv = np.linalg.inv(pbc_matrix)
        tl = TileList(cutoff=10.0, skin=2.0)
        tl.rebuild(positions, topology, pbc_matrix, pbc_inv)
        assert tl.num_blocks >= 1
        assert tl.num_tiles >= 1

    def test_zero_cross_tiles_far_particles(self):
        topology = _make_topology(4)
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [101.0, 0.0, 0.0],
        ], dtype=np.float32)
        pbc_matrix = _make_pbc(200.0)
        pbc_inv = np.linalg.inv(pbc_matrix)
        tl = TileList(cutoff=5.0, skin=1.0)
        tl.rebuild(positions, topology, pbc_matrix, pbc_inv)
        assert tl.num_tiles > 0

    def test_70_particles_masks(self):
        np.random.seed(77)
        n = 70
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
        positions = np.random.uniform(0, 30, (n, 3)).astype(np.float32)
        pbc_matrix = _make_pbc(50.0)
        pbc_inv = np.linalg.inv(pbc_matrix)
        tl = TileList(cutoff=8.0, skin=2.0)
        tl.rebuild(positions, topology, pbc_matrix, pbc_inv)

        if tl.num_tiles > 0:
            assert tl.exclusion_masks.shape == (tl.num_tiles, W)
            assert tl.scaling_masks.shape == (tl.num_tiles, W)
