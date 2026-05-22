import numpy as np
import pytest
from mdpy.core.topology import Topology, Builder
from mdpy.core.tile_list import TileList, W


def _make_box(size=50.0):
    pbc_matrix = np.diag([size, size, size]).astype(np.float32)
    pbc_inv = np.linalg.inv(pbc_matrix).astype(np.float32)
    return pbc_matrix, pbc_inv


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
    return builder.build()


def test_block_coverage():
    np.random.seed(42)
    number_particles = 50
    topology_builder = Builder()
    topology_builder.set_particles(
        masses=np.ones(number_particles, dtype=np.float32),
        charges=np.zeros(number_particles, dtype=np.float32),
        particle_types=np.zeros(number_particles, dtype=np.int32),
    )
    topology_builder.build_exclusion_map()
    topology = topology_builder.build()

    positions = np.random.randn(number_particles, 3).astype(np.float32) * 10
    positions -= positions.min(axis=0)
    pbc_matrix, pbc_inv = _make_box(50.0)

    tile_list = TileList(cutoff=10.0, skin=2.0)
    tile_list.rebuild(positions, topology, pbc_matrix, pbc_inv)

    assert tile_list.block_atoms.shape[1] == W
    assert tile_list.block_atoms.dtype == np.int32

    found = set()
    for block_index in range(tile_list.num_blocks):
        for slot in range(W):
            atom_id = int(tile_list.block_atoms[block_index, slot])
            if atom_id >= 0:
                assert atom_id not in found, f"particle {atom_id} appears in multiple blocks"
                found.add(atom_id)
    assert found == set(range(number_particles))

    total_slots = tile_list.num_blocks * W
    padding_count = total_slots - number_particles
    actual_padding = int(np.sum(tile_list.block_atoms == -1))
    assert actual_padding == padding_count


def test_self_tile_small():
    topology_builder = Builder()
    topology_builder.set_particles(
        masses=np.ones(4, dtype=np.float32),
        charges=np.zeros(4, dtype=np.float32),
        particle_types=np.zeros(4, dtype=np.int32),
    )
    topology_builder.build_exclusion_map()
    topology = topology_builder.build()

    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ], dtype=np.float32)
    pbc_matrix, pbc_inv = _make_box(50.0)

    tile_list = TileList(cutoff=100.0, skin=10.0)
    tile_list.rebuild(positions, topology, pbc_matrix, pbc_inv)

    assert tile_list.num_blocks == 1
    assert tile_list.num_self == 1
    assert tile_list.num_cross > 0
    assert all(tile_list.cross_tiles_i[k] == 0 for k in range(tile_list.num_cross))
    assert all(tile_list.cross_tiles_j[k] == 0 for k in range(tile_list.num_cross))


def test_interaction_tiles_cull():
    number_particles = 100
    topology_builder = Builder()
    topology_builder.set_particles(
        masses=np.ones(number_particles, dtype=np.float32),
        charges=np.zeros(number_particles, dtype=np.float32),
        particle_types=np.zeros(number_particles, dtype=np.int32),
    )
    topology_builder.build_exclusion_map()
    topology = topology_builder.build()

    np.random.seed(42)
    positions = np.zeros((number_particles, 3), dtype=np.float32)
    positions[:32] = [0, 0, 0]
    positions[32:64] = [150, 0, 0]
    positions[64:96] = [0, 150, 0]
    positions[96:] = [150, 150, 0]
    positions += np.random.randn(number_particles, 3).astype(np.float32) * 2
    positions -= positions.min(axis=0)
    pbc_matrix, pbc_inv = _make_box(500.0)

    tile_list = TileList(cutoff=5.0, skin=2.0)
    tile_list.rebuild(positions, topology, pbc_matrix, pbc_inv)

    num_tiles = tile_list.num_blocks
    max_possible = num_tiles * (num_tiles + 1) // 2
    assert tile_list.num_interactions < max_possible


def test_exclusion_masks_bond():
    topology = _make_topology_4_particles_bond()
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.5, 0.0, 0.0],
    ], dtype=np.float32)
    pbc_matrix, pbc_inv = _make_box(50.0)

    tile_list = TileList(cutoff=100.0, skin=10.0)
    tile_list.rebuild(positions, topology, pbc_matrix, pbc_inv)

    masks = tile_list.self_exclusion_masks
    assert masks.shape[0] == 1
    assert masks.shape[1] == W

    row_0 = int(masks[0, 0])
    assert row_0 & (1 << 0), "diagonal should be excluded"
    assert row_0 & (1 << 1), "1-2 pair (0,1) should be excluded"
    assert row_0 & (1 << 2), "1-3 pair (0,2) should be excluded"

    row_1 = int(masks[0, 1])
    assert row_1 & (1 << 0), "lower triangle excluded for self-tile"
    assert row_1 & (1 << 1), "diagonal"
    assert row_1 & (1 << 2), "1-2 pair (1,2) excluded"
    assert row_1 & (1 << 3), "1-3 pair (1,3) excluded"


def test_exclusion_masks_dihedral():
    topology_builder = Builder()
    topology_builder.set_particles(
        masses=np.ones(4, dtype=np.float32),
        charges=np.zeros(4, dtype=np.float32),
        particle_types=np.zeros(4, dtype=np.int32),
    )
    topology_builder.add_bond(0, 1, k=305.0, r0=1.5)
    topology_builder.add_bond(1, 2, k=310.0, r0=1.4)
    topology_builder.add_bond(2, 3, k=300.0, r0=1.5)
    topology_builder.add_angle(0, 1, 2, force_constant=50.0, equilibrium_angle=1.9)
    topology_builder.add_angle(1, 2, 3, force_constant=50.0, equilibrium_angle=1.9)
    topology_builder.add_dihedral(
        0, 1, 2, 3, force_constant=0.5, periodicity=3, phase=0.0
    )
    topology_builder.build_exclusion_map()
    topology = topology_builder.build()

    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.5, 0.0, 0.0],
    ], dtype=np.float32)
    pbc_matrix, pbc_inv = _make_box(50.0)

    tile_list = TileList(cutoff=100.0, skin=10.0)
    tile_list.rebuild(positions, topology, pbc_matrix, pbc_inv)

    scaling_masks = tile_list.self_scaling_masks
    row_0 = int(scaling_masks[0, 0])
    assert row_0 & (1 << 3), "1-4 pair (0,3) should have scaling mask"

    row_3 = int(scaling_masks[0, 3])
    assert row_3 & (1 << 0), "1-4 pair (3,0) should have scaling mask (reverse direction)"

    exclusion_masks = tile_list.self_exclusion_masks
    row_0_excl = int(exclusion_masks[0, 0])
    assert not (row_0_excl & (1 << 3)), "1-4 pair should NOT be in exclusion mask"


def test_check_rebuild():
    topology_builder = Builder()
    topology_builder.set_particles(
        masses=np.ones(4, dtype=np.float32),
        charges=np.zeros(4, dtype=np.float32),
        particle_types=np.zeros(4, dtype=np.int32),
    )
    topology_builder.build_exclusion_map()
    topology = topology_builder.build()

    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ], dtype=np.float32)
    pbc_matrix, pbc_inv = _make_box(50.0)

    tile_list = TileList(cutoff=10.0, skin=2.0)
    tile_list.rebuild(positions, topology, pbc_matrix, pbc_inv)

    small_move = positions + np.array([[0.1, 0.1, 0.1]], dtype=np.float32)
    assert not tile_list.check_rebuild(small_move)

    large_move = positions + np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
    assert tile_list.check_rebuild(large_move)


def test_check_rebuild_uninitialized():
    topology_builder = Builder()
    topology_builder.set_particles(
        masses=np.ones(4, dtype=np.float32),
        charges=np.zeros(4, dtype=np.float32),
        particle_types=np.zeros(4, dtype=np.int32),
    )
    topology_builder.build_exclusion_map()
    topology = topology_builder.build()

    positions = np.zeros((4, 3), dtype=np.float32)
    pbc_matrix, pbc_inv = _make_box(50.0)

    tile_list = TileList(cutoff=10.0, skin=2.0)
    assert tile_list.check_rebuild(positions)


def test_empty_system():
    number_particles = 10
    topology_builder = Builder()
    topology_builder.set_particles(
        masses=np.ones(number_particles, dtype=np.float32),
        charges=np.zeros(number_particles, dtype=np.float32),
        particle_types=np.zeros(number_particles, dtype=np.int32),
    )
    topology_builder.build_exclusion_map()
    topology = topology_builder.build()

    positions = np.zeros((number_particles, 3), dtype=np.float32)
    positions[0] = [0, 0, 0]
    positions[1] = [100, 0, 0]
    positions[2] = [0, 100, 0]
    positions[3] = [0, 0, 100]
    positions[4] = [100, 100, 0]
    positions[5] = [100, 0, 100]
    positions[6] = [0, 100, 100]
    positions[7] = [100, 100, 100]
    positions[8] = [50, 50, 50]
    positions[9] = [-50, -50, -50]
    pbc_matrix, pbc_inv = _make_box(300.0)

    tile_list = TileList(cutoff=1.0, skin=0.5)
    tile_list.rebuild(positions, topology, pbc_matrix, pbc_inv)

    assert tile_list.num_self > 0
    for tile_idx in range(tile_list.num_self):
        block_k = int(tile_list.self_tile_indices[tile_idx])
        block_atoms = tile_list.block_atoms[block_k]
        valid_atoms = block_atoms[block_atoms >= 0]
        for i in range(len(valid_atoms)):
            for j in range(i + 1, len(valid_atoms)):
                dx = positions[valid_atoms[i]] - positions[valid_atoms[j]]
                dist = np.linalg.norm(dx)
                assert dist < tile_list.build_radius or dist > tile_list.build_radius * 5, \
                    f"Atoms {valid_atoms[i]} and {valid_atoms[j]} in same block but distance {dist}"


def test_zero_cross_tiles_far_particles():
    topology_builder = Builder()
    topology_builder.set_particles(
        masses=np.ones(4, dtype=np.float32),
        charges=np.zeros(4, dtype=np.float32),
        particle_types=np.zeros(4, dtype=np.int32),
    )
    topology_builder.build_exclusion_map()
    topology = topology_builder.build()

    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [100.0, 0.0, 0.0],
        [101.0, 0.0, 0.0],
    ], dtype=np.float32)
    pbc_matrix, pbc_inv = _make_box(200.0)

    tile_list = TileList(cutoff=5.0, skin=1.0)
    tile_list.rebuild(positions, topology, pbc_matrix, pbc_inv)

    assert tile_list.num_interactions > 0
    assert tile_list.num_self > 0
    for tile_idx in range(tile_list.num_self):
        block_k = int(tile_list.self_tile_indices[tile_idx])
        block_atoms = tile_list.block_atoms[block_k]
        valid = block_atoms[block_atoms >= 0]
        if len(valid) > 1:
            for i in range(len(valid)):
                for j in range(i + 1, len(valid)):
                    dist = np.linalg.norm(positions[valid[i]] - positions[valid[j]])
                    assert dist < tile_list.build_radius


def test_morton_code_ordering():
    np.random.seed(123)
    number_particles = 200
    topology_builder = Builder()
    topology_builder.set_particles(
        masses=np.ones(number_particles, dtype=np.float32),
        charges=np.zeros(number_particles, dtype=np.float32),
        particle_types=np.zeros(number_particles, dtype=np.int32),
    )
    topology_builder.build_exclusion_map()
    topology = topology_builder.build()

    positions = np.random.uniform(0, 40, (number_particles, 3)).astype(np.float32)
    pbc_matrix, pbc_inv = _make_box(50.0)

    tile_list = TileList(cutoff=8.0, skin=2.0)
    tile_list.rebuild(positions, topology, pbc_matrix, pbc_inv)

    for block_index in range(tile_list.num_blocks):
        bx, by, bz = tile_list.block_bin[block_index]
        for slot in range(W):
            atom_id = int(tile_list.block_atoms[block_index, slot])
            if atom_id < 0:
                continue
            cell_size = tile_list.build_radius
            expected_bx = int(positions[atom_id, 0] / cell_size)
            expected_by = int(positions[atom_id, 1] / cell_size)
            expected_bz = int(positions[atom_id, 2] / cell_size)
            assert (bx, by, bz) == (expected_bx, expected_by, expected_bz), \
                f"atom {atom_id} in wrong bin: ({bx},{by},{bz}) vs expected ({expected_bx},{expected_by},{expected_bz})"


def test_small_grid_single_bin():
    topology_builder = Builder()
    topology_builder.set_particles(
        masses=np.ones(8, dtype=np.float32),
        charges=np.zeros(8, dtype=np.float32),
        particle_types=np.zeros(8, dtype=np.int32),
    )
    topology_builder.build_exclusion_map()
    topology = topology_builder.build()

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
    pbc_matrix, pbc_inv = _make_box(15.0)

    tile_list = TileList(cutoff=10.0, skin=2.0)
    tile_list.rebuild(positions, topology, pbc_matrix, pbc_inv)

    assert tile_list.num_blocks >= 1
    assert tile_list.num_self >= 1
    assert tile_list.num_self == tile_list.num_blocks


def test_pbc_cross_tiles():
    topology_builder = Builder()
    topology_builder.set_particles(
        masses=np.ones(64, dtype=np.float32),
        charges=np.zeros(64, dtype=np.float32),
        particle_types=np.zeros(64, dtype=np.int32),
    )
    topology_builder.build_exclusion_map()
    topology = topology_builder.build()

    positions = np.zeros((64, 3), dtype=np.float32)
    positions[:32, 0] = 1.0
    positions[32:, 0] = 49.0
    pbc_matrix, pbc_inv = _make_box(50.0)

    wrapped_frac = positions @ np.linalg.inv(pbc_matrix)
    wrapped_frac -= np.floor(wrapped_frac)
    wrapped_positions = (wrapped_frac @ pbc_matrix).astype(np.float32)

    tile_list = TileList(cutoff=5.0, skin=2.0)
    tile_list.rebuild(wrapped_positions, topology, pbc_matrix, pbc_inv)

    assert tile_list.num_cross > 0, "should find cross-tiles across PBC boundary"

    found_nonzero_shift = False
    for tile_idx in range(tile_list.num_cross):
        shift = tile_list.cross_tiles_shift[tile_idx]
        if np.any(shift != 0):
            found_nonzero_shift = True
            break
    assert found_nonzero_shift, "at least one cross-tile should have nonzero PBC shift"


def test_pair_completeness():
    np.random.seed(99)
    number_particles = 30
    topology_builder = Builder()
    topology_builder.set_particles(
        masses=np.ones(number_particles, dtype=np.float32),
        charges=np.zeros(number_particles, dtype=np.float32),
        particle_types=np.zeros(number_particles, dtype=np.int32),
    )
    topology_builder.build_exclusion_map()
    topology = topology_builder.build()

    positions = np.random.uniform(0, 20, (number_particles, 3)).astype(np.float32)
    pbc_matrix, pbc_inv = _make_box(30.0)

    tile_list = TileList(cutoff=10.0, skin=2.0)
    tile_list.rebuild(positions, topology, pbc_matrix, pbc_inv)

    tile_pairs = set()
    for tile_idx in range(tile_list.num_self):
        block_k = int(tile_list.self_tile_indices[tile_idx])
        for a in range(W):
            atom_a = int(tile_list.block_atoms[block_k, a])
            if atom_a < 0:
                continue
            for b in range(a + 1, W):
                atom_b = int(tile_list.block_atoms[block_k, b])
                if atom_b < 0:
                    continue
                pair = (min(atom_a, atom_b), max(atom_a, atom_b))
                tile_pairs.add(pair)

    for tile_idx in range(tile_list.num_cross):
        bi = int(tile_list.cross_tiles_i[tile_idx])
        bj = int(tile_list.cross_tiles_j[tile_idx])
        for a in range(W):
            atom_a = int(tile_list.block_atoms[bi, a])
            if atom_a < 0:
                continue
            for b in range(W):
                atom_b = int(tile_list.block_atoms[bj, b])
                if atom_b < 0:
                    continue
                pair = (min(atom_a, atom_b), max(atom_a, atom_b))
                tile_pairs.add(pair)

    build_radius = tile_list.build_radius
    for i in range(number_particles):
        for j in range(i + 1, number_particles):
            dx = positions[j] - positions[i]
            dx = dx - np.round(dx / 30.0) * 30.0
            dist = np.linalg.norm(dx)
            if dist < build_radius:
                assert (i, j) in tile_pairs, \
                    f"pair ({i},{j}) dist={dist:.2f} < build_radius={build_radius} not in tiles"


def test_cross_tile_exclusion_masks():
    np.random.seed(77)
    number_particles = 70
    topology_builder = Builder()
    topology_builder.set_particles(
        masses=np.ones(number_particles, dtype=np.float32),
        charges=np.zeros(number_particles, dtype=np.float32),
        particle_types=np.zeros(number_particles, dtype=np.int32),
    )
    for i in range(number_particles - 1):
        topology_builder.add_bond(i, i + 1, k=300.0, r0=1.5)
    topology_builder.build_exclusion_map()
    topology = topology_builder.build()

    positions = np.random.uniform(0, 30, (number_particles, 3)).astype(np.float32)
    pbc_matrix, pbc_inv = _make_box(50.0)

    tile_list = TileList(cutoff=8.0, skin=2.0)
    tile_list.rebuild(positions, topology, pbc_matrix, pbc_inv)

    if tile_list.num_cross > 0:
        assert tile_list.cross_exclusion_masks.shape[0] == tile_list.num_cross
        assert tile_list.cross_exclusion_masks.shape[1] == W
        assert tile_list.cross_scaling_masks.shape[0] == tile_list.num_cross


def test_atom_to_block_mapping_gpu():
    np.random.seed(42)
    number_particles = 50
    topology_builder = Builder()
    topology_builder.set_particles(
        masses=np.ones(number_particles, dtype=np.float32),
        charges=np.zeros(number_particles, dtype=np.float32),
        particle_types=np.zeros(number_particles, dtype=np.int32),
    )
    topology_builder.build_exclusion_map()
    topology = topology_builder.build()

    positions = np.random.randn(number_particles, 3).astype(np.float32) * 10
    positions -= positions.min(axis=0)
    pbc_matrix, pbc_inv = _make_box(50.0)

    tile_list = TileList(cutoff=10.0, skin=2.0)
    tile_list.rebuild(positions, topology, pbc_matrix, pbc_inv)

    import cupy as cp
    d_atom_to_block = tile_list.d_atom_to_block
    d_atom_to_slot = tile_list.d_atom_to_slot
    atom_to_block = cp.asnumpy(d_atom_to_block)
    atom_to_slot = cp.asnumpy(d_atom_to_slot)

    assert atom_to_block.dtype == np.int32
    assert atom_to_slot.dtype == np.int32
    assert atom_to_block.shape == (number_particles,)
    assert atom_to_slot.shape == (number_particles,)

    for block_index in range(tile_list.num_blocks):
        for slot in range(W):
            atom_id = int(tile_list.block_atoms[block_index, slot])
            if atom_id >= 0:
                assert atom_to_block[atom_id] == block_index, \
                    f"atom {atom_id} should map to block {block_index}, got {atom_to_block[atom_id]}"
                assert atom_to_slot[atom_id] == slot, \
                    f"atom {atom_id} should map to slot {slot}, got {atom_to_slot[atom_id]}"
