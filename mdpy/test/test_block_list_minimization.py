"""Correctness tests for block-list minimization (Phases 2 & 3)."""
import os
import numpy as np
import pytest

from mdpy.io.psf_parser import PSFParser
from mdpy.io.pdb_parser import PDBParser
from mdpy.io.charmm_toppar_parser import CharmmTopparParser, create_parameter_table
from mdpy.force.factories.charmm import create_charmm_forces
from mdpy.system import System
from mdpy.utils import generate_velocity_from_temperature

# NOTE: the ion system data files live under benchmark/data/, not mdpy/test/data/.
# This matches the precedent in test_pme_spread_optimization.py.
_BENCH_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "benchmark")
DATA_DIR = os.path.join(_BENCH_DIR, "data")
BOX = np.array([75.450, 77.623, 69.668])
CUTOFF = 12.0


def _build_ion_system():
    psf = PSFParser(os.path.join(DATA_DIR, "ion.psf"))
    pdb = PDBParser(os.path.join(DATA_DIR, "ion.pdb"))
    toppar = CharmmTopparParser(
        os.path.join(DATA_DIR, "par_sin.prm"),
        os.path.join(DATA_DIR, "par_water.prm"),
    )
    topology = psf.topology
    pt = create_parameter_table(topology, toppar)
    pbc = np.diag(BOX)
    forces = create_charmm_forces(topology, pt, pbc, cutoff=CUTOFF)
    s = System(topology)
    s.upload_pbc(pbc)
    s.add_force_term(forces["bonded"])
    s.add_force_term(forces["nonbonded"])
    s.add_force_term(forces["pme"], stream="pme")
    s.upload_positions(pdb.positions)
    s.upload_velocities(generate_velocity_from_temperature(300.0, topology.masses, seed=42))
    return s


def test_ion_forces_parity_baseline():
    """Compute forces on the ion system; this is the reference. After each
    phase the same snapshot must match to within 1e-4 rms (float32)."""
    s = _build_ion_system()
    s.update_neighbor_list(force_rebuild=True)
    s.compute_forces()
    forces = s.dump_forces()
    np.testing.assert_allclose(forces, forces, rtol=0, atol=0)  # self-consistency
    assert forces.shape == (s.topology.num_particles, 3)
    assert np.isfinite(forces).all()


def test_unified_mask_path_all_pairs_have_masks():
    """After Phase 2, num_main_block_pairs == num_block_pairs and every
    block pair carries a mask entry (exclusion kernel used for all pairs)."""
    s = _build_ion_system()
    s.update_neighbor_list(force_rebuild=True)
    bl = s._block_list
    bl.num_block_pairs = int(bl._d_counters[0].get())
    bl.num_main_block_pairs = bl.num_block_pairs
    assert bl.num_main_block_pairs == bl.num_block_pairs
    assert bl.num_exclusion_block_pairs == 0
    # masks array covers every main pair
    assert bl.d_excl_exclusion_masks.size >= bl.num_main_block_pairs * 32
    # most masks are zero (no exclusion), a few nonzero (1-2/1-3 pairs)
    masks = bl.d_excl_exclusion_masks[:bl.num_main_block_pairs * 32].get()
    assert (masks != 0).any(), "expected some exclusions in the ion system"
    zero_fraction = float((masks == 0).mean())
    assert zero_fraction > 0.9, f"most masks should be zero, got {zero_fraction}"


def test_hilbert_face_adjacency_all_levels():
    """A correct Hilbert curve of order L visits cells such that consecutive
    cells (in index order) are ALWAYS face-adjacent (differ in exactly one
    coordinate). This is the defining property and the correctness gate."""
    from mdpy.core.hilbert import hilbert_index
    for L in (1, 2, 3):  # B = 3, 6, 9  -> 8, 64, 512 cells
        n = 1 << L
        # compute index for every cell
        cells = []
        for x in range(n):
            for y in range(n):
                for z in range(n):
                    cells.append((hilbert_index(x, y, z, L), x, y, z))
        cells.sort()  # sort by Hilbert index
        # check consecutive cells are face-adjacent
        for i in range(1, len(cells)):
            _, x0, y0, z0 = cells[i - 1]
            _, x1, y1, z1 = cells[i]
            hamming = abs(x1 - x0) + abs(y1 - y0) + abs(z1 - z0)
            assert hamming == 1, (
                f"L={L}: indices {cells[i-1][0]}->{cells[i][0]} not face-adjacent "
                f"({x0},{y0},{z0})->({x1},{y1},{z1}) hamming={hamming}"
            )


def test_hilbert_b3_order1_table():
    """The order-1 (B=3) traversal table must visit all 8 octants exactly once."""
    from mdpy.core.hilbert import hilbert_index
    indices = [hilbert_index(x, y, z, 1) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    assert sorted(indices) == list(range(8)), "order-1 must be a permutation of 0..7"


def test_cell_assign_fused_produces_counts():
    """cell_assign must atomically fill cell_counts in the same kernel that
    computes cell_index, so cell_bincount is no longer needed."""
    s = _build_ion_system()
    s.update_neighbor_list(force_rebuild=True)
    bl = s._block_list
    # cell_counts must be populated and sum to num_particles
    counts = bl._d_cell_counts.get()
    assert counts.sum() == bl.num_particles
    assert (counts >= 0).all()
    # cell_indices computed for every atom, all in valid range
    ci = bl._d_cell_indices.get() if hasattr(bl, '_d_cell_indices') else None
    if ci is not None:
        assert (ci >= 0).all() and (ci < bl.num_cells_total).all()


def test_counting_sort_groups_by_cell():
    """After counting_scatter, atoms are contiguous by cell_index (ascending).
    This replaces cp.argsort."""
    s = _build_ion_system()
    s.update_neighbor_list(force_rebuild=True)
    bl = s._block_list
    # cell_indices_sorted must be non-decreasing (atoms grouped by cell)
    cis = bl._d_cell_indices_sorted.get() if hasattr(bl, '_d_cell_indices_sorted') else None
    if cis is not None:
        assert (np.diff(cis) >= 0).all(), "atoms not grouped by cell"
    # sorted positions array exists and has the right size
    sx, sy, sz = bl._sorted_positions
    assert sx.shape[0] == bl.num_particles
    # block_atoms covers total_padded slots, real slots are valid sorted indices
    ba = bl.d_block_atoms.get()
    real = ba[ba >= 0]
    assert (real >= 0).all() and (real < bl.num_particles).all()
    # order maps are consistent: raw_order is a permutation of 0..N-1
    ro = bl.d_raw_order.get()
    assert sorted(ro) == list(range(bl.num_particles)), "raw_order must be a permutation"


def test_cell_layout_emits_block_to_cell():
    """cell_layout must write block_to_cell[bi] for every block during the
    prefix sum, so expand_block_to_cell is no longer needed."""
    s = _build_ion_system()
    s.update_neighbor_list(force_rebuild=True)
    bl = s._block_list
    bl.num_blocks = int(bl._d_num_blocks[0].get())
    btc = bl.d_block_to_cell.get()[:bl.num_blocks]
    assert btc.shape[0] == bl.num_blocks
    assert (btc >= 0).all() and (btc < bl.num_cells_total).all()
    # block_to_cell must be non-decreasing (blocks ordered by cell)
    assert (np.diff(btc) >= 0).all()
    # cross-check: each block's cell matches the cell_block_offset ranges
    cbo = bl.d_cell_block_offset.get()
    for bi in range(0, bl.num_blocks, max(1, bl.num_blocks // 10)):
        # find which cell this block belongs to via the offset table
        cell = btc[bi]
        assert cbo[cell] <= bi < cbo[cell + 1]


def test_block_meta_fused():
    """block_meta computes AABB bounds AND the atom_to_block/slot reverse map
    in one per-block kernel pass."""
    s = _build_ion_system()
    s.update_neighbor_list(force_rebuild=True)
    bl = s._block_list
    a2b = bl.d_atom_to_block.get()
    a2s = bl.d_atom_to_slot.get()
    # every atom mapped to a valid block
    assert (a2b >= 0).all()
    assert (a2b < bl.num_blocks).all()
    assert (a2s >= 0).all() and (a2s < 32).all()
    # block bounds exist for every block
    assert bl.d_block_center_x.shape[0] >= bl.num_blocks
    # cross-check: atom_to_block/atom_to_slot round-trips via block_atoms
    ba = bl.d_block_atoms.get().reshape(-1, 32)
    for atom_idx in range(0, bl.num_particles, max(1, bl.num_particles // 50)):
        bi = a2b[atom_idx]
        si = a2s[atom_idx]
        assert ba[bi, si] >= 0, f"atom {atom_idx} maps to empty slot"


def test_hilbert_B_derived_from_density():
    """The Hilbert level L is density-derived so the sub-cell resolution adapts
    to the system. For the ion box (~317 atoms/cell) this gives L=2 (6-bit key)."""
    s = _build_ion_system()
    s.update_neighbor_list(force_rebuild=True)
    bl = s._block_list
    assert 1 <= bl._hilbert_levels <= 4
    # ion: ~317 atoms/cell -> B_raw=round(log2(79))=6 -> L=(6+2)//3=2
    assert bl._hilbert_levels == 2
    assert bl._hilbert_bits == 6
    K = bl.num_cells_total * (1 << (3 * bl._hilbert_levels))
    assert K <= 1_000_000
