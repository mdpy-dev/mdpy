"""Newton's 3rd law and momentum conservation check for mdpy block list.

Checks:
1. Total momentum (sum of all forces) — should be ~zero
2. Per-pair force symmetry — detect duplicate or missing pairs
3. Compare total momentum against brute-force reference

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n md_analysis python -u mdpy/test/newton3_check.py
"""

import os
import sys
import time
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

PSF_PATH = os.path.join(DATA_DIR, '1M9Z.psf')
PDB_PATH = os.path.join(DATA_DIR, '1M9Z_minimized.pdb')
PRM_PATH = os.path.join(DATA_DIR, 'par_all36_prot.prm')
STR_PATH = os.path.join(DATA_DIR, 'toppar_water_ions.str')
REF_PATH = os.path.join(DATA_DIR, 'bruteforce_reference_1M9Z.npz')

BOX_SIZE = 108.0
CUTOFF = 12.0
SENTINEL = 0x7FFFFFFF
BLOCK_SIZE = 32


def setup_mdpy_system():
    from mdpy.io.psf_parser import PSFParser
    from mdpy.io.pdb_parser import PDBParser
    from mdpy.io.charmm_toppar_parser import CharmmTopparParser
    from mdpy.io.charmm_toppar_parser import create_parameter_table
    from mdpy.force.bonded_force import BondedForce
    from mdpy.force.nonbonded_force import NonbondedForce
    from mdpy.force.expressions.lennard_jones import lennard_jones
    from mdpy.force.expressions.coulomb import coulomb
    from mdpy.system import System

    psf = PSFParser(PSF_PATH)
    pdb = PDBParser(PDB_PATH)
    toppar = CharmmTopparParser(PRM_PATH, STR_PATH)
    topology = psf.topology
    parameter_table = create_parameter_table(topology, toppar)
    pbc_matrix = np.eye(3, dtype=np.float64) * BOX_SIZE
    pbc_inv = np.linalg.inv(pbc_matrix)

    system = System(topology, pbc_matrix, cutoff=CUTOFF)
    system.add_force_term(BondedForce.charmm(topology, parameter_table))
    nb = NonbondedForce(lennard_jones + coulomb, cutoff=CUTOFF)
    lj_pair = parameter_table.type_pair_parameters['lj_pair']
    nb.set_pair_parameter('sigma', lj_pair[0::2].astype(np.float32))
    nb.set_pair_parameter('epsilon', lj_pair[1::2].astype(np.float32))
    system.add_force_term(nb)

    raw = pdb.positions.astype(np.float64)
    frac = raw @ pbc_inv
    frac -= np.floor(frac)
    wrapped = frac @ pbc_matrix
    system.upload_positions(wrapped.astype(np.float32))
    system.upload_velocities(
        np.zeros((topology.num_particles, 3), dtype=np.float32)
    )
    positions_2d = (
        system.gpu.d_positions_x,
        system.gpu.d_positions_y,
        system.gpu.d_positions_z,
    )
    system.block_list.rebuild(
        positions_2d, topology, system.pbc_matrix, system.pbc_inv,
    )
    system._permute_all_arrays()
    system.block_list.build_block_pairs(topology, system.pbc_matrix)
    for term in system.force_terms:
        if hasattr(term, 'bind_sorted'):
            term.bind_sorted(topology, system.block_list, system.gpu)
    return system


def get_pdb_forces(system):
    import cupy as cp
    bl = system.block_list
    gpu = system.gpu
    sorted_to_pdb = bl.d_sorted_to_pdb
    pdb_fx = gpu.permute_from_sorted(sorted_to_pdb, gpu.d_forces_x).get()
    pdb_fy = gpu.permute_from_sorted(sorted_to_pdb, gpu.d_forces_y).get()
    pdb_fz = gpu.permute_from_sorted(sorted_to_pdb, gpu.d_forces_z).get()
    return np.stack([pdb_fx, pdb_fy, pdb_fz], axis=1).astype(np.float64)


def extract_block_list_pairs_with_counts(system):
    """Extract pairs AND count occurrences to detect duplicates."""
    import cupy as cp
    bl = system.block_list
    block_atoms = bl.block_atoms
    block_pairs = bl.block_pairs
    interacting_atoms = bl.interacting_atoms
    exclusion_masks = bl.exclusion_masks
    scaling_masks = bl.scaling_masks

    sorted_to_pdb = cp.asnumpy(bl.d_sorted_to_pdb)

    pair_counts = {}
    excluded_pair_set = set()
    scaled_pair_set = set()

    num_block_pairs = len(block_pairs)
    for pair_idx in range(num_block_pairs):
        block_x = block_pairs[pair_idx]
        exc_mask = exclusion_masks[pair_idx]
        scl_mask = scaling_masks[pair_idx]
        for slot_j in range(BLOCK_SIZE):
            atom_j_sorted = interacting_atoms[pair_idx, slot_j]
            if atom_j_sorted == SENTINEL:
                continue
            atom_j = int(sorted_to_pdb[atom_j_sorted])
            for slot_i in range(BLOCK_SIZE):
                atom_i_sorted = block_atoms[block_x, slot_i]
                if atom_i_sorted == SENTINEL:
                    continue
                atom_i = int(sorted_to_pdb[atom_i_sorted])
                if atom_i == atom_j:
                    continue
                a, b = min(atom_i, atom_j), max(atom_i, atom_j)
                key = (a, b)
                pair_counts[key] = pair_counts.get(key, 0) + 1
                bit = 1 << slot_i
                if exc_mask[slot_j] & bit:
                    excluded_pair_set.add(key)
                if scl_mask[slot_j] & bit:
                    scaled_pair_set.add(key)

    return pair_counts, excluded_pair_set, scaled_pair_set


def main():
    print("=" * 70)
    print("Newton's 3rd Law & Momentum Conservation Check")
    print("=" * 70)

    # --- Setup ---
    print("\n[1] Setting up mdpy system (1M9Z, 95567 atoms)...")
    t0 = time.time()
    system = setup_mdpy_system()
    print(f"    System ready in {time.time() - t0:.1f}s")

    # --- Compute forces ---
    print("\n[2] Computing forces...")
    t0 = time.time()
    system.compute_forces()
    forces = get_pdb_forces(system)
    energies = system.dump_energy()
    print(f"    Forces computed in {time.time() - t0:.2f}s")
    print(f"    Energies: { {k: f'{v:.6f}' for k, v in energies.items()} }")

    # --- Total momentum ---
    print("\n[3] Total momentum check (sum of all forces)...")
    total_force = forces.sum(axis=0)
    total_momentum_norm = np.linalg.norm(total_force)
    per_atom_force_mag = np.linalg.norm(forces, axis=1)
    max_force_mag = per_atom_force_mag.max()
    mean_force_mag = per_atom_force_mag.mean()
    print(f"    Sum of forces: [{total_force[0]:.10e}, {total_force[1]:.10e}, {total_force[2]:.10e}]")
    print(f"    Total momentum norm: {total_momentum_norm:.10e}")
    print(f"    Max single-atom force magnitude: {max_force_mag:.6e}")
    print(f"    Mean single-atom force magnitude: {mean_force_mag:.6e}")
    print(f"    Momentum / max_force ratio: {total_momentum_norm / max_force_mag:.10e}")
    if total_momentum_norm < 1e-6:
        print("    PASS: Total momentum is near zero")
    elif total_momentum_norm < 1e-2:
        print(f"    WARN: Total momentum is small but not zero ({total_momentum_norm:.6e})")
    else:
        print(f"    FAIL: Total momentum is large ({total_momentum_norm:.6e})")

    # --- Extract block list pairs with duplicate detection ---
    print("\n[4] Extracting block list pairs with duplicate detection...")
    t0 = time.time()
    pair_counts, excluded_pair_set, scaled_pair_set = (
        extract_block_list_pairs_with_counts(system)
    )
    elapsed = time.time() - t0
    print(f"    Extracted in {elapsed:.1f}s")

    unique_pairs = set(pair_counts.keys())
    non_excluded_pairs = unique_pairs - excluded_pair_set
    duplicates = {p: c for p, c in pair_counts.items() if c > 1}

    print(f"    Total unique pairs: {len(unique_pairs)}")
    print(f"    Non-excluded pairs: {len(non_excluded_pairs)}")
    print(f"    Excluded pairs: {len(excluded_pair_set)}")
    print(f"    Scaled (1-4) pairs: {len(scaled_pair_set)}")
    print(f"    Duplicate pairs: {len(duplicates)}")

    if duplicates:
        dup_counts = {}
        for c in duplicates.values():
            dup_counts[c] = dup_counts.get(c, 0) + 1
        print(f"    Duplicate breakdown (count -> occurrences): {dup_counts}")
        sorted_dups = sorted(duplicates.items(), key=lambda x: -x[1])[:10]
        print(f"    Top 10 most duplicated pairs: {sorted_dups}")
        print("    FAIL: Duplicate pairs detected — Newton's 3rd law violated!")
    else:
        print("    PASS: No duplicate pairs found")

    # --- Per-atom pair count ---
    print("\n[5] Per-atom pair count analysis...")
    atom_pair_count = {}
    for (a, b), count in pair_counts.items():
        atom_pair_count[a] = atom_pair_count.get(a, 0) + count
        atom_pair_count[b] = atom_pair_count.get(b, 0) + count
    counts = np.array(list(atom_pair_count.values()))
    print(f"    Atoms with pairs: {len(atom_pair_count)} / {system.topology.num_particles}")
    print(f"    Pair count per atom: min={counts.min()}, max={counts.max()}, "
          f"mean={counts.mean():.1f}, median={np.median(counts):.1f}")
    atoms_no_pairs = system.topology.num_particles - len(atom_pair_count)
    if atoms_no_pairs > 0:
        print(f"    WARN: {atoms_no_pairs} atoms have no neighbor pairs")

    # --- Compare against brute-force reference ---
    print("\n[6] Comparison against brute-force reference...")
    ref_momentum_norm = None
    momentum_diff = None
    missing_from_bl = None
    has_ref = os.path.exists(REF_PATH)

    if not has_ref:
        print(f"    SKIP: Reference file not found: {REF_PATH}")
        print("    Run: conda run -n md_analysis python mdpy/test/generate_bruteforce_reference.py")
    else:
        ref = dict(np.load(REF_PATH, allow_pickle=True))
        ref_i = ref['neighbor_pairs_i']
        ref_j = ref['neighbor_pairs_j']
        ref_flags = ref['pair_exclusion_flags']
        ref_total_forces = ref['total_forces']

        ref_all_pairs = set(zip(ref_i.astype(np.int64), ref_j.astype(np.int64)))
        ref_non_excluded = set(
            zip(ref_i[ref_flags != 1].astype(np.int64),
                ref_j[ref_flags != 1].astype(np.int64))
        )
        ref_excluded = set(
            zip(ref_i[ref_flags == 1].astype(np.int64),
                ref_j[ref_flags == 1].astype(np.int64))
        )

        print(f"    Reference total pairs: {len(ref_all_pairs)}")
        print(f"    Reference non-excluded pairs: {len(ref_non_excluded)}")
        print(f"    Reference excluded pairs: {len(ref_excluded)}")

        missing_from_bl = ref_non_excluded - non_excluded_pairs
        extra_in_bl = non_excluded_pairs - ref_non_excluded
        print(f"\n    Non-excluded pair comparison:")
        print(f"      Block list non-excluded: {len(non_excluded_pairs)}")
        print(f"      Reference non-excluded:  {len(ref_non_excluded)}")
        print(f"      Missing from block list:  {len(missing_from_bl)}")
        print(f"      Extra in block list:      {len(extra_in_bl)}")

        if missing_from_bl:
            sample = list(missing_from_bl)[:5]
            print(f"      Sample missing pairs: {sample}")
        if extra_in_bl:
            sample = list(extra_in_bl)[:5]
            print(f"      Sample extra pairs: {sample}")

        missing_excl = ref_excluded - excluded_pair_set
        extra_excl = excluded_pair_set - ref_excluded
        print(f"\n    Excluded pair comparison:")
        print(f"      Block list excluded: {len(excluded_pair_set)}")
        print(f"      Reference excluded:  {len(ref_excluded)}")
        print(f"      Missing from block list: {len(missing_excl)}")
        print(f"      Extra in block list:     {len(extra_excl)}")

        ref_total_momentum = ref_total_forces.sum(axis=0)
        ref_momentum_norm = np.linalg.norm(ref_total_momentum)
        momentum_diff = np.linalg.norm(total_force - ref_total_momentum)
        print(f"\n    Total momentum comparison:")
        print(f"      GPU total momentum norm:  {total_momentum_norm:.10e}")
        print(f"      Ref total momentum norm:  {ref_momentum_norm:.10e}")
        print(f"      Momentum difference norm: {momentum_diff:.10e}")
        print(f"      Momentum diff / max_force: {momentum_diff / max_force_mag:.10e}")

        if momentum_diff < 1e-4:
            print("      PASS: Momentum matches brute-force reference")
        else:
            print(f"      WARN: Momentum difference is {momentum_diff:.6e}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total block-list pairs:        {len(unique_pairs)}")
    print(f"  Duplicate pairs:               {len(duplicates)}")
    print(f"  Total momentum norm (GPU):     {total_momentum_norm:.10e}")
    if has_ref and missing_from_bl is not None:
        print(f"  Total momentum norm (ref):     {ref_momentum_norm:.10e}")
        print(f"  Momentum diff (GPU vs ref):    {momentum_diff:.10e}")
        print(f"  Missing non-excl pairs vs ref: {len(missing_from_bl)}")
    print("=" * 70)

    has_issues = False
    if total_momentum_norm > 1e-2:
        print("ISSUE: Total momentum is large — forces may not be conservative")
        has_issues = True
    if duplicates:
        print("ISSUE: Duplicate pairs detected — Newton's 3rd law violated")
        has_issues = True
    if has_ref and missing_from_bl and len(missing_from_bl) > 0:
        print("ISSUE: Block list is missing pairs vs reference")
        has_issues = True

    if not has_issues:
        print("All checks passed.")
    return 0 if not has_issues else 1


if __name__ == '__main__':
    sys.exit(main())
