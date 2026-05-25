"""Diagnostic: verify that _permute_all_arrays computes correct permutations
on the 2nd tile-list rebuild. Tests with 6PO6 (49 atoms)."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cupy as cp

from mdpy.forcefield.charmm_forcefield import CharmmForcefield
from mdpy.force.bonded_force import BondedForce
from mdpy.force.nonbonded_force import NonbondedForce
from mdpy.force.expressions.lennard_jones import lennard_jones
from mdpy.force.expressions.coulomb import coulomb
from mdpy.integrator.verlet import VerletIntegrator
from mdpy.system import System

DATA_DIR = 'mdpy/test/data'


def check_permutation_correctness():
    print("=" * 70)
    print("TEST: _permute_all_arrays permutation correctness")
    print("=" * 70)

    ff = CharmmForcefield(
        os.path.join(DATA_DIR, '6PO6.psf'),
        os.path.join(DATA_DIR, '6PO6.pdb'),
        [os.path.join(DATA_DIR, 'par_all36_prot.prm'),
         os.path.join(DATA_DIR, 'toppar_water_ions.str')],
    )
    topology = ff.create_topology()
    params = ff.create_parameter_table()

    box = np.eye(3, dtype=np.float32) * 30.0
    system = System(topology, box, cutoff=12.0, skin=1.0)

    bonded = BondedForce.charmm(topology, params)
    system.add_force_term(bonded)
    nb = NonbondedForce(lennard_jones + coulomb)
    nb.bind(topology, params, 12.0)
    system.add_force_term(nb)

    raw = ff._pdb.positions.astype(np.float64)
    pbc_inv = np.linalg.inv(box)
    frac = raw @ pbc_inv
    frac -= np.floor(frac)
    wrapped = (frac @ box).astype(np.float32)
    system.particles.positions[:] = wrapped
    system.particles.velocities[:] = 0.0

    N = topology.num_particles
    pdb_masses = topology.masses.copy()
    pdb_positions_init = system.particles.positions.copy()

    print(f"\nN = {N} atoms")

    # Upload and trigger first rebuild
    system.gpu.upload_positions(system.particles)
    system.gpu.upload_velocities(system.particles)
    system._positions_uploaded = True
    system._velocities_uploaded = True

    # --- First rebuild ---
    print("\n--- First rebuild ---")
    positions_soa = system.gpu.get_positions_2d()
    pdb_to_sorted_gpu, pdb_to_sorted_np = system.tile_list.rebuild(
        positions_soa, system.topology, system.pbc_matrix, system.pbc_inv
    )
    sorted_to_pdb_1 = cp.asnumpy(system.tile_list.d_sorted_to_pdb)

    system._permute_all_arrays(pdb_to_sorted_gpu, pdb_to_sorted_np)

    sorted_masses_1 = cp.asnumpy(system.gpu.d_masses)
    expected_masses_1 = pdb_masses[sorted_to_pdb_1]
    mass_err_1 = np.max(np.abs(sorted_masses_1 - expected_masses_1))
    print(f"  Mass error after 1st permute: {mass_err_1:.2e}")
    assert mass_err_1 < 1e-6, "BUG in first rebuild masses!"

    system.tile_list.build_tiles(system.topology, system.pbc_matrix)
    for term in system.force_terms:
        if hasattr(term, 'bind_sorted'):
            s2p = system._sorted_to_pdb_np()
            term.bind_sorted(system.topology, system.tile_list, system.gpu,
                             sorted_particle_types=system._particle_types_pdb[s2p])

    # --- Run integrator steps to move atoms ---
    print("\n--- Running 100 integrator steps (dt=0.5fs) ---")
    integrator = VerletIntegrator(0.5)
    for i in range(100):
        system.compute_forces()
        integrator.step(system.gpu)

    # Save current sorted positions and masses
    sorted_posx_before_2nd = cp.asnumpy(system.gpu.d_positions_x).copy()
    sorted_masses_before_2nd = cp.asnumpy(system.gpu.d_masses).copy()

    # Save old pdb_to_current_sorted
    old_p2c = system._pdb_to_current_sorted.copy()
    print(f"  old _pdb_to_current_sorted[:10]: {old_p2c[:10]}")

    # --- Second rebuild ---
    print("\n--- Second rebuild ---")
    positions_soa = system.gpu.get_positions_2d()
    pdb_to_sorted_gpu_2, pdb_to_sorted_np_2 = system.tile_list.rebuild(
        positions_soa, system.topology, system.pbc_matrix, system.pbc_inv
    )
    sorted_to_pdb_2 = cp.asnumpy(system.tile_list.d_sorted_to_pdb)
    pdb_to_sorted_2 = pdb_to_sorted_np_2.copy()

    # Compute correct permutation manually
    correct_perm = old_p2c[sorted_to_pdb_2]

    # Compute what the buggy code would do
    old_sorted_to_pdb = np.empty(N, dtype=np.int32)
    old_sorted_to_pdb[old_p2c] = np.arange(N, dtype=np.int32)
    buggy_perm = old_sorted_to_pdb[sorted_to_pdb_2]

    perm_match = np.array_equal(correct_perm, buggy_perm)
    diff_count = np.sum(correct_perm != buggy_perm)
    print(f"  Correct perm[:10]: {correct_perm[:10]}")
    print(f"  Buggy perm[:10]:   {buggy_perm[:10]}")
    print(f"  Match: {perm_match}, differing entries: {diff_count}/{N}")

    # Now call _permute_all_arrays for the SECOND time
    system._permute_all_arrays(pdb_to_sorted_gpu_2, pdb_to_sorted_np_2)

    sorted_masses_2 = cp.asnumpy(system.gpu.d_masses)
    expected_masses_2 = pdb_masses[sorted_to_pdb_2]
    mass_err_2 = np.max(np.abs(sorted_masses_2 - expected_masses_2))
    print(f"\n  Mass error after 2nd permute: {mass_err_2:.2e}")

    if mass_err_2 > 1e-6:
        wrong = np.where(np.abs(sorted_masses_2 - expected_masses_2) > 1e-6)[0]
        print(f"  *** BUG: {len(wrong)} atoms have wrong masses after 2nd rebuild! ***")
        for idx in wrong[:10]:
            print(f"    sorted_idx={idx}: got={sorted_masses_2[idx]:.6f}, "
                  f"expected={expected_masses_2[idx]:.6f}, pdb_idx={sorted_to_pdb_2[idx]}")

    # Check if positions are consistent
    sorted_posx_2 = cp.asnumpy(system.gpu.d_positions_x)
    # After correct permute: sorted_posx_2[i] = sorted_posx_before_2nd[correct_perm[i]]
    expected_posx_2 = sorted_posx_before_2nd[correct_perm]
    pos_err = np.max(np.abs(sorted_posx_2 - expected_posx_2))
    print(f"  Position error (vs correct perm): {pos_err:.2e}")

    # Also check against buggy perm
    buggy_posx = sorted_posx_before_2nd[buggy_perm]
    buggy_pos_err = np.max(np.abs(sorted_posx_2 - buggy_posx))
    print(f"  Position error (vs buggy perm): {buggy_pos_err:.2e}")

    # Check _pdb_to_current_sorted
    p2c_after = system._pdb_to_current_sorted
    p2c_err = np.max(np.abs(p2c_after - pdb_to_sorted_2))
    print(f"  _pdb_to_current_sorted error: {p2c_err:.2e}")

    print("\n" + "=" * 70)
    if mass_err_2 > 1e-6:
        print("RESULT: *** BUG CONFIRMED *** in _permute_all_arrays!")
        print(f"  Root cause: line ~110-112 in system.py uses old_sorted_to_pdb[new_sorted_to_pdb]")
        print(f"  instead of self._pdb_to_current_sorted[new_sorted_to_pdb]")
        print(f"  Fix: replace the permutation computation for 2nd+ rebuilds")
    elif not perm_match:
        print("RESULT: Permutation differs but masses happen to match (unlikely)")
    else:
        print("RESULT: No bug detected")
    print("=" * 70)


if __name__ == '__main__':
    check_permutation_correctness()
