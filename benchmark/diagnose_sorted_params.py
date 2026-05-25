"""Check sorted parameter consistency after tile list rebuild.

For each atom in PDB order, track its sorted index, position, and
per-atom parameters across rebuild to verify they stay consistent.
"""
import os
import numpy as np
import cupy as cp
from mdpy.forcefield.charmm_forcefield import CharmmForcefield
from mdpy.force.bonded_force import BondedForce
from mdpy.force.nonbonded_force import NonbondedForce
from mdpy.force.expressions.lennard_jones import lennard_jones
from mdpy.force.expressions.coulomb import coulomb
from mdpy.system import System

DATA_DIR = 'mdpy/test/data'
BOX = 108.0

ff = CharmmForcefield(
    os.path.join(DATA_DIR, '1M9Z.psf'),
    os.path.join(DATA_DIR, '1M9Z_minimized.pdb'),
    [os.path.join(DATA_DIR, 'par_all36_prot.prm'),
     os.path.join(DATA_DIR, 'toppar_water_ions.str')])
topology = ff.create_topology()
parameter_table = ff.create_parameter_table()
pbc_matrix = np.eye(3, dtype=np.float64) * BOX
pbc_inv = np.linalg.inv(pbc_matrix)

# Build per-atom parameter tables in PDB order (ground truth)
N = topology.num_particles
ptypes = topology.particle_types.copy()
sigma_pdb = parameter_table.expand_to_per_atom('sigma', ptypes)
epsilon_pdb = parameter_table.expand_to_per_atom('epsilon', ptypes)
charge_pdb = parameter_table.expand_to_per_atom('charge', ptypes)
sigma_half_pdb = 0.5 * sigma_pdb
sqrt_epsilon_pdb = np.sqrt(np.maximum(epsilon_pdb, 0.0)).astype(np.float32)
charge_pdb = charge_pdb.astype(np.float32)
sigma_half_pdb = sigma_half_pdb.astype(np.float32)

print(f"N = {N}")
print(f"Ground truth params (PDB order) first 10:")
for i in range(10):
    print(f"  atom {i}: type={ptypes[i]}, sigma_half={sigma_half_pdb[i]:.6f}, sqrt_eps={sqrt_epsilon_pdb[i]:.6f}, charge={charge_pdb[i]:.6f}")

# Upload positions
raw = ff._pdb.positions.astype(np.float64)
frac = raw @ pbc_inv
frac -= np.floor(frac)
wrapped = frac @ pbc_matrix

system = System(topology, pbc_matrix, cutoff=12.0)
system.add_force_term(BondedForce.charmm(topology, parameter_table))
nb = NonbondedForce(lennard_jones + coulomb)
nb.bind(topology, parameter_table, 12.0)
system.add_force_term(nb)

system.particles.positions[:] = wrapped.astype(np.float32)
system.particles.velocities[:] = 0.0
system.gpu.upload_positions(system.particles)
system.gpu.upload_velocities(system.particles)

# ===== First rebuild =====
print("\n" + "="*80)
print("  FIRST REBUILD")
print("="*80)

positions_soa = system.gpu.get_positions_2d()
system.tile_list.check_rebuild(positions_soa)
pdb_to_sorted_gpu, pdb_to_sorted_np = system.tile_list.rebuild(
    positions_soa, system.topology, system.pbc_matrix, system.pbc_inv)
system._permute_all_arrays(pdb_to_sorted_gpu, pdb_to_sorted_np)
system.tile_list.build_tiles(system.topology, system.pbc_matrix)
for term in system.force_terms:
    if hasattr(term, 'bind_sorted'):
        s2p = system._sorted_to_pdb_np()
        term.bind_sorted(
            system.topology, system.tile_list, system.gpu,
            sorted_particle_types=system._particle_types_pdb[s2p],
        )

# After first rebuild, get sorted_to_pdb mapping
s2p_1 = cp.asnumpy(system.tile_list.d_sorted_to_pdb)
print(f"\n  sorted_to_pdb: shape={s2p_1.shape}, dtype={s2p_1.dtype}")
print(f"  sorted_to_pdb first 20: {s2p_1[:20]}")
print(f"  unique values: {len(np.unique(s2p_1))} (should be {N})")
print(f"  range: [{s2p_1.min()}, {s2p_1.max()}]")

# Check GPU positions vs expected sorted positions
gx = cp.asnumpy(system.gpu.d_positions_x)
gy = cp.asnumpy(system.gpu.d_positions_y)
gz = cp.asnumpy(system.gpu.d_positions_z)

# Ground truth: position of atom s2p_1[i] in PDB order
expected_px = wrapped[s2p_1, 0].astype(np.float32)
expected_py = wrapped[s2p_1, 1].astype(np.float32)
expected_pz = wrapped[s2p_1, 2].astype(np.float32)

err_x = np.abs(gx - expected_px)
err_y = np.abs(gy - expected_py)
err_z = np.abs(gz - expected_pz)
print(f"\n  Position check (sorted[i] should == pdb_pos[sorted_to_pdb[i]]):")
print(f"    max err x: {err_x.max():.8f}  mean: {err_x.mean():.10f}")
print(f"    max err y: {err_y.max():.8f}  mean: {err_y.mean():.10f}")
print(f"    max err z: {err_z.max():.8f}  mean: {err_z.mean():.10f}")
pos_mismatch = (err_x > 0.01) | (err_y > 0.01) | (err_z > 0.01)
print(f"    mismatches (>0.01): {pos_mismatch.sum()}")

# Check NonbondedForce parameter arrays
nb = system.force_terms[1]
print(f"\n  NonbondedForce._parameter_arrays after bind_sorted:")
for name, arr in nb._parameter_arrays.items():
    gpu_arr = nb._d_parameter_arrays.get(name)
    if gpu_arr is not None:
        gpu_vals = cp.asnumpy(gpu_arr)
        print(f"    {name}: shape={gpu_vals.shape}, first 10 = {gpu_vals[:10]}")
        
        # Check: for sorted index i, param should be pdb_param[sorted_to_pdb[i]]
        if name == 'sigma_half':
            expected = sigma_half_pdb[s2p_1]
            err = np.abs(gpu_vals - expected)
            print(f"      vs pdb[s2p]: max_err={err.max():.8f}, mismatches(>0.001)={np.sum(err>0.001)}")
            mismatch_idx = np.where(err > 0.001)[0]
            if len(mismatch_idx) > 0:
                for idx in mismatch_idx[:5]:
                    print(f"      sorted_idx={idx}, s2p={s2p_1[idx]}, gpu={gpu_vals[idx]:.6f}, expected={expected[idx]:.6f}")
        elif name == 'sqrt_epsilon':
            expected = sqrt_epsilon_pdb[s2p_1]
            err = np.abs(gpu_vals - expected)
            print(f"      vs pdb[s2p]: max_err={err.max():.8f}, mismatches(>0.001)={np.sum(err>0.001)}")
        elif name == 'charge':
            expected = charge_pdb[s2p_1]
            err = np.abs(gpu_vals - expected)
            print(f"      vs pdb[s2p]: max_err={err.max():.8f}, mismatches(>0.001)={np.sum(err>0.001)}")
        elif name == 'sigma_epsilon':
            expected_sh = sigma_half_pdb[s2p_1]
            expected_se = sqrt_epsilon_pdb[s2p_1]
            gpu_sh = gpu_vals[0::2]
            gpu_se = gpu_vals[1::2]
            err_sh = np.abs(gpu_sh - expected_sh)
            err_se = np.abs(gpu_se - expected_se)
            print(f"      sigma_half: max_err={err_sh.max():.8f}, mismatches(>0.001)={np.sum(err_sh>0.001)}")
            print(f"      sqrt_epsilon: max_err={err_se.max():.8f}, mismatches(>0.001)={np.sum(err_se>0.001)}")
            se_mismatch = (err_sh > 0.001) | (err_se > 0.001)
            if se_mismatch.sum() > 0:
                idx = np.where(se_mismatch)[0][:5]
                for i in idx:
                    print(f"      sorted_idx={i}, s2p={s2p_1[i]}, gpu_sh={gpu_sh[i]:.6f} vs {expected_sh[i]:.6f}, gpu_se={gpu_se[i]:.6f} vs {expected_se[i]:.6f}")
        elif name == 'sigma_epsilon_14':
            pass

# Check sorted params in tile_list
print(f"\n  TileList sorted params:")
for attr in ['d_sorted_sigma_epsilon', 'd_sorted_sigma_epsilon_14',
             'd_sorted_charge_14', 'd_sorted_posq']:
    arr = getattr(system.tile_list, attr, None)
    if arr is not None:
        vals = cp.asnumpy(arr)
        print(f"    {attr}: shape={vals.shape}, first 10 = {vals[:10].ravel()}")
    else:
        print(f"    {attr}: not found")

# Check _sorted_to_pdb_np (the inverse mapping)
s2p_func = system._sorted_to_pdb_np()
print(f"\n  _sorted_to_pdb_np(): shape={s2p_func.shape}, first 20 = {s2p_func[:20]}")
print(f"  Same as d_sorted_to_pdb? {np.array_equal(s2p_func, s2p_1)}")

# Check system._pdb_to_current_sorted
print(f"\n  _pdb_to_current_sorted: shape={system._pdb_to_current_sorted.shape}, first 20 = {system._pdb_to_current_sorted[:20]}")
# pdb_to_current_sorted[pdb_idx] = sorted_idx
# So s2p = pdb_to_current_sorted should be the inverse: s2p[sorted_idx] = pdb_idx
# Verify: s2p_1[i] should equal _pdb_to_current_sorted's inverse
inv = np.empty(N, dtype=np.int32)
inv[system._pdb_to_current_sorted] = np.arange(N, dtype=np.int32)
# inv[pdb_idx] = sorted_idx
# s2p_1[sorted_idx] = pdb_idx
# So inv[s2p_1[i]] should equal i
check = inv[s2p_1]
print(f"  inv[s2p_1] == arange(N)? {np.array_equal(check, np.arange(N))}")

# ===== Simulate a few steps, trigger second rebuild, check again =====
from mdpy.integrator.verlet import VerletIntegrator
integrator = VerletIntegrator(0.5)

# Do step 0
system.compute_forces()
integrator.step(system.gpu)

# Step 1-5
for s in range(1, 6):
    positions_soa = system.gpu.get_positions_2d()
    need = system.tile_list.check_rebuild(positions_soa)
    if need:
        print(f"\n{'='*80}")
        print(f"  SECOND REBUILD at step {s}")
        print(f"{'='*80}")
        
        pdb_to_sorted_gpu, pdb_to_sorted_np = system.tile_list.rebuild(
            positions_soa, system.topology, system.pbc_matrix, system.pbc_inv)
        
        # Before permute, save the sorted_to_pdb from tile_list
        s2p_new = cp.asnumpy(system.tile_list.d_sorted_to_pdb)
        print(f"  NEW sorted_to_pdb first 20: {s2p_new[:20]}")
        print(f"  Same as first? {np.array_equal(s2p_new, s2p_1)}")
        
        system._permute_all_arrays(pdb_to_sorted_gpu, pdb_to_sorted_np)
        system.tile_list.build_tiles(system.topology, system.pbc_matrix)
        for term in system.force_terms:
            if hasattr(term, 'bind_sorted'):
                s2p_func2 = system._sorted_to_pdb_np()
                term.bind_sorted(
                    system.topology, system.tile_list, system.gpu,
                    sorted_particle_types=system._particle_types_pdb[s2p_func2],
                )
        
        # After second rebuild + permute + bind_sorted, check again
        s2p_2 = cp.asnumpy(system.tile_list.d_sorted_to_pdb)
        gx2 = cp.asnumpy(system.gpu.d_positions_x)
        gy2 = cp.asnumpy(system.gpu.d_positions_y)
        gz2 = cp.asnumpy(system.gpu.d_positions_z)
        
        print(f"\n  After 2nd rebuild + permute:")
        print(f"  sorted_to_pdb first 20: {s2p_2[:20]}")
        
        # The positions are now in new sorted order
        # For sorted index i, the atom is s2p_2[i] in PDB space
        # Its position should be what it was BEFORE the rebuild
        # But _permute_all_arrays rearranged positions using perm = s2p_2
        # permute kernel: new[i] = old[perm[i]] = old[s2p_2[i]]
        # But old is in OLD sorted order, not PDB order!
        # So new[i] = old_sorted[s2p_2[i]], which is WRONG unless s2p_2 maps
        # to old sorted indices (but it maps to PDB indices)
        
        # Let's check: are positions correct?
        # The correct position at sorted index i should be:
        # position of PDB atom s2p_2[i] in the CURRENT state
        # But we don't have "current PDB order positions" easily...
        # We can check if positions make sense (no huge jumps)
        
        print(f"  pos x: [{gx2.min():.4f}, {gx2.max():.4f}]")
        print(f"  pos y: [{gy2.min():.4f}, {gy2.max():.4f}]")
        print(f"  pos z: [{gz2.min():.4f}, {gz2.max():.4f}]")
        
        # Check params
        nb2 = system.force_terms[1]
        for name, arr in nb2._parameter_arrays.items():
            if name in ('sigma_epsilon', 'sigma_epsilon_14', 'charge_14'):
                continue
            gpu_arr = nb2._d_parameter_arrays.get(name)
            if gpu_arr is not None:
                gpu_vals = cp.asnumpy(gpu_arr)
                if name == 'sigma_half':
                    expected = sigma_half_pdb[s2p_2]
                elif name == 'sqrt_epsilon':
                    expected = sqrt_epsilon_pdb[s2p_2]
                elif name == 'charge':
                    expected = charge_pdb[s2p_2]
                else:
                    continue
                err = np.abs(gpu_vals - expected)
                mismatches = np.sum(err > 0.001)
                print(f"  {name}: max_err={err.max():.8f}, mismatches={mismatches}")
                if mismatches > 0:
                    idx = np.where(err > 0.001)[0][:5]
                    for i in idx:
                        print(f"    sorted_idx={i}, s2p={s2p_2[i]}, gpu={gpu_vals[i]:.6f}, expected={expected[i]:.6f}")
        
        break
    
    system.compute_forces()
    integrator.step(system.gpu)
else:
    print("\n  No second rebuild triggered in steps 1-5")
    print("  Forcing a rebuild by manually triggering...")
    
    # Force check by moving an atom far
    gx = cp.asnumpy(system.gpu.d_positions_x)
    gx[0] += 50.0
    system.gpu.d_positions_x[:] = cp.asarray(gx)
    
    positions_soa = system.gpu.get_positions_2d()
    need = system.tile_list.check_rebuild(positions_soa)
    print(f"  check_rebuild after forced displacement: {need}")
    
    if need:
        pdb_to_sorted_gpu, pdb_to_sorted_np = system.tile_list.rebuild(
            positions_soa, system.topology, system.pbc_matrix, system.pbc_inv)
        s2p_new = cp.asnumpy(system.tile_list.d_sorted_to_pdb)
        
        system._permute_all_arrays(pdb_to_sorted_gpu, pdb_to_sorted_np)
        system.tile_list.build_tiles(system.topology, system.pbc_matrix)
        for term in system.force_terms:
            if hasattr(term, 'bind_sorted'):
                s2p_func2 = system._sorted_to_pdb_np()
                term.bind_sorted(
                    system.topology, system.tile_list, system.gpu,
                    sorted_particle_types=system._particle_types_pdb[s2p_func2],
                )
        
        s2p_2 = cp.asnumpy(system.tile_list.d_sorted_to_pdb)
        gx2 = cp.asnumpy(system.gpu.d_positions_x)
        
        print(f"\n  After forced rebuild:")
        print(f"  sorted_to_pdb first 20: {s2p_2[:20]}")
        
        # Check params
        nb2 = system.force_terms[1]
        for name in ['sigma_half', 'sqrt_epsilon', 'charge']:
            gpu_arr = nb2._d_parameter_arrays.get(name)
            if gpu_arr is not None:
                gpu_vals = cp.asnumpy(gpu_arr)
                pdb_map = {'sigma_half': sigma_half_pdb, 'sqrt_epsilon': sqrt_epsilon_pdb, 'charge': charge_pdb}
                expected = pdb_map[name][s2p_2]
                err = np.abs(gpu_vals - expected)
                mismatches = np.sum(err > 0.001)
                print(f"  {name}: max_err={err.max():.8f}, mismatches={mismatches}")
                if mismatches > 0:
                    idx = np.where(err > 0.001)[0][:10]
                    for i in idx:
                        print(f"    sorted_idx={i}, s2p={s2p_2[i]}, gpu={gpu_vals[i]:.6f}, expected={expected[i]:.6f}, pdb_val={pdb_map[name][s2p_2[i]]:.6f}")
