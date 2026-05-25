"""Step-by-step energy diagnostic for 1M9Z.

Prints ALL GPU computation parameters at each step to identify
why energy explodes.
"""
import os
import sys
import numpy as np
import cupy as cp

from benchmark._data_path import DATA_DIR
PSF_PATH = os.path.join(DATA_DIR, '1M9Z.psf')
PDB_PATH = os.path.join(DATA_DIR, '1M9Z_minimized.pdb')
PRM_PATH = os.path.join(DATA_DIR, 'par_all36_prot.prm')
STR_PATH = os.path.join(DATA_DIR, 'toppar_water_ions.str')

BOX_SIZE = 108.0
CUTOFF = 12.0
DT_FS = 0.5
NUM_STEPS = 20
KCAL_PER_INTERNAL = 1.0 / 4.1840286576e-4


def print_separator(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def main():
    from mdpy.forcefield.charmm_forcefield import CharmmForcefield
    from mdpy.force.bonded_force import BondedForce
    from mdpy.force.nonbonded_force import NonbondedForce
    from mdpy.force.expressions.lennard_jones import lennard_jones
    from mdpy.force.expressions.coulomb import coulomb
    from mdpy.integrator.verlet import VerletIntegrator
    from mdpy.system import System

    print_separator("STEP 0: System Setup")

    ff = CharmmForcefield(PSF_PATH, PDB_PATH, [PRM_PATH, STR_PATH])
    topology = ff.create_topology()
    parameter_table = ff.create_parameter_table()

    print(f"  num_particles:        {topology.num_particles}")
    print(f"  num_bonds:            {topology.bond_indices.shape[0] if hasattr(topology, 'bond_indices') and topology.bond_indices is not None else 0}")
    print(f"  num_angles:           {topology.angle_indices.shape[0] if hasattr(topology, 'angle_indices') and topology.angle_indices is not None else 0}")
    print(f"  num_dihedrals:        {topology.dihedral_indices.shape[0] if hasattr(topology, 'dihedral_indices') and topology.dihedral_indices is not None else 0}")
    print(f"  num_impropers:        {topology.improper_indices.shape[0] if hasattr(topology, 'improper_indices') and topology.improper_indices is not None else 0}")
    print(f"  num_particle_types:   {len(np.unique(topology.particle_types))}")

    print(f"\n  --- Topology masses (first 20) ---")
    print(f"  {topology.masses[:20]}")
    print(f"  mass min={topology.masses.min():.6f} max={topology.masses.max():.6f}")
    print(f"  mass zeros: {(topology.masses == 0).sum()}")

    print(f"\n  --- Particle types (first 20) ---")
    print(f"  {topology.particle_types[:20]}")

    print(f"\n  --- Charges (first 20) ---")
    if 'charge' in parameter_table.per_atom:
        charges = parameter_table.per_atom['charge']
    elif 'charge' in parameter_table.per_type:
        charges = parameter_table.expand_to_per_atom('charge', topology.particle_types)
    else:
        charges = np.zeros(topology.num_particles)
    print(f"  {charges[:20]}")
    print(f"  charge min={charges.min():.6f} max={charges.max():.6f}")
    print(f"  charge sum={charges.sum():.6f}")

    print(f"\n  --- LJ parameters (first 20 atoms) ---")
    sigma = parameter_table.expand_to_per_atom('sigma', topology.particle_types)
    epsilon = parameter_table.expand_to_per_atom('epsilon', topology.particle_types)
    sigma_half = 0.5 * sigma
    sqrt_epsilon = np.sqrt(np.maximum(epsilon, 0.0))
    print(f"  sigma_half:  {sigma_half[:20]}")
    print(f"  sqrt_epsilon:{sqrt_epsilon[:20]}")
    print(f"  sigma_half  min={sigma_half.min():.6f} max={sigma_half.max():.6f}")
    print(f"  sqrt_epsilon min={sqrt_epsilon.min():.6f} max={sqrt_epsilon.max():.6f}")
    print(f"  epsilon zeros: {(epsilon == 0).sum()} / {len(epsilon)}")

    print(f"\n  --- 1-4 parameters (first 20 atoms) ---")
    has_14_sigma = 'sigma_14' in parameter_table.per_type or 'sigma_14' in parameter_table.per_atom
    has_14_eps = 'epsilon_14' in parameter_table.per_type or 'epsilon_14' in parameter_table.per_atom
    print(f"  has sigma_14: {has_14_sigma}")
    print(f"  has epsilon_14: {has_14_eps}")
    if has_14_sigma:
        sigma_14 = parameter_table.expand_to_per_atom('sigma_14', topology.particle_types)
        eps_14 = parameter_table.expand_to_per_atom('epsilon_14', topology.particle_types)
        sh_14 = 0.5 * sigma_14
        se_14 = np.sqrt(np.maximum(eps_14, 0.0))
        print(f"  sigma_half_14:  {sh_14[:20]}")
        print(f"  sqrt_epsilon_14:{se_14[:20]}")
        print(f"  sigma_half_14  min={sh_14.min():.6f} max={sh_14.max():.6f}")
        print(f"  sqrt_epsilon_14 min={se_14.min():.6f} max={se_14.max():.6f}")

    print(f"\n  --- Bonded parameters sample ---")
    if 'bond' in parameter_table.per_term:
        bp = parameter_table.per_term['bond']
        print(f"  bond params shape: {bp.shape}")
        print(f"  bond k  (first 5): {bp[:5, 0]}")
        print(f"  bond r0 (first 5): {bp[:5, 1]}")
        print(f"  bond k  min={bp[:,0].min():.4f} max={bp[:,0].max():.4f}")
        print(f"  bond r0 min={bp[:,1].min():.4f} max={bp[:,1].max():.4f}")
    if 'angle' in parameter_table.per_term:
        ap = parameter_table.per_term['angle']
        print(f"  angle params shape: {ap.shape}")
        print(f"  angle k (first 5): {ap[:5, 0]}")
        print(f"  angle theta0 (first 5): {ap[:5, 1]}")
    if 'dihedral' in parameter_table.per_term:
        dp = parameter_table.per_term['dihedral']
        print(f"  dihedral params shape: {dp.shape}")
        print(f"  dihedral k (first 5): {dp[:5, 0]}")
        print(f"  dihedral n (first 5): {dp[:5, 1]}")
        print(f"  dihedral delta (first 5): {dp[:5, 2]}")

    print(f"\n  --- PBC ---")
    print(f"  box_size: {BOX_SIZE}")
    print(f"  cutoff:   {CUTOFF}")

    pbc_matrix = np.eye(3, dtype=np.float64) * BOX_SIZE
    pbc_inv = np.linalg.inv(pbc_matrix)
    system = System(topology, pbc_matrix, cutoff=CUTOFF)

    print(f"\n  --- PBC on GPU ---")
    d_pbc = cp.asnumpy(system.gpu.d_pbc_matrix)
    d_pbc_inv = cp.asnumpy(system.gpu.d_pbc_inv)
    print(f"  d_pbc_matrix: {d_pbc}")
    print(f"  d_pbc_inv:    {d_pbc_inv}")
    print(f"  pbc_matrix reshaped:\n{d_pbc.reshape(3,3)}")
    print(f"  pbc_inv reshaped:\n{d_pbc_inv.reshape(3,3)}")

    print(f"\n  --- Box dims on GPU ---")
    system.gpu.set_box_dims(BOX_SIZE, BOX_SIZE, BOX_SIZE)
    d_box = cp.asnumpy(system.gpu.d_box_dims)
    print(f"  d_box_dims: {d_box}")
    print(f"  _box_x={system.gpu._box_x}, _box_y={system.gpu._box_y}, _box_z={system.gpu._box_z}")
    print(f"  _inv_box_x={system.gpu._inv_box_x:.8f}, _inv_box_y={system.gpu._inv_box_y:.8f}, _inv_box_z={system.gpu._inv_box_z:.8f}")

    system.add_force_term(BondedForce.charmm(topology, parameter_table))
    nb = NonbondedForce(lennard_jones + coulomb)
    nb.bind(topology, parameter_table, CUTOFF)
    system.add_force_term(nb)

    print(f"\n  --- NonbondedForce internal state ---")
    print(f"  cutoff: {nb._cutoff}")
    print(f"  cutoff_sq: {nb._cutoff_sq}")
    print(f"  use_posq: {nb._use_posq()}")
    print(f"  expression parameter_names: {nb.expression.parameter_names}")
    for pname in nb.expression.parameter_names:
        arr = nb._parameter_arrays.get(pname)
        if arr is not None:
            print(f"  {pname}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.6f}, max={arr.max():.6f}, has_nan={np.any(np.isnan(arr))}, has_inf={np.any(np.isinf(arr))}")
    print(f"\n  CUDA fragment (LJ+Coulomb):\n{nb.expression.cuda_fragment}")

    raw = ff._pdb.positions.astype(np.float64)
    frac = raw @ pbc_inv
    frac -= np.floor(frac)
    wrapped = frac @ pbc_matrix
    system.particles.positions[:] = wrapped
    system.particles.velocities[:] = 0.0
    system.gpu.upload_positions(system.particles)
    system.gpu.upload_velocities(system.particles)

    print(f"\n  --- Initial positions (PDB, first 10) ---")
    print(f"  raw PDB pos (first 5):\n{ff._pdb.positions[:5]}")
    print(f"  wrapped pos (first 5):\n{wrapped[:5]}")
    print(f"  pos x min={wrapped[:,0].min():.4f} max={wrapped[:,0].max():.4f}")
    print(f"  pos y min={wrapped[:,1].min():.4f} max={wrapped[:,1].max():.4f}")
    print(f"  pos z min={wrapped[:,2].min():.4f} max={wrapped[:,2].max():.4f}")
    print(f"  any outside box [0,{BOX_SIZE}]? {np.any(wrapped < -0.01) or np.any(wrapped > BOX_SIZE + 0.01)}")

    print(f"\n  --- Positions on GPU (first 10) ---")
    gx = cp.asnumpy(system.gpu.d_positions_x[:10])
    gy = cp.asnumpy(system.gpu.d_positions_y[:10])
    gz = cp.asnumpy(system.gpu.d_positions_z[:10])
    print(f"  d_positions_x[:10]: {gx}")
    print(f"  d_positions_y[:10]: {gy}")
    print(f"  d_positions_z[:10]: {gz}")

    print(f"\n  --- Velocities on GPU (first 10) ---")
    vx = cp.asnumpy(system.gpu.d_velocities_x[:10])
    vy = cp.asnumpy(system.gpu.d_velocities_y[:10])
    vz = cp.asnumpy(system.gpu.d_velocities_z[:10])
    print(f"  d_velocities_x[:10]: {vx}")
    print(f"  all velocities zero? {cp.all(system.gpu.d_velocities_x == 0) and cp.all(system.gpu.d_velocities_y == 0) and cp.all(system.gpu.d_velocities_z == 0)}")

    print(f"\n  --- Masses on GPU (first 20) ---")
    dm = cp.asnumpy(system.gpu.d_masses[:20])
    print(f"  {dm}")
    d_mass_all = cp.asnumpy(system.gpu.d_masses)
    print(f"  mass min={d_mass_all.min():.6f} max={d_mass_all.max():.6f}")
    print(f"  mass zeros: {(d_mass_all == 0).sum()}")

    integrator = VerletIntegrator(DT_FS)
    print(f"  dt={integrator.dt}, dt_sq={integrator.dt_sq}")

    print_separator("STEP 0.5: Initial Force Computation (before any integration)")
    system.compute_forces()
    energies_init = system.dump_energy()
    print(f"  Initial energies (internal units):")
    for name, val in energies_init.items():
        print(f"    {name}: {val:.6f} ({val * KCAL_PER_INTERNAL:.2f} kcal/mol)")
    e_total_init = sum(energies_init.values())
    print(f"  TOTAL: {e_total_init:.6f} ({e_total_init * KCAL_PER_INTERNAL:.2f} kcal/mol)")

    fx = cp.asnumpy(system.gpu.d_forces_x)
    fy = cp.asnumpy(system.gpu.d_forces_y)
    fz = cp.asnumpy(system.gpu.d_forces_z)
    print(f"\n  Forces (all atoms):")
    print(f"    fx: min={fx.min():.6f} max={fx.max():.6f} mean={fx.mean():.6f} std={fx.std():.6f}")
    print(f"    fy: min={fy.min():.6f} max={fy.max():.6f} mean={fy.mean():.6f} std={fy.std():.6f}")
    print(f"    fz: min={fz.min():.6f} max={fz.max():.6f} mean={fz.mean():.6f} std={fz.std():.6f}")
    fmag = np.sqrt(fx**2 + fy**2 + fz**2)
    print(f"    |f|: min={fmag.min():.6f} max={fmag.max():.6f} mean={fmag.mean():.6f}")
    print(f"    any nan? {np.any(np.isnan(fx)) or np.any(np.isnan(fy)) or np.any(np.isnan(fz))}")
    print(f"    any inf? {np.any(np.isinf(fx)) or np.any(np.isinf(fy)) or np.any(np.isinf(fz))}")
    top_force_idx = np.argsort(fmag)[-5:][::-1]
    print(f"    Top 5 force atoms:")
    for idx in top_force_idx:
        print(f"      atom {idx}: fx={fx[idx]:.4f} fy={fy[idx]:.4f} fz={fz[idx]:.4f} |f|={fmag[idx]:.4f}")

    print_separator("RUNNING STEPS")
    print(f"  {'Step':>6s}  {'E_bonded':>16s}  {'E_nonbonded':>16s}  {'E_total':>16s}  {'E_total(kcal)':>16s}  {'max|f|':>12s}  {'pos_range':>30s}  {'vel_max':>12s}")

    for step in range(NUM_STEPS):
        positions_soa = system.gpu.get_positions_2d()
        if system.tile_list.check_rebuild(positions_soa):
            pdb_to_sorted_gpu, pdb_to_sorted_np = system.tile_list.rebuild(
                positions_soa, system.topology,
                system.pbc_matrix, system.pbc_inv,
            )
            system._permute_all_arrays(pdb_to_sorted_gpu, pdb_to_sorted_np)
            system.tile_list.build_tiles(system.topology, system.pbc_matrix)
            for term in system.force_terms:
                if hasattr(term, 'bind_sorted'):
                    s2p = system._sorted_to_pdb_np()
                    term.bind_sorted(
                        system.topology, system.tile_list, system.gpu,
                        sorted_particle_types=system._particle_types_pdb[s2p],
                    )

        system.compute_forces()

        energies = system.dump_energy()
        e_total = sum(energies.values())
        e_bonded = energies.get('bonded', 0.0)
        e_nb = energies.get('nonbonded', 0.0)

        fx = cp.asnumpy(system.gpu.d_forces_x)
        fy = cp.asnumpy(system.gpu.d_forces_y)
        fz = cp.asnumpy(system.gpu.d_forces_z)
        fmag = np.sqrt(fx**2 + fy**2 + fz**2)
        max_force = fmag.max()

        px = cp.asnumpy(system.gpu.d_positions_x)
        py = cp.asnumpy(system.gpu.d_positions_y)
        pz = cp.asnumpy(system.gpu.d_positions_z)

        vx = cp.asnumpy(system.gpu.d_velocities_x)
        vy = cp.asnumpy(system.gpu.d_velocities_y)
        vz = cp.asnumpy(system.gpu.d_velocities_z)
        vel_mag = np.sqrt(vx**2 + vy**2 + vz**2)
        max_vel = vel_mag.max()

        has_nan_f = np.any(np.isnan(fx)) or np.any(np.isnan(fy)) or np.any(np.isnan(fz))
        has_inf_f = np.any(np.isinf(fx)) or np.any(np.isinf(fy)) or np.any(np.isinf(fz))
        has_nan_p = np.any(np.isnan(px)) or np.any(np.isnan(py)) or np.any(np.isnan(pz))
        has_nan_v = np.any(np.isnan(vx)) or np.any(np.isnan(vy)) or np.any(np.isnan(vz))

        pos_range = f"x[{px.min():.2f},{px.max():.2f}] y[{py.min():.2f},{py.max():.2f}] z[{pz.min():.2f},{pz.max():.2f}]"
        flags = ""
        if has_nan_f: flags += " NAN_F"
        if has_inf_f: flags += " INF_F"
        if has_nan_p: flags += " NAN_P"
        if has_nan_v: flags += " NAN_V"

        print(f"  {step:6d}  {e_bonded:16.6e}  {e_nb:16.6e}  {e_total:16.6e}  {e_total*KCAL_PER_INTERNAL:16.2f}  {max_force:12.4f}  {pos_range}  {max_vel:12.6f}{flags}")

        if abs(e_total * KCAL_PER_INTERNAL) > 1e10:
            print(f"\n  *** ENERGY EXPLOSION DETECTED at step {step} ***")
            print(f"  Dumping detailed state...")
            top_force_idx = np.argsort(fmag)[-10:][::-1]
            print(f"  Top 10 force atoms:")
            for idx in top_force_idx:
                print(f"    atom {idx}: pos=({px[idx]:.4f},{py[idx]:.4f},{pz[idx]:.4f}) f=({fx[idx]:.4f},{fy[idx]:.4f},{fz[idx]:.4f}) |f|={fmag[idx]:.4f} v=({vx[idx]:.6f},{vy[idx]:.6f},{vz[idx]:.6f}) mass={d_mass_all[idx]:.6f}")
            break

        integrator.step(system.gpu)

    print_separator("FINAL STATE ANALYSIS")
    px = cp.asnumpy(system.gpu.d_positions_x)
    py = cp.asnumpy(system.gpu.d_positions_y)
    pz = cp.asnumpy(system.gpu.d_positions_z)
    print(f"  Position ranges:")
    print(f"    x: [{px.min():.4f}, {px.max():.4f}]")
    print(f"    y: [{py.min():.4f}, {py.max():.4f}]")
    print(f"    z: [{pz.min():.4f}, {pz.max():.4f}]")
    print(f"  Any outside [0, {BOX_SIZE}]? {np.any(px < -0.1) or np.any(px > BOX_SIZE+0.1) or np.any(py < -0.1) or np.any(py > BOX_SIZE+0.1) or np.any(pz < -0.1) or np.any(pz > BOX_SIZE+0.1)}")

    vx = cp.asnumpy(system.gpu.d_velocities_x)
    vy = cp.asnumpy(system.gpu.d_velocities_y)
    vz = cp.asnumpy(system.gpu.d_velocities_z)
    print(f"  Velocity ranges:")
    print(f"    vx: [{vx.min():.6f}, {vx.max():.6f}]")
    print(f"    vy: [{vy.min():.6f}, {vy.max():.6f}]")
    print(f"    vz: [{vz.min():.6f}, {vz.max():.6f}]")


if __name__ == '__main__':
    main()
