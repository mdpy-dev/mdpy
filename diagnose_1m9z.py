"""Comprehensive diagnostic for 1M9Z energy explosion.
Runs step by step, checks ALL sorted parameters at every rebuild."""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cupy as cp

DATA_DIR = 'mdpy/test/data'

def main():
    from mdpy.forcefield.charmm_forcefield import CharmmForcefield
    from mdpy.force.bonded_force import BondedForce
    from mdpy.force.nonbonded_force import NonbondedForce
    from mdpy.force.expressions.lennard_jones import lennard_jones
    from mdpy.force.expressions.coulomb import coulomb
    from mdpy.integrator.verlet import VerletIntegrator
    from mdpy.system import System

    ff = CharmmForcefield(
        os.path.join(DATA_DIR, '1M9Z.psf'),
        os.path.join(DATA_DIR, '1M9Z_minimized.pdb'),
        [os.path.join(DATA_DIR, 'par_all36_prot.prm'),
         os.path.join(DATA_DIR, 'toppar_water_ions.str')],
    )
    topology = ff.create_topology()
    params = ff.create_parameter_table()

    N = topology.num_particles
    BOX = 108.0
    CUTOFF = 12.0
    DT = 0.5

    pbc_matrix = np.eye(3, dtype=np.float32) * BOX

    system = System(topology, pbc_matrix, cutoff=CUTOFF)
    bonded = BondedForce.charmm(topology, params)
    system.add_force_term(bonded)
    nb = NonbondedForce(lennard_jones + coulomb)
    nb.bind(topology, params, CUTOFF)
    system.add_force_term(nb)

    raw = ff._pdb.positions.astype(np.float64)
    pbc_inv = np.linalg.inv(pbc_matrix.astype(np.float64))
    frac = raw @ pbc_inv
    frac -= np.floor(frac)
    wrapped = (frac @ pbc_matrix.astype(np.float64)).astype(np.float32)
    system.particles.positions[:] = wrapped
    system.particles.velocities[:] = 0.0
    system.gpu.upload_positions(system.particles)
    system.gpu.upload_velocities(system.particles)

    # Save PDB-order references
    pdb_masses = topology.masses.copy()
    pdb_charges = topology.charges.copy()
    pdb_types = topology.particle_types.copy()

    KCAL = 1.0 / 4.1840286576e-4

    integrator = VerletIntegrator(DT)
    rebuild_count = 0

    print(f"N={N}, Box={BOX}, Cutoff={CUTOFF}, dt={DT}fs")
    print(f"{'Step':>5} {'Rebuild':>7} {'E_bonded':>16} {'E_nonbond':>16} {'E_total':>16} {'MassErr':>10} {'ChargeErr':>10}")
    print(f"{'':>5} {'':>7} {'(kcal)':>16} {'(kcal)':>16} {'(kcal)':>16}")

    for step in range(200):
        # Check rebuild manually
        positions_soa = system.gpu.get_positions_2d()
        needs_rebuild = system.tile_list.check_rebuild(positions_soa)
        is_rebuild = needs_rebuild

        if needs_rebuild:
            rebuild_count += 1
            pdb_to_sorted_gpu, pdb_to_sorted_np = system.tile_list.rebuild(
                positions_soa, system.topology, system.pbc_matrix, system.pbc_inv
            )
            sorted_to_pdb = cp.asnumpy(system.tile_list.d_sorted_to_pdb)
            system._permute_all_arrays(pdb_to_sorted_gpu, pdb_to_sorted_np)
            system.tile_list.build_tiles(system.topology, system.pbc_matrix)
            for term in system.force_terms:
                if hasattr(term, 'bind_sorted'):
                    s2p = system._sorted_to_pdb_np()
                    term.bind_sorted(
                        system.topology, system.tile_list, system.gpu,
                        sorted_particle_types=system._particle_types_pdb[s2p],
                    )

            # Verify masses and charges after rebuild
            sorted_masses = cp.asnumpy(system.gpu.d_masses)
            expected_masses = pdb_masses[sorted_to_pdb]
            mass_err = np.max(np.abs(sorted_masses - expected_masses))

            # Check nonbonded charges
            if hasattr(nb, '_parameter_arrays') and 'charge' in nb._parameter_arrays:
                sorted_charges = nb._parameter_arrays['charge']
                expected_charges = pdb_charges[sorted_to_pdb]
                charge_err = np.max(np.abs(sorted_charges - expected_charges))
            else:
                charge_err = -1.0
        else:
            mass_err = 0.0
            charge_err = 0.0

        # Compute forces and step
        system.compute_forces()
        energies = system.dump_energy()
        e_bonded = energies.get('bonded', 0.0) * KCAL
        e_nonbond = energies.get('nonbonded', 0.0) * KCAL
        e_total = sum(energies.values()) * KCAL

        integrator.step(system.gpu)

        rb_str = f"#{rebuild_count}" if is_rebuild else ""
        print(f"{step:5d} {rb_str:>7s} {e_bonded:16.1f} {e_nonbond:16.1f} {e_total:16.1f} {mass_err:10.2e} {charge_err:10.2e}")

        # Early exit on NaN/Inf
        if not np.isfinite(e_total) or abs(e_total) > 1e8:
            print(f"\n*** ENERGY EXPLODED at step {step}! ***")
            break

    # Deep dive: check bonded indices after last rebuild
    print("\n=== BONDED INDEX CHECK ===")
    topo = system.topology
    s2p = system._sorted_to_pdb_np()
    # Check if bonded indices are valid (< N)
    for name in ['bond_indices', 'angle_indices', 'dihedral_indices', 'improper_indices']:
        idx = getattr(topo, name, None)
        if idx is not None and len(idx) > 0:
            max_idx = np.max(idx)
            min_idx = np.min(idx)
            invalid = np.sum((idx < 0) | (idx >= N))
            print(f"  {name}: shape={idx.shape}, min={min_idx}, max={max_idx}, invalid={invalid}/{idx.size}")

    # Check if forces are reasonable
    print("\n=== FORCE CHECK ===")
    fx = cp.asnumpy(system.gpu.d_forces_x)
    fy = cp.asnumpy(system.gpu.d_forces_y)
    fz = cp.asnumpy(system.gpu.d_forces_z)
    fmag = np.sqrt(fx**2 + fy**2 + fz**2)
    print(f"  Force magnitude: max={np.max(fmag):.4f}, mean={np.mean(fmag):.4f}, median={np.median(fmag):.4f}")
    huge = np.sum(fmag > 1e4)
    print(f"  Atoms with |F| > 1e4: {huge}")

    # Check positions
    print("\n=== POSITION CHECK ===")
    px = cp.asnumpy(system.gpu.d_positions_x)
    py = cp.asnumpy(system.gpu.d_positions_y)
    pz = cp.asnumpy(system.gpu.d_positions_z)
    print(f"  X: min={np.min(px):.2f}, max={np.max(px):.2f}")
    print(f"  Y: min={np.min(py):.2f}, max={np.max(py):.2f}")
    print(f"  Z: min={np.min(pz):.2f}, max={np.max(pz):.2f}")
    outside = np.sum((px < 0) | (px > BOX) | (py < 0) | (py > BOX) | (pz < 0) | (pz > BOX))
    print(f"  Atoms outside box: {outside}")


if __name__ == '__main__':
    main()
