"""Multi-step brute-force comparison script for mdpy dynamics.

Runs N steps of mdpy dynamics and compares GPU forces against O(N^2) brute-force
at each step. Identifies exactly WHEN forces start to diverge.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n md_analysis python mdpy/test/multistep_debug.py

Default: 3 steps on 1M9Z (95,567 atoms). Each step takes ~10-20 minutes due to
O(N^2) nonbonded brute-force computation.
"""

import os
import time
import numpy as np
import cupy as cp
from types import SimpleNamespace

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

PSF_PATH = os.path.join(DATA_DIR, '1M9Z.psf')
PDB_PATH = os.path.join(DATA_DIR, '1M9Z_minimized.pdb')
PRM_PATH = os.path.join(DATA_DIR, 'par_all36_prot.prm')
STR_PATH = os.path.join(DATA_DIR, 'toppar_water_ions.str')

BOX_SIZE = 108.0
CUTOFF = 12.0
NUM_STEPS = 3
ENERGY_CONVERSION = 1.0 / 4.1840286576e-4


def main():
    from mdpy.test.generate_bruteforce_reference import (
        compute_neighbor_pairs,
        compute_exclusion_scaling_flags,
        compute_bond_forces,
        compute_angle_forces,
        compute_dihedral_forces,
        compute_improper_forces,
        compute_nonbonded_forces,
    )
    from mdpy.io.psf_parser import PSFParser
    from mdpy.io.pdb_parser import PDBParser
    from mdpy.io.charmm_toppar_parser import CharmmTopparParser
    from mdpy.io.charmm_toppar_parser import create_parameter_table
    from mdpy.force.bonded_force import BondedForce
    from mdpy.force.factories.charmm import create_bonded_group
    from mdpy.force.nonbonded_force import NonbondedForce
    from mdpy.force.expressions.lennard_jones import lennard_jones
    from mdpy.force.expressions.coulomb import coulomb
    from mdpy.system import System
    from mdpy.integrator.verlet import VerletIntegrator

    print("=" * 80)
    print("mdpy multi-step brute-force comparison")
    print("=" * 80)

    psf = PSFParser(PSF_PATH)
    pdb = PDBParser(PDB_PATH)
    toppar = CharmmTopparParser(PRM_PATH, STR_PATH)
    topology = psf.topology
    parameter_table = create_parameter_table(topology, toppar)
    N = topology.num_particles

    # Save brute-force data BEFORE system mutates topology indices
    bf_bond_indices = topology.bond_indices.copy()
    bf_angle_indices = topology.angle_indices.copy()
    bf_dihedral_indices = topology.dihedral_indices.copy()
    bf_improper_indices = topology.improper_indices.copy()
    bf_bond_params = parameter_table.get_term_parameter('bond').copy()
    bf_angle_params = parameter_table.get_term_parameter('angle').copy()
    bf_dihedral_params = parameter_table.get_term_parameter('dihedral').copy()
    bf_improper_params = parameter_table.get_term_parameter('improper').copy()
    bf_charges = topology.charges.astype(np.float64).copy()
    bf_charges_14 = bf_charges.copy()
    bf_lj_pair = parameter_table.type_pair_parameters['lj_pair'].astype(
        np.float64,
    ).copy()
    bf_lj_pair_14 = parameter_table.type_pair_parameters.get(
        'lj_pair_14', bf_lj_pair,
    ).astype(np.float64).copy()
    bf_n_types = int(np.sqrt(len(bf_lj_pair) // 2))
    bf_particle_type_indices = topology.particle_type_indices.copy()

    _offset, _neighbors = topology.exclusion_csr
    bf_topology = SimpleNamespace(
        num_particles=N,
        exclusion_offset=cp.asnumpy(_offset),
        exclusion_neighbors=cp.asnumpy(_neighbors),
    )

    print(f"System: {N} atoms, box={BOX_SIZE} A, cutoff={CUTOFF} A")
    print(f"Steps:  {NUM_STEPS}, time_step=0.5 fs")
    print()

    # Setup mdpy system
    pbc_matrix = np.eye(3, dtype=np.float64) * BOX_SIZE
    pbc_inv = np.linalg.inv(pbc_matrix)

    system = System(topology)

    system.set_pbc(pbc_matrix)
    system.add_force_term(create_bonded_group(topology, parameter_table))
    nb = NonbondedForce(lennard_jones + coulomb, cutoff=CUTOFF)
    lj_pair = parameter_table.type_pair_parameters['lj_pair']
    nb.set_pair_parameter('sigma', lj_pair[0::2].astype(np.float32))
    nb.set_pair_parameter('epsilon', lj_pair[1::2].astype(np.float32))
    system.add_force_term(nb)

    raw = pdb.positions.astype(np.float64)
    frac = raw @ pbc_inv
    frac -= np.floor(frac)
    wrapped = frac @ pbc_matrix
    system.set_positions(wrapped.astype(np.float32))
    system.set_velocities(np.zeros((N, 3), dtype=np.float32))

    system.update_neighbor_list(force_rebuild=True)

    integrator = VerletIntegrator(0.5)

    header = (
        f"{'Step':>4s}  "
        f"{'GPU E (kcal/mol)':>16s}  "
        f"{'BF E (kcal/mol)':>16s}  "
        f"{'Rel Err':>12s}  "
        f"{'F Corr':>8s}  "
        f"{'Max |dF|':>12s}  "
        f"{'GPU |SumF|':>12s}  "
        f"{'BF |SumF|':>12s}"
    )
    print(header)
    print("-" * len(header))

    for step in range(NUM_STEPS):
        t_step = time.time()
        print(f"\n--- Step {step} ---")

        system.update_neighbor_list(sync_interval=10)

        positions, velocities = system.dump_state()
        positions = positions.astype(np.float64)

        energies = system.dump_energy()
        gpu_forces = system.dump_forces().astype(np.float64)

        gpu_energy_internal = sum(energies.values())
        gpu_energy_kcal = gpu_energy_internal * ENERGY_CONVERSION

        print("  [brute-force] Computing neighbor pairs...")
        pairs_i, pairs_j, _ = compute_neighbor_pairs(positions, BOX_SIZE, CUTOFF)
        flags = compute_exclusion_scaling_flags(bf_topology, pairs_i, pairs_j)

        print("  [brute-force] Computing bonded forces...")
        t_bonded = time.time()
        bond_f, bond_e = compute_bond_forces(
            positions, bf_bond_indices, bf_bond_params, BOX_SIZE,
        )
        angle_f, angle_e = compute_angle_forces(
            positions, bf_angle_indices, bf_angle_params, BOX_SIZE,
        )
        dih_f, dih_e = compute_dihedral_forces(
            positions, bf_dihedral_indices, bf_dihedral_params, BOX_SIZE,
        )
        imp_f, imp_e = compute_improper_forces(
            positions, bf_improper_indices, bf_improper_params, BOX_SIZE,
        )
        bonded_forces = bond_f + angle_f + dih_f + imp_f
        bonded_energy = bond_e + angle_e + dih_e + imp_e
        print(
            f"  [brute-force] Bonded done in {time.time() - t_bonded:.1f}s, "
            f"E={bonded_energy * ENERGY_CONVERSION:.2f} kcal/mol"
        )

        nonbonded_forces, nonbonded_energy = compute_nonbonded_forces(
            positions, pairs_i, pairs_j, flags,
            bf_charges, bf_charges_14, bf_lj_pair, bf_lj_pair_14,
            bf_particle_type_indices, BOX_SIZE, bf_n_types,
        )

        bf_forces = bonded_forces + nonbonded_forces
        bf_energy = bonded_energy + nonbonded_energy
        bf_energy_kcal = bf_energy * ENERGY_CONVERSION

        denom = max(abs(gpu_energy_kcal), abs(bf_energy_kcal))
        if denom > 1e-14:
            rel_err = abs(gpu_energy_kcal - bf_energy_kcal) / denom
        else:
            rel_err = 0.0

        mask = np.any(np.abs(bf_forces) > 1e-8, axis=1)
        if np.any(mask):
            corr = np.corrcoef(
                gpu_forces[mask].flatten(), bf_forces[mask].flatten(),
            )[0, 1]
        else:
            corr = 1.0

        force_diff = gpu_forces - bf_forces
        max_force_err = np.max(np.linalg.norm(force_diff, axis=1))

        gpu_sum_f = np.linalg.norm(np.sum(gpu_forces, axis=0))
        bf_sum_f = np.linalg.norm(np.sum(bf_forces, axis=0))

        print(
            f"  {step:4d}  "
            f"{gpu_energy_kcal:16.4f}  "
            f"{bf_energy_kcal:16.4f}  "
            f"{rel_err:12.6e}  "
            f"{corr:8.6f}  "
            f"{max_force_err:12.4e}  "
            f"{gpu_sum_f:12.4e}  "
            f"{bf_sum_f:12.4e}"
        )

        print(f"  GPU energy breakdown (kcal/mol): ", end="")
        print(
            "  ".join(
                f"{k}={v * ENERGY_CONVERSION:.2f}" for k, v in energies.items()
            )
        )

        integrator.step(system)
        elapsed = time.time() - t_step
        print(f"  Step wall time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    print("\nDone.")


if __name__ == '__main__':
    main()
