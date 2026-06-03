"""mdpy 1M9Z (95,567 atoms) PME performance benchmark.

Usage:
    conda run -n md_analysis python benchmark/benchmark_1m9z_pme.py

Uses Verlet integrator at dt=0.5fs with PME electrostatics.
Reports ms/step and ns/day with per-term energy breakdown.
"""

import os
import time

import cupy as cp
import numpy as np

from benchmark._data_path import DATA_DIR

PSF_PATH = os.path.join(DATA_DIR, "1M9Z.psf")
PDB_PATH = os.path.join(DATA_DIR, "1M9Z_minimized.pdb")
PRM_PATH = os.path.join(DATA_DIR, "par_all36_prot.prm")
STR_PATH = os.path.join(DATA_DIR, "toppar_water_ions.str")

BOX_SIZE = 108.0
CUTOFF = 12.0
DT_FS = 0.5
KCAL_PER_INTERNAL = 1.0 / 4.1840286576e-4
WARMUP_STEPS = 50
BLOCK_STEPS = 2500
NUM_BLOCKS = 5


def main():
    from mdpy.io.psf_parser import PSFParser
    from mdpy.io.pdb_parser import PDBParser
    from mdpy.io.charmm_toppar_parser import CharmmTopparParser
    from mdpy.io.charmm_toppar_parser import create_parameter_table
    from mdpy.force.bonded_force import BondedForce
    from mdpy.force.nonbonded_force import NonbondedForce
    from mdpy.force.expressions.lennard_jones import lennard_jones
    from mdpy.force.expressions.screened_coulomb import screened_coulomb
    from mdpy.force.pme_parameters import PMEParameters
    from mdpy.force.pme_reciprocal_force import PMEReciprocalForce
    from mdpy.integrator.verlet import VerletIntegrator
    from mdpy.system import System

    psf = PSFParser(PSF_PATH)
    pdb = PDBParser(PDB_PATH)
    toppar = CharmmTopparParser(PRM_PATH, STR_PATH)
    topology = psf.topology
    parameter_table = create_parameter_table(topology, toppar)

    pbc_matrix = np.eye(3, dtype=np.float64) * BOX_SIZE
    pbc_inv = np.linalg.inv(pbc_matrix)

    pme_params = PMEParameters.from_box(BOX_SIZE, BOX_SIZE, BOX_SIZE, cutoff=CUTOFF)

    system = System(topology, pbc_matrix, cutoff=CUTOFF)
    system.add_force_term(BondedForce.charmm(topology, parameter_table))

    nb = NonbondedForce(lennard_jones + screened_coulomb)
    nb.bind(topology, parameter_table, CUTOFF, alpha=pme_params.alpha)
    system.add_force_term(nb)

    pme = PMEReciprocalForce(pme_params, CUTOFF)
    pme.bind(topology, parameter_table, pbc_matrix=pbc_matrix)
    system.add_force_term(pme)

    raw = pdb.positions.astype(np.float64)
    frac = raw @ pbc_inv
    frac -= np.floor(frac)
    wrapped = frac @ pbc_matrix
    system.particles.positions[:] = wrapped
    system.particles.velocities[:] = 0.0
    system.gpu.upload_positions(system.particles)
    system.gpu.upload_velocities(system.particles)

    integrator = VerletIntegrator(DT_FS)

    print("mdpy 1M9Z PME benchmark")
    print(f"  Atoms:      {topology.num_particles}")
    print(f"  Box:        {BOX_SIZE} A")
    print(f"  Cutoff:     {CUTOFF} A")
    print(f"  dt:         {DT_FS} fs")
    print(f"  Integrator: Verlet (no constraints)")
    print(f"  PME alpha:  {pme_params.alpha:.4f}")
    print(f"  PME grid:   {pme_params.grid_x} x {pme_params.grid_y} x {pme_params.grid_z}")
    print(f"  PME order:  {pme_params.order}")
    print()

    print(f"Warmup ({WARMUP_STEPS} steps)...")
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    system.step(integrator, WARMUP_STEPS)
    cp.cuda.Stream.null.synchronize()
    t_warm = time.perf_counter() - t0
    print(f"  {t_warm:.1f}s ({t_warm / WARMUP_STEPS * 1000:.2f} ms/step)")

    print(f"\nBenchmark: {NUM_BLOCKS} x {BLOCK_STEPS} steps")
    print(
        f"  {'Block':>6s}  {'ms/step':>10s}  {'ns/day':>10s}  {'E_pot (kcal/mol)':>18s}"
    )
    print(
        f"  {'------':>6s}  {'----------':>10s}  {'----------':>10s}  {'------------------':>18s}"
    )

    block_times = []
    energy_dict = {}
    for i in range(NUM_BLOCKS):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        system.step(integrator, BLOCK_STEPS)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - t0

        energy_dict = system.dump_energy()
        e_total = sum(energy_dict.values())
        e_kcal = e_total * KCAL_PER_INTERNAL
        ms_per_step = elapsed / BLOCK_STEPS * 1000
        ns_day = 86400.0 / (elapsed / BLOCK_STEPS) * DT_FS * 1e-6
        block_times.append(ms_per_step)
        print(f"  {i + 1:6d}  {ms_per_step:10.3f}  {ns_day:10.1f}  {e_kcal:18.1f}")

    avg = np.mean(block_times)
    med = np.median(block_times)
    ns_avg = 86400.0 / (avg * 1e-3) * DT_FS * 1e-6
    ns_med = 86400.0 / (med * 1e-3) * DT_FS * 1e-6

    print()
    print(f"  avg  {avg:.3f} ms/step = {ns_avg:.1f} ns/day")
    print(f"  med  {med:.3f} ms/step = {ns_med:.1f} ns/day")
    print()
    print("Per-term energies (last block, kcal/mol):")
    for name, val in energy_dict.items():
        print(f"  {name:>16s}: {val * KCAL_PER_INTERNAL:18.1f}")

    print()
    print("Note: dt=0.5fs required because mdpy has no bond constraints.")
    print("      Electrostatics: PME (screened Coulomb direct + reciprocal grid)")


if __name__ == "__main__":
    main()
