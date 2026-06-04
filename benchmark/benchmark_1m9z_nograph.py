"""mdpy 1M9Z benchmark WITHOUT CUDA Graph — for nsys kernel-level profiling.

Forces the non-graph path so every kernel appears individually in nsys timeline.
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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
    from mdpy.force.expressions.coulomb import coulomb
    from mdpy.integrator.verlet import VerletIntegrator
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
    nb = NonbondedForce(lennard_jones + coulomb)
    nb.bind(topology, parameter_table, CUTOFF)
    system.add_force_term(nb)

    raw = pdb.positions.astype(np.float64)
    frac = raw @ pbc_inv
    frac -= np.floor(frac)
    wrapped = frac @ pbc_matrix
    system.particles.positions[:] = wrapped
    system.particles.velocities[:] = 0.0
    system.gpu.upload_positions(system.particles)
    system.gpu.upload_velocities(system.particles)

    integrator = VerletIntegrator(DT_FS)

    print("mdpy 1M9Z benchmark (NO CUDA Graph)")
    print(f"  Atoms:      {topology.num_particles}")
    print(f"  Box:        {BOX_SIZE} A")
    print(f"  Cutoff:     {CUTOFF} A")
    print(f"  dt:         {DT_FS} fs")
    print(f"  Integrator: Verlet (no constraints)")
    print()

    print(f"Warmup ({WARMUP_STEPS} steps)...")
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    system.step(integrator, WARMUP_STEPS)
    cp.cuda.Stream.null.synchronize()
    t_warm = time.perf_counter() - t0
    print(f"  {t_warm:.1f}s ({t_warm / WARMUP_STEPS * 1000:.2f} ms/step)")

    # Bypass step() entirely — manually implement the step loop on the
    # default stream with correct synchronisation so rebuilds work properly
    # and no CUDA graph is ever captured.
    steps_since_check = 0
    rebuild_interval = system.block_list.rebuild_check_interval

    def _run_steps(n_steps):
        nonlocal steps_since_check
        for _ in range(n_steps):
            system._emit_step_kernels(integrator)
            steps_since_check += 1
            if steps_since_check >= rebuild_interval:
                cp.cuda.Stream.null.synchronize()
                if int(system.block_list.d_rebuild_flag[0]) == 1:
                    positions_soa = (
                        system.gpu.d_wrapped_positions_x,
                        system.gpu.d_wrapped_positions_y,
                        system.gpu.d_wrapped_positions_z,
                    )
                    system._do_full_rebuild(positions_soa)
                steps_since_check = 0

    print(f"\nBenchmark: {NUM_BLOCKS} x {BLOCK_STEPS} steps (no graph)")
    print(
        f"  {'Block':>6s}  {'ms/step':>10s}  {'ns/day':>10s}  {'E_pot (kcal/mol)':>18s}"
    )
    print(
        f"  {'------':>6s}  {'----------':>10s}  {'----------':>10s}  {'------------------':>18s}"
    )

    block_times = []
    for i in range(NUM_BLOCKS):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        _run_steps(BLOCK_STEPS)
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
        print(f"  {name:>12s}: {val * KCAL_PER_INTERNAL:18.1f}")

    print()
    print("Note: dt=0.5fs required because mdpy has no bond constraints.")
    print("      OpenMM uses HBond constraints allowing dt=2fs.")
    print("      CUDA Graph DISABLED — all kernels emit individually.")


if __name__ == "__main__":
    main()
