"""OpenMM CUDA nonbonded benchmark for 1M9Z.

Uses CUDA platform with PME + 12A cutoff (production config).
Measures pure nonbonded kernel time via nsys profiling.

Usage:
    # Benchmark
    CUDA_VISIBLE_DEVICES=0 conda run -n md_analysis python benchmark/benchmark_openmm_nb.py

    # nsys timeline
    CUDA_VISIBLE_DEVICES=0 /home/ubuntu/Programs/cuda/13.0/bin/nsys profile \
      --trace=cuda,nvtx,osrt --output=benchmark/openmm_nb_timeline \
      conda run -n md_analysis python benchmark/benchmark_openmm_nb.py

    # ncu deep dive on nonbonded kernel
    CUDA_VISIBLE_DEVICES=0 /home/ubuntu/Programs/cuda/13.0/bin/ncu \
      --set full --launch-skip 50 --launch-count 10 \
      -k "regex:computeNonbonded" \
      -o benchmark/openmm_nb_ncu \
      conda run -n md_analysis python benchmark/benchmark_openmm_nb.py
"""
import os
import time
import openmm as mm
import openmm.app as app
from openmm import unit
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'mdpy', 'test', 'data')
PSF_PATH = os.path.join(DATA_DIR, '1M9Z.psf')
PDB_PATH = os.path.join(DATA_DIR, '1M9Z_minimized.pdb')
PRM_PATH = os.path.join(DATA_DIR, 'par_all36_prot.prm')
STR_PATH = os.path.join(DATA_DIR, 'toppar_water_ions.str')

BOX_SIZE = 10.8  # nm
CUTOFF = 1.2     # nm
DT = 0.002       # ps
WARMUP = 500
BENCH_STEPS = 500
NUM_BLOCKS = 5


def main():
    print("=== OpenMM CUDA Nonbonded Benchmark (1M9Z, 95567 atoms) ===")
    print(f"Box: {BOX_SIZE} nm, Cutoff: {CUTOFF} nm, PME, dt: {DT} ps")
    print()

    psf = app.CharmmPsfFile(PSF_PATH)
    pdb = app.PDBFile(PDB_PATH)
    params = app.CharmmParameterSet(PRM_PATH, STR_PATH)
    psf.setBox(BOX_SIZE, BOX_SIZE, BOX_SIZE)

    system = psf.createSystem(params,
                               nonbondedMethod=app.PME,
                               nonbondedCutoff=CUTOFF * unit.nanometer,
                               constraints=app.HBonds)

    integrator = mm.VerletIntegrator(DT * unit.picoseconds)
    platform = mm.Platform.getPlatformByName('CUDA')
    properties = {'Precision': 'single', 'DeviceIndex': '0'}

    sim = app.Simulation(psf.topology, system, integrator, platform, properties)
    sim.context.setPositions(pdb.getPositions())
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin)

    print(f"Atoms: {system.getNumParticles()}")
    print(f"Forces: {[system.getForce(i).__class__.__name__ for i in range(system.getNumForces())]}")
    print()

    print(f"Warmup ({WARMUP} steps)...")
    t0 = time.perf_counter()
    sim.step(WARMUP)
    import cupy as cp
    cp.cuda.Stream.null.synchronize()
    t_warm = time.perf_counter() - t0
    print(f"  {t_warm:.1f}s ({t_warm/WARMUP*1000:.2f} ms/step)")

    print(f"\nBenchmark: {NUM_BLOCKS} x {BENCH_STEPS} steps")
    print(f"  {'Block':>6s}  {'ms/step':>10s}  {'ns/day':>10s}")
    print(f"  {'------':>6s}  {'----------':>10s}  {'----------':>10s}")

    block_times = []
    for i in range(NUM_BLOCKS):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        sim.step(BENCH_STEPS)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - t0

        ms = elapsed / BENCH_STEPS * 1000
        ns_day = 86400.0 / (elapsed / BENCH_STEPS) * DT * 1e-6
        block_times.append(ms)
        print(f"  {i+1:6d}  {ms:10.3f}  {ns_day:10.1f}")

    avg = np.mean(block_times)
    ns_avg = 86400.0 / (avg * 1e-3) * DT * 1e-6
    print(f"\n  avg  {avg:.3f} ms/step = {ns_avg:.1f} ns/day")


if __name__ == '__main__':
    main()
