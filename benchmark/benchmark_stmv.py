"""mdpy STMV (1,066,628 atoms) PME performance benchmark.

STMV (Satellite Tobacco Mosaic Virus) is the standard million-atom MD benchmark.
System: CHARMM22/27 force field, protein capsid + ssRNA + water/ions, cubic box.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n md_analysis python benchmark/benchmark_stmv.py
"""

import os, sys, time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cupy as cp
import numpy as np
from _data_path import DATA_DIR

from mdpy.io.psf_parser import PSFParser
from mdpy.io.pdb_parser import PDBParser
from mdpy.io.charmm_toppar_parser import CharmmTopparParser
from mdpy.force.factories.charmm import create_charmm_forces
from mdpy.integrator.langevin import LangevinBAOABIntegrator
from mdpy.core.state import State
from mdpy.system import System
from mdpy.utils import generate_velocity_from_temperature

STMV_DIR = os.path.join(os.path.dirname(__file__), "data", "stmv")

BOX_SIZE = 216.832
CUTOFF = 12.0
TIME_STEP_FS = 2
NUM_BLOCKS = 5
BLOCK_STEPS = 500
WARMUP_STEPS = 20

psf = PSFParser(os.path.join(STMV_DIR, "stmv.psf"))
pdb = PDBParser(os.path.join(STMV_DIR, "stmv_minimized.pdb"))
toppar = CharmmTopparParser(
    os.path.join(STMV_DIR, "par_all27_prot_na.prm"),
    os.path.join(STMV_DIR, "toppar_water_ions.str"),
)
topology = psf.topology
parameter_set = toppar.resolve_parameter_set(topology, psf.particle_type_names)
pbc_matrix = np.eye(3, dtype=np.float64) * BOX_SIZE

forces = create_charmm_forces(
    topology, parameter_set, pbc_matrix, cutoff=CUTOFF)

state = State(topology.num_particles)
state.set_masses(psf.particle_masses)
state.set_charges(psf.particle_charges)
state.set_type_indices(parameter_set.particle_type_indices)
system = System(topology, state)
system.set_pbc(pbc_matrix)
system.add_force_term(forces["bonded"], stream='pme')
system.add_force_term(forces["nonbonded"])
system.add_force_term(forces["pme"], stream='pme')

positions = pdb.positions
velocities = generate_velocity_from_temperature(10.0, psf.particle_masses, seed=42)
system.set_positions(positions)
system.set_velocities(velocities)

integrator = LangevinBAOABIntegrator(TIME_STEP_FS, 300.0, 1.0)


def _run_steps(n):
    for i in range(n):
        system.update_neighbor_list(sync_interval=10)
        system.compute_forces()
        integrator.step(system)


print("mdpy STMV PME benchmark")
print(f"  Atoms:      {topology.num_particles}")
print(f"  Box:        {BOX_SIZE} A")
print(f"  Cutoff:     {CUTOFF} A")
print(f"  time_step: {TIME_STEP_FS} fs")
print(f"  Integrator: Langevin BAOAB")
print(f"  PME alpha:  {forces['pme'].alpha:.4f}")
print(
    f"  PME grid:   {forces['pme'].grid_x} x {forces['pme'].grid_y} x {forces['pme'].grid_z}"
)
print()

print(f"Warmup ({WARMUP_STEPS} steps)...")
cp.cuda.Stream.null.synchronize()
t0 = time.perf_counter()
_run_steps(WARMUP_STEPS)
cp.cuda.Stream.null.synchronize()
print(f"  {time.perf_counter()-t0:.1f}s")

print(f"\nBenchmark: {NUM_BLOCKS} x {BLOCK_STEPS} steps")
print(f"  {'Block':>6s}  {'ms/step':>10s}  {'ns/day':>10s}  {'E_pot (kcal/mol)':>18s}")
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
    KCAL_PER_INTERNAL = 1.0 / 4.1840286576e-4
    ms = elapsed / BLOCK_STEPS * 1000
    ns = 86400.0 / (elapsed / BLOCK_STEPS) * TIME_STEP_FS * 1e-6
    block_times.append(ms)
    print(f"  {i+1:6d}  {ms:10.3f}  {ns:10.1f}  {e_total*KCAL_PER_INTERNAL:18.1f}")

avg = np.mean(block_times)
med = np.median(block_times)
ns_avg = 86400.0 / (avg * 1e-3) * TIME_STEP_FS * 1e-6
ns_med = 86400.0 / (med * 1e-3) * TIME_STEP_FS * 1e-6
print(f"\n  avg  {avg:.3f} ms/step = {ns_avg:.1f} ns/day")
print(f"  med  {med:.3f} ms/step = {ns_med:.1f} ns/day")
