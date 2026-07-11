"""OpenMM 1M9Z NPT benchmark — Monte Carlo barostat comparison with mdpy.

Mirror of benchmark/benchmark_1m9z_npt_mdpy.py for apples-to-apples NPT
comparison. Uses the same PSF/PDB/force-field files, box size, cutoff, and
MC barostat parameters. Bond and angle forces use periodic boundary conditions.

System: 1M9Z (protein in water), CHARMM36 force field, cubic box.
Ensemble: NPT, 1 bar, 300 K, Monte Carlo barostat (frequency=25).

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n md_analysis python benchmark/benchmark_1m9z_npt_openmm.py
"""

import os, sys, time
from collections import defaultdict, deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit


def make_molecules_whole(positions, bonds, box):
    """Unwrap molecules split across PBC boundaries.

    BFS over the bond graph; for each bond (i, j) shift j into the same
    periodic image as i via minimum-image convention.
    """
    positions = np.array(positions, dtype=np.float64).copy()
    adj = defaultdict(list)
    for a, b in bonds:
        adj[a].append(b)
        adj[b].append(a)
    visited = set()
    for start in range(len(positions)):
        if start in visited:
            continue
        visited.add(start)
        queue = deque([start])
        while queue:
            i = queue.popleft()
            for j in adj[i]:
                if j not in visited:
                    visited.add(j)
                    positions[j] -= np.round((positions[j] - positions[i]) / box) * box
                    queue.append(j)
    return positions


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

CUTOFF_A = 12.0
TIME_STEP_FS = 2
PRESSURE_BAR = 1.0
TEMPERATURE = 300.0
MC_FREQUENCY = 25
NVT_STEPS = 100
NPT_STEPS = 2000
REPORT_INTERVAL = 50

psf = app.CharmmPsfFile(os.path.join(DATA_DIR, "1M9Z.psf"))
pdb = app.PDBFile(os.path.join(DATA_DIR, "1M9Z_minimized.pdb"))
params = app.CharmmParameterSet(
    os.path.join(DATA_DIR, "par_all36_prot.prm"),
    os.path.join(DATA_DIR, "toppar_water_ions.str"),
)

with open(os.path.join(DATA_DIR, "1M9Z_minimized.pdb"), 'r') as _f:
    _cryst1 = _f.readline()
_box_a = float(_cryst1[6:15])
_box_b = float(_cryst1[15:24])
_box_c = float(_cryst1[24:33])
psf.setBox(_box_a / 10, _box_b / 10, _box_c / 10)

system = psf.createSystem(
    params,
    nonbondedMethod=app.PME,
    nonbondedCutoff=CUTOFF_A * unit.angstrom,
    constraints=app.HBonds,
    rigidWater=True,
    ewaldErrorTolerance=1e-4,
)

for force in system.getForces():
    if isinstance(force, (mm.HarmonicBondForce, mm.HarmonicAngleForce)):
        force.setUsesPeriodicBoundaryConditions(True)

platform = mm.Platform.getPlatformByName("CUDA")
properties = {"Precision": "single"}

total_mass = sum(
    system.getParticleMass(i).value_in_unit(unit.dalton)
    for i in range(system.getNumParticles())
)
initial_volume = _box_a * _box_b * _box_c
initial_density = total_mass * 1.66054 / initial_volume

bond_list = [(b[0].index, b[1].index) for b in psf.topology.bonds()]
pos = pdb.positions.value_in_unit(unit.angstrom)
pos = make_molecules_whole(pos, bond_list, _box_a)

# ---- NVT equilibration (no barostat) ----
integrator_nvt = mm.LangevinIntegrator(
    TEMPERATURE * unit.kelvin, 1 / unit.picosecond, TIME_STEP_FS * unit.femtoseconds
)
integrator_nvt.setRandomNumberSeed(42)
simulation_nvt = app.Simulation(psf.topology, system, integrator_nvt, platform, properties)
simulation_nvt.context.setPositions(pos * unit.angstrom)
simulation_nvt.context.setVelocitiesToTemperature(TEMPERATURE * unit.kelvin, 42)

nb = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)][0]
alpha, grid_x, grid_y, grid_z = nb.getPMEParametersInContext(simulation_nvt.context)

print("OpenMM 1M9Z NPT benchmark (Monte Carlo barostat)")
print(f"  Atoms:          {system.getNumParticles()}")
print(f"  Box:            {_box_a:.1f} x {_box_b:.1f} x {_box_c:.1f} A")
print(f"  Volume:         {initial_volume:.0f} A^3")
print(f"  Density:        {initial_density:.4f} g/cm^3")
print(f"  PME grid:       {grid_x} x {grid_y} x {grid_z}  (alpha={alpha:.4f})")
print(f"  Target:         {PRESSURE_BAR} bar, {TEMPERATURE} K")
print(f"  MC frequency:   {MC_FREQUENCY}")
print()

print(f"NVT equilibration ({NVT_STEPS} steps)...")
t0 = time.perf_counter()
simulation_nvt.step(NVT_STEPS)
nvt_elapsed = time.perf_counter() - t0
print(f"  done in {nvt_elapsed:.1f}s ({nvt_elapsed/NVT_STEPS*1000:.1f} ms/step)")

# ---- Transfer state to NPT simulation ----
state_nvt = simulation_nvt.context.getState(getPositions=True, getVelocities=True)
positions_nvt = state_nvt.getPositions()
velocities_nvt = state_nvt.getVelocities()

# ---- Add barostat for NPT ----
barostat = mm.MonteCarloBarostat(
    PRESSURE_BAR * unit.bar, TEMPERATURE * unit.kelvin, MC_FREQUENCY
)
barostat.setRandomNumberSeed(42)
system.addForce(barostat)

integrator_npt = mm.LangevinIntegrator(
    TEMPERATURE * unit.kelvin, 1 / unit.picosecond, TIME_STEP_FS * unit.femtoseconds
)
integrator_npt.setRandomNumberSeed(42)
simulation_npt = app.Simulation(psf.topology, system, integrator_npt, platform, properties)
simulation_npt.context.setPositions(positions_nvt)
simulation_npt.context.setVelocities(velocities_nvt)

# ---- NPT production ----
print(f"\nNPT simulation ({NPT_STEPS} steps):")
print(f"  {'Step':>6s}  {'Volume(A^3)':>12s}  {'Box_len':>8s}  {'Density':>8s}  {'E_pot(kcal/mol)':>18s}")
print(f"  {'------':>6s}  {'------------':>12s}  {'--------':>8s}  {'--------':>8s}  {'------------------':>18s}")

block_times = []
for i in range(0, NPT_STEPS, REPORT_INTERVAL):
    t0 = time.perf_counter()
    simulation_npt.step(REPORT_INTERVAL)
    elapsed = time.perf_counter() - t0

    state = simulation_npt.context.getState(getEnergy=True)
    box_vectors = state.getPeriodicBoxVectors()
    box_a = box_vectors[0][0].value_in_unit(unit.angstrom)
    box_b = box_vectors[1][1].value_in_unit(unit.angstrom)
    box_c = box_vectors[2][2].value_in_unit(unit.angstrom)
    vol = box_a * box_b * box_c
    density = total_mass * 1.66054 / vol
    e_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    e_kcal = e_kj / 4.184
    ms = elapsed / REPORT_INTERVAL * 1000
    block_times.append(ms)
    print(f"  {i+REPORT_INTERVAL:6d}  {vol:12.0f}  {box_a:8.2f}  {density:8.4f}  {e_kcal:18.1f}")

avg_ms = np.mean(block_times)
state_final = simulation_npt.context.getState(getEnergy=True)
box_vectors_final = state_final.getPeriodicBoxVectors()
final_box_a = box_vectors_final[0][0].value_in_unit(unit.angstrom)
final_box_b = box_vectors_final[1][1].value_in_unit(unit.angstrom)
final_box_c = box_vectors_final[2][2].value_in_unit(unit.angstrom)
final_vol = final_box_a * final_box_b * final_box_c
final_density = total_mass * 1.66054 / final_vol
vol_change = (final_vol - initial_volume) / initial_volume * 100

print(f"\nSummary:")
print(f"  Initial density:  {initial_density:.4f} g/cm^3")
print(f"  Final density:    {final_density:.4f} g/cm^3")
print(f"  Volume change:    {vol_change:+.1f}%")
print(f"  NPT ms/step:      {avg_ms:.1f} (vs {nvt_elapsed/NVT_STEPS*1000:.1f} NVT)")
