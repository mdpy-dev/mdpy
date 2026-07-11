"""mdpy 1M9Z NPT benchmark — Monte Carlo barostat vs OpenMM comparison.

Runs NVT equilibration then NPT with MC barostat at 1 bar, 300K.
Reports volume, density, acceptance rate, and energy at intervals.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n md_analysis python benchmark/benchmark_1m9z_npt_mdpy.py
"""

import os, sys, time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cupy as cp
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

from mdpy.io.psf_parser import PSFParser
from mdpy.io.pdb_parser import PDBParser
from mdpy.io.charmm_toppar_parser import CharmmTopparParser
from mdpy.force.factories.charmm import create_charmm_forces
from mdpy.integrator.langevin import LangevinBAOABIntegrator
from mdpy.core.state import State
from mdpy.system import System
from mdpy.constraint.constraint_scheme import create_constraints
from mdpy.barostat import MonteCarloBarostat
from mdpy.utils import generate_velocity_from_temperature

CUTOFF = 12.0
TIME_STEP_FS = 2
EWALD_RTOL = float(os.environ.get("EWALD_RTOL", "1e-5"))
FOURIER_SPACING = float(os.environ.get("FOURIER_SPACING", "1.2"))
PRESSURE_BAR = 1.0
TEMPERATURE = 300.0
MC_FREQUENCY = 25
NVT_STEPS = 100
NPT_STEPS = 2000
REPORT_INTERVAL = 50

KCAL_PER_INTERNAL = 1.0 / 4.1840286576e-4

psf = PSFParser(os.path.join(DATA_DIR, "1M9Z.psf"))
pdb = PDBParser(os.path.join(DATA_DIR, "1M9Z_minimized.pdb"))
toppar = CharmmTopparParser(
    os.path.join(DATA_DIR, "par_all36_prot.prm"),
    os.path.join(DATA_DIR, "toppar_water_ions.str"),
)
topology = psf.topology
parameter_set = toppar.resolve_parameter_set(topology, psf.particle_type_names)
pbc_matrix = pdb.pbc_matrix

state = State(topology.num_particles)
state.set_pbc(pbc_matrix)
state.set_positions(pdb.positions)
state.set_particle_charges(psf.particle_charges)
state.set_particle_masses(psf.particle_masses)
state.set_particle_type_indices(parameter_set.particle_type_indices)
state.set_particle_molecule_ids(psf.particle_molecule_ids)

forces = create_charmm_forces(
    topology, parameter_set, pbc_matrix,
    cutoff=CUTOFF, ewald_rtol=EWALD_RTOL, fourier_spacing=FOURIER_SPACING,
)
system = System(topology, state)
for f in forces["bonded"]:
    system.add_force_term(f)
system.add_force_term(forces["nonbonded"])
system.add_force_term(forces["pme"], stream="pme")

constraints = create_constraints(
    topology, parameter_set, scheme="h-bonds",
    particle_masses=psf.particle_masses,
    particle_residue_ids=psf.particle_residue_ids,
    particle_residue_names=psf.particle_residue_names,
)
for c in constraints:
    system.add_constraint(c)

velocities = generate_velocity_from_temperature(TEMPERATURE, psf.particle_masses, seed=42)
system.set_velocities(velocities)

pme = forces["pme"]
integrator = LangevinBAOABIntegrator(TIME_STEP_FS, TEMPERATURE, 1.0)

num_molecules = len(set(psf.particle_molecule_ids))
total_mass = float(np.sum(psf.particle_masses))
initial_volume = float(pbc_matrix[0, 0] * pbc_matrix[1, 1] * pbc_matrix[2, 2])
initial_density = total_mass * 1.66054 / initial_volume

print("mdpy 1M9Z NPT benchmark (Monte Carlo barostat)")
print(f"  Atoms:          {topology.num_particles}")
print(f"  Molecules:      {num_molecules}")
print(f"  Box:            {pbc_matrix[0,0]:.1f} x {pbc_matrix[1,1]:.1f} x {pbc_matrix[2,2]:.1f} A")
print(f"  Volume:         {initial_volume:.0f} A^3")
print(f"  Density:        {initial_density:.4f} g/cm^3")
print(f"  PME grid:       {pme.grid_x} x {pme.grid_y} x {pme.grid_z}")
print(f"  Target:         {PRESSURE_BAR} bar, {TEMPERATURE} K")
print(f"  MC frequency:   {MC_FREQUENCY}")
print()

# ---- NVT equilibration (no barostat) ----
print(f"NVT equilibration ({NVT_STEPS} steps)...")
cp.cuda.Stream.null.synchronize()
t0 = time.perf_counter()
for i in range(NVT_STEPS):
    system.update_neighbor_list(sync_interval=20)
    system.compute_forces()
    integrator.step(system)
    system.apply_constraints(TIME_STEP_FS)
cp.cuda.Stream.null.synchronize()
nvt_elapsed = time.perf_counter() - t0
print(f"  done in {nvt_elapsed:.1f}s ({nvt_elapsed/NVT_STEPS*1000:.1f} ms/step)")

# ---- Add barostat for NPT ----
np.random.seed(42)
barostat = MonteCarloBarostat(PRESSURE_BAR, TEMPERATURE, frequency=MC_FREQUENCY)
system.add_barostat(barostat)

# ---- NPT production ----
print(f"\nNPT simulation ({NPT_STEPS} steps):")
print(f"  {'Step':>6s}  {'Volume(A^3)':>12s}  {'Box_len':>8s}  {'Density':>8s}  {'Accept%':>8s}  {'E_pot(kcal/mol)':>18s}")
print(f"  {'------':>6s}  {'------------':>12s}  {'--------':>8s}  {'--------':>8s}  {'--------':>8s}  {'------------------':>18s}")

block_times = []
for i in range(NPT_STEPS):
    cp.cuda.Stream.null.synchronize()
    t_step = time.perf_counter()

    system.update_neighbor_list(sync_interval=20)
    system.compute_forces()
    integrator.step(system)
    system.apply_constraints(TIME_STEP_FS)
    system.apply_barostats()

    cp.cuda.Stream.null.synchronize()
    block_times.append(time.perf_counter() - t_step)

    if (i + 1) % REPORT_INTERVAL == 0:
        s = system.state
        vol = s.box_x * s.box_y * s.box_z
        box_len = s.box_x
        density = total_mass * 1.66054 / vol
        accept_pct = barostat.acceptance_rate * 100
        energy_dict = system.dump_energy()
        e_total = sum(energy_dict.values()) * KCAL_PER_INTERNAL
        recent_ms = np.mean(block_times[-REPORT_INTERVAL:]) * 1000
        print(f"  {i+1:6d}  {vol:12.0f}  {box_len:8.2f}  {density:8.4f}  {accept_pct:8.1f}  {e_total:18.1f}")

avg_ms = np.mean(block_times) * 1000
final_vol = system.state.box_x * system.state.box_y * system.state.box_z
final_density = total_mass * 1.66054 / final_vol
vol_change = (final_vol - initial_volume) / initial_volume * 100

print(f"\nSummary:")
print(f"  Initial density:  {initial_density:.4f} g/cm^3")
print(f"  Final density:    {final_density:.4f} g/cm^3")
print(f"  Volume change:    {vol_change:+.1f}%")
print(f"  Avg acceptance:   {barostat.acceptance_rate*100:.1f}%")
print(f"  NPT ms/step:      {avg_ms:.1f} (vs {nvt_elapsed/NVT_STEPS*1000:.1f} NVT)")
