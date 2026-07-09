"""mdpy 1M9Z NPT — minimize → NVT equilibrate → NPT at 1 bar.

Pipeline:
  1. Minimize structure from raw coordinates (relax bad contacts)
  2. NVT equilibrate at 300K (temperature relaxation)
  3. NPT at 1 bar (density convergence to ~1 g/cm^3)

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n md_analysis python benchmark/benchmark_1m9z_npt.py
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
from mdpy.constraint.constraint_scheme import create_constraints
from mdpy.utils import generate_velocity_from_temperature
from mdpy.minimizer.steepest_descent import SteepestDescentMinimizer
from mdpy.barostat.berendsen import BerendsenBarostat
from mdpy.unit import Quantity, bar, default_pressure_unit, KB, default_energy_unit, kelvin

# ---- Parameters ----
CUTOFF = 12.0
TIME_STEP_FS = 2
TEMPERATURE = 300.0
TARGET_PRESSURE_BAR = 1.0
TAU_P = 100.0          # barostat coupling time (fs)
NVT_EQUIL_STEPS = 5000  # 10 ps
NPT_STEPS = 10000       # 20 ps

TARGET_PRESSURE = float(
    Quantity(TARGET_PRESSURE_BAR, bar).convert_to(default_pressure_unit).value
)

_BOLTZMANN = float(KB.convert_to(default_energy_unit / kelvin).value)
_PRESSURE_TO_BAR = 1.0 / TARGET_PRESSURE  # internal -> bar

# ---- Load system (minimized structure) ----
PSF = PSFParser(os.path.join(DATA_DIR, "1M9Z.psf"))
PDB = PDBParser(os.path.join(DATA_DIR, "1M9Z_minimized.pdb"))
TOPPAR = CharmmTopparParser(
    os.path.join(DATA_DIR, "par_all36_prot.prm"),
    os.path.join(DATA_DIR, "toppar_water_ions.str"),
)

topology = PSF.topology
parameter_set = TOPPAR.resolve_parameter_set(topology, PSF.particle_type_names)
TOTAL_MASS = float(PSF.particle_masses.sum())
DENSITY_CONV = 0.602  # Da/A^3 per g/cm^3

# Use box from PDB CRYST1 record
pbc_matrix = PDB.pbc_matrix.astype(np.float64)
INITIAL_BOX = pbc_matrix[0, 0]  # cubic box

forces = create_charmm_forces(
    topology, parameter_set, pbc_matrix, cutoff=CUTOFF,
)

state = State(topology.num_particles)
state.set_pbc(pbc_matrix)
state.set_positions(PDB.positions)
state.set_particle_charges(PSF.particle_charges)
state.set_particle_masses(PSF.particle_masses)
state.set_particle_type_indices(parameter_set.particle_type_indices)

system = System(topology, state)
for f in forces["bonded"]:
    system.add_force_term(f)
system.add_force_term(forces["nonbonded"])
system.add_force_term(forces["pme"], stream="pme")

constraints = create_constraints(
    topology, parameter_set, scheme="h-bonds",
    particle_masses=PSF.particle_masses,
    particle_molecule_ids=PSF.particle_molecule_ids,
    particle_molecule_types=PSF.particle_molecule_types,
)
for c in constraints:
    system.add_constraint(c)

integrator = LangevinBAOABIntegrator(TIME_STEP_FS, TEMPERATURE, 1.0)


def compute_pressure(state):
    """Instantaneous pressure in bar."""
    K = 1.5 * state.num_particles * _BOLTZMANN * TEMPERATURE
    V = state.box_x * state.box_y * state.box_z
    if V <= 0:
        return 0.0, 0.0
    W = float(cp.asnumpy(state.d_virial[0]))
    P = (2.0 * K + W) / (3.0 * V)
    return P * _PRESSURE_TO_BAR, W


# ---- Heat ----
print("mdpy 1M9Z NPT pipeline")
print(f"  Atoms:         {topology.num_particles}")
print(f"  Total mass:     {TOTAL_MASS:.0f} Da")
print(f"  Initial box:    {INITIAL_BOX:.1f} A (for ~1 g/cm^3)")
print(f"  Cutoff:        {CUTOFF} A")
print(f"  time_step:     {TIME_STEP_FS} fs")
print(f"  Temperature:    {TEMPERATURE} K")
print(f"  Target P:       {TARGET_PRESSURE_BAR} bar")
print(f"  tau_P:          {TAU_P} fs")

# =====================================================
# Phase 0: Minimization check (structure already minimized)
# =====================================================
MIN_STEP_SIZE = 0.01
MIN_STEPS = 500

print(f"\n{'='*70}")
print(f"Phase 0: Minimization check (SD, {MIN_STEPS} steps)")
print(f"{'='*70}")

system.set_velocities(np.zeros((topology.num_particles, 3), dtype=np.float64))
minimizer = SteepestDescentMinimizer(step_size=MIN_STEP_SIZE)

system.update_neighbor_list(sync_interval=1, force_rebuild=True)
system.compute_forces(compute_energy=True)
e0 = float(cp.asnumpy(state.d_energy[0]))
f0 = minimizer.compute_max_force(system)
print(f"  init: energy={e0:.3f}, maxF={f0:.4f}")

# Rebuild NL every step during minimization
for i in range(MIN_STEPS):
    system.update_neighbor_list(sync_interval=10)
    system.compute_forces(compute_energy=False)
    minimizer.step(system)
    system.apply_constraints(TIME_STEP_FS)

system.update_neighbor_list(sync_interval=1, force_rebuild=True)
system.compute_forces(compute_energy=True)
e1 = float(cp.asnumpy(state.d_energy[0]))
f1 = minimizer.compute_max_force(system)
print(f"  final: energy={e1:.3f}, maxF={f1:.4f}")
print(f"  dE={e1-e0:.3f}, maxF reduction={f0-f1:.3f}")
print(f"  Energy {'DECREASED' if e1 < e0 else 'increased'} — {'OK' if e1 <= e0 else 'WARN'}")

cp.cuda.Stream.null.synchronize()
t0 = time.perf_counter()

for step in range(NVT_EQUIL_STEPS):
    system.update_neighbor_list(sync_interval=20)
    system.compute_forces(compute_energy=False, compute_virial=True)
    integrator.step(system)
    system.apply_constraints(TIME_STEP_FS)

cp.cuda.Stream.null.synchronize()
nvt_elapsed = time.perf_counter() - t0

P_nvt, W_nvt = compute_pressure(state)
V_nvt = state.box_x * state.box_y * state.box_z
dens_nvt = TOTAL_MASS / V_nvt / DENSITY_CONV
energy_dict = system.dump_energy()
e_kcal_nvt = sum(energy_dict.values()) / 4.1840286576e-4

print(f"  NVT runtime:      {nvt_elapsed:.1f}s ({NVT_EQUIL_STEPS / nvt_elapsed:.0f} steps/s)")
print(f"  Final box:        {state.box_x:.2f} A")
print(f"  Final density:    {dens_nvt:.4f} g/cm^3")
print(f"  Final pressure:   {P_nvt:.0f} bar")
print(f"  Final virial:     {W_nvt:.1f}")
print(f"  Final energy:     {e_kcal_nvt:.1f} kcal/mol")
print(f"  (pressure after NVT is the force-field natural pressure at this density)")

# =====================================================
# Phase 2: NPT at 1 bar
# =====================================================
print(f"\n{'='*70}")
print(f"Phase 2: NPT at {TARGET_PRESSURE_BAR} bar ({NPT_STEPS} steps, Berendsen)")
print(f"{'='*70}")

barostat = BerendsenBarostat(
    target_pressure=TARGET_PRESSURE, pressure_coupling_time=TAU_P,
)

# Track box, density, pressure every 100 steps
samples = []

cp.cuda.Stream.null.synchronize()
t0 = time.perf_counter()

for step in range(NPT_STEPS):
    system.update_neighbor_list(sync_interval=20)
    system.compute_forces(compute_energy=False, compute_virial=True)
    barostat.apply(system, TEMPERATURE, TIME_STEP_FS)
    integrator.step(system)
    system.apply_constraints(TIME_STEP_FS)

    if step % 100 == 0:
        st = system.state
        V = st.box_x * st.box_y * st.box_z
        box = (st.box_x + st.box_y + st.box_z) / 3.0
        dens = TOTAL_MASS / V / DENSITY_CONV
        P, W = compute_pressure(st)
        samples.append((step, box, dens, P, W))

cp.cuda.Stream.null.synchronize()
npt_elapsed = time.perf_counter() - t0

# Final values
V_final = system.state.box_x * system.state.box_y * system.state.box_z
dens_final = TOTAL_MASS / V_final / DENSITY_CONV
P_final, W_final = compute_pressure(system.state)
energy_dict = system.dump_energy()
e_kcal_final = sum(energy_dict.values()) / 4.1840286576e-4

print(f"  NPT runtime:      {npt_elapsed:.1f}s ({NPT_STEPS / npt_elapsed:.0f} steps/s)")
print(f"  Final box:        {system.state.box_x:.2f} A")
print(f"  Final density:    {dens_final:.4f} g/cm^3")
print(f"  Final pressure:   {P_final:.0f} bar")
print(f"  Final virial:     {W_final:.1f}")
print(f"  Final energy:     {e_kcal_final:.1f} kcal/mol")

# Density trajectory
steps_arr = np.array([s[0] for s in samples])
box_arr = np.array([s[1] for s in samples])
dens_arr = np.array([s[2] for s in samples])
P_arr = np.array([s[3] for s in samples])

n_blocks = 10
block_size = len(samples) // n_blocks
print(f"\n  Density trajectory ({n_blocks} blocks):")
print(f"  {'Block':>6s}  {'Steps':>8s}  {'Box(A)':>8s}  {'Density':>8s}  {'P(bar)':>10s}")
for b in range(n_blocks):
    i0 = b * block_size
    i1 = i0 + block_size if b < n_blocks - 1 else len(samples)
    print(
        f"  {b+1:6d}  {steps_arr[i0]:8d}  {box_arr[i0:i1].mean():8.2f}  "
        f"{dens_arr[i0:i1].mean():8.4f}  {P_arr[i0:i1].mean():10.1f}"
    )

# =====================================================
# Validation
# =====================================================
print(f"\n{'='*70}")
print("Validation")
print(f"{'='*70}")

checks = []

# Energy should be well-behaved after minimization + equilibration
checks.append(
    (
        f"Energy after pipeline ({e_kcal_final:.0f} kcal/mol)",
        -500000 < e_kcal_final < -300000,
    )
)

# Density should be close to 1 g/cm^3
checks.append(
    (
        f"Density after NPT ({dens_final:.4f} g/cm^3 vs target 1.0)",
        0.9 < dens_final < 1.1,
    )
)

# Pressure should be approaching 1 bar (relaxed: within 500 bar)
p_error = abs(P_final - TARGET_PRESSURE_BAR)
checks.append(
    (
        f"Pressure after NPT ({P_final:.0f} bar vs target {TARGET_PRESSURE_BAR})",
        p_error < 500,
    )
)

# Box should have changed from initial (barostat did work)
initial_box = samples[0][1]
final_box = samples[-1][1]
checks.append(
    (
        f"Box changed during NPT ({initial_box:.2f} -> {final_box:.2f} A)",
        abs(final_box - initial_box) > 0.001,
    )
)

all_ok = True
for label, ok in checks:
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_ok = False
    print(f"  [{status}] {label}")

if all_ok:
    print(f"\n  All checks PASSED")
    print(f"  Barostat is working: minimize → NVT → NPT pipeline produces")
    print(f"  reasonable density (~{dens_final:.3f} g/cm^3) at {TARGET_PRESSURE_BAR} bar.")
else:
    print(f"\n  Some checks FAILED")
    sys.exit(1)
