"""mdpy 1M9Z minimization benchmark — verifies SteepestDescentMinimizer.

Loads the pre-minimized 1M9Z structure (from CRYST1 box), runs steepest
descent, and checks that energy decreases and max force is reduced.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n md_analysis python benchmark/benchmark_1m9z_minimize.py
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
from mdpy.core.state import State
from mdpy.system import System
from mdpy.minimizer.steepest_descent import SteepestDescentMinimizer
from mdpy.unit import (
    Quantity,
    default_energy_unit,
    default_force_unit,
    kilojoule_permol,
    kilojoule_permol_over_angstrom,
)

# ---- Parameters ----
CUTOFF = 12.0
STEP_SIZE = 0.01
MAX_STEPS = 1000
REPORT_EVERY = 50

# ---- Unit conversion factors ----
_TO_KJMOL = float(
    Quantity(1.0, default_energy_unit).convert_to(kilojoule_permol).value
)
_TO_KJMOLA = float(
    Quantity(1.0, default_force_unit).convert_to(kilojoule_permol_over_angstrom).value
)

# ---- Load system ----
print("Loading 1M9Z...")
psf = PSFParser(os.path.join(DATA_DIR, "1M9Z.psf"))
pdb = PDBParser(os.path.join(DATA_DIR, "1M9Z_minimized.pdb"))
toppar = CharmmTopparParser(
    os.path.join(DATA_DIR, "par_all36_prot.prm"),
    os.path.join(DATA_DIR, "toppar_water_ions.str"),
)

topology = psf.topology
parameter_set = toppar.resolve_parameter_set(topology, psf.particle_type_names)

# Use box from PDB CRYST1 record
pbc_matrix = pdb.pbc_matrix.astype(np.float64)
box_size = pbc_matrix[0, 0]  # cubic

forces = create_charmm_forces(
    topology, parameter_set, pbc_matrix, cutoff=CUTOFF,
)

state = State(topology.num_particles)
state.set_pbc(pbc_matrix)
state.set_positions(pdb.positions)
state.set_particle_charges(psf.particle_charges)
state.set_particle_masses(psf.particle_masses)
state.set_particle_type_indices(parameter_set.particle_type_indices)
state.set_velocities(np.zeros((topology.num_particles, 3), dtype=np.float64))

system = System(topology, state)
for f in forces["bonded"]:
    system.add_force_term(f)
system.add_force_term(forces["nonbonded"])
system.add_force_term(forces["pme"])

# ---- Minimization ----
print(f"\n{'='*70}")
print(f"Steepest descent minimization")
print(f"  Atoms:     {topology.num_particles}")
print(f"  Box:       {box_size:.1f} A (from CRYST1)")
print(f"  step_size: {STEP_SIZE}")
print(f"  max steps: {MAX_STEPS}")
print(f"{'='*70}")

minimizer = SteepestDescentMinimizer(step_size=STEP_SIZE)

# Initial evaluation
system.update_neighbor_list(force_rebuild=True)
system.compute_forces(compute_energy=True)
cp.cuda.Stream.null.synchronize()

e_initial = float(cp.asnumpy(state.d_energy[0]))
mf_initial = minimizer.compute_max_force(system)
rf_initial = minimizer.compute_rms_force(system)

print(f"\n  {'':>6s}  {'Energy (kJ/mol)':>18s}  {'MaxF (kJ/mol/A)':>18s}  {'RMSF (kJ/mol/A)':>18s}")
print(f"  {'init':>6s}  {e_initial * _TO_KJMOL:18.3f}  {mf_initial * _TO_KJMOLA:18.4f}  {rf_initial * _TO_KJMOLA:18.4f}")

# Minimization loop
print(f"\n  {'Step':>6s}  {'Energy (kJ/mol)':>18s}  {'MaxF (kJ/mol/A)':>18s}  {'RMSF (kJ/mol/A)':>18s}")
print(f"  {'------':>6s}  {'------------------':>18s}  {'------------------':>18s}  {'------------------':>18s}")

t0 = time.perf_counter()
for step in range(MAX_STEPS):
    system.update_neighbor_list(sync_interval=10)
    system.compute_forces(compute_energy=False)
    minimizer.step(system)

    if step % REPORT_EVERY == 0:
        system.compute_forces(compute_energy=True)
        e = float(cp.asnumpy(state.d_energy[0]))
        mf = minimizer.compute_max_force(system)
        rf = minimizer.compute_rms_force(system)
        print(
            f"  {step:6d}  {e * _TO_KJMOL:18.3f}  "
            f"{mf * _TO_KJMOLA:18.4f}  {rf * _TO_KJMOLA:18.4f}"
        )
        if not np.isfinite(e):
            print(f"  ERROR: energy is NaN at step {step}")
            sys.exit(1)

cp.cuda.Stream.null.synchronize()
elapsed = time.perf_counter() - t0

# Final evaluation
system.compute_forces(compute_energy=True)
e_final = float(cp.asnumpy(state.d_energy[0]))
mf_final = minimizer.compute_max_force(system)
rf_final = minimizer.compute_rms_force(system)

print(
    f"\n  {'final':>6s}  {e_final * _TO_KJMOL:18.3f}  "
    f"{mf_final * _TO_KJMOLA:18.4f}  {rf_final * _TO_KJMOLA:18.4f}"
)
print(f"\n  dE = {(e_final - e_initial) * _TO_KJMOL:+.3f} kJ/mol")
print(f"  maxF: {mf_initial * _TO_KJMOLA:.4f} -> {mf_final * _TO_KJMOLA:.4f} kJ/(mol·A)")
print(f"  duration: {elapsed:.1f}s ({MAX_STEPS/elapsed:.0f} steps/s)")

# ---- Validation ----
print(f"\n{'='*70}")
print("Validation")
print(f"{'='*70}")

checks = []
checks.append(
    (
        f"Energy decreased (dE = {(e_final - e_initial) * _TO_KJMOL:+.1f} kJ/mol)",
        e_final <= e_initial + 1e-6,
    )
)
checks.append(
    (
        f"Max force decreased ({mf_initial * _TO_KJMOLA:.2f} -> {mf_final * _TO_KJMOLA:.2f})",
        mf_final < mf_initial,
    )
)
checks.append(("Energy finite", np.isfinite(e_final)))
checks.append(("Max force finite", np.isfinite(mf_final)))

all_ok = True
for label, ok in checks:
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_ok = False
    print(f"  [{status}] {label}")

if all_ok:
    print(f"\n  All checks PASSED")
else:
    print(f"\n  Some checks FAILED")
    sys.exit(1)
