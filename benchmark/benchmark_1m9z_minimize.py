"""mdpy 1M9Z minimization benchmark — verifies SteepestDescentMinimizer.

Loads the pre-minimized 1M9Z structure, runs steepest descent, and checks
that energy decreases and max force is reduced.

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
from mdpy.constraint.constraint_scheme import create_constraints
from mdpy.minimizer.steepest_descent import SteepestDescentMinimizer

# ---- Parameters ----
BOX_SIZE = 108.0
CUTOFF = 12.0
STEP_SIZE = 0.01
MAX_STEPS = 1000
REPORT_EVERY = 50

# ---- Load system ----
print("Loading 1M9Z (minimized)...")
psf = PSFParser(os.path.join(DATA_DIR, "1M9Z.psf"))
pdb = PDBParser(os.path.join(DATA_DIR, "1M9Z_minimized.pdb"))
toppar = CharmmTopparParser(
    os.path.join(DATA_DIR, "par_all36_prot.prm"),
    os.path.join(DATA_DIR, "toppar_water_ions.str"),
)
topology = psf.topology
parameter_set = toppar.resolve_parameter_set(topology, psf.particle_type_names)

pbc_matrix = np.eye(3, dtype=np.float64) * BOX_SIZE

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
system.add_force_term(forces["pme"], stream="pme")

constraints = create_constraints(
    topology, parameter_set, scheme="h-bonds",
    particle_masses=psf.particle_masses,
    particle_molecule_ids=psf.particle_molecule_ids,
    particle_molecule_types=psf.particle_molecule_types,
)
for c in constraints:
    system.add_constraint(c)

# ---- Minimization ----
print(f"\n{'='*70}")
print(f"Steepest descent minimization")
print(f"  Atoms:    {topology.num_particles}")
print(f"  Box:      {BOX_SIZE} A")
print(f"  step_size: {STEP_SIZE}")
print(f"  max steps: {MAX_STEPS}")
print(f"{'='*70}")

minimizer = SteepestDescentMinimizer(step_size=STEP_SIZE)

# Initial evaluation
system.update_neighbor_list(force_rebuild=True)
system.compute_forces(compute_energy=True)
cp.cuda.Stream.null.synchronize()

energy_initial = float(cp.asnumpy(state.d_energy[0]))
max_f_initial = minimizer.compute_max_force(system)
rms_f_initial = minimizer.compute_rms_force(system)

print(f"\n  Initial:")
print(f"    energy:     {energy_initial:.6f} internal")
print(f"    max force:  {max_f_initial:.6f}")
print(f"    RMS force:  {rms_f_initial:.6f}")

# Minimization loop
print(f"\n  {'Step':>6s}  {'Energy':>14s}  {'MaxF':>14s}  {'RMSF':>14s}")
print(f"  {'------':>6s}  {'--------------':>14s}  {'--------------':>14s}  {'--------------':>14s}")

t0 = time.perf_counter()
for step in range(MAX_STEPS):
    system.update_neighbor_list(sync_interval=10)
    system.compute_forces(compute_energy=False)
    minimizer.step(system)

    if step % REPORT_EVERY == 0:
        # Recompute energy for reporting (zero_energy is called inside compute_forces)
        system.compute_forces(compute_energy=True)
        e = float(cp.asnumpy(state.d_energy[0]))
        mf = minimizer.compute_max_force(system)
        rf = minimizer.compute_rms_force(system)
        print(f"  {step:6d}  {e:14.6f}  {mf:14.6f}  {rf:14.6f}")

        if not np.isfinite(e):
            print(f"  ERROR: energy is NaN at step {step}")
            sys.exit(1)

cp.cuda.Stream.null.synchronize()
elapsed = time.perf_counter() - t0

# Final evaluation
system.compute_forces(compute_energy=True)
energy_final = float(cp.asnumpy(state.d_energy[0]))
max_f_final = minimizer.compute_max_force(system)
rms_f_final = minimizer.compute_rms_force(system)

print(f"\n  Final (after {MAX_STEPS} steps, {elapsed:.1f}s):")
print(f"    energy:     {energy_final:.6f}")
print(f"    max force:  {max_f_final:.6f}  (reduction: {max_f_initial - max_f_final:.6f})")
print(f"    RMS force:  {rms_f_final:.6f}  (reduction: {rms_f_initial - rms_f_final:.6f})")
print(f"    dE:         {energy_final - energy_initial:+.6f}")

# ---- Validation ----
print(f"\n{'='*70}")
print("Validation")
print(f"{'='*70}")

checks = []
checks.append(
    (
        f"Energy decreased or stayed same (dE = {energy_final - energy_initial:+.6f})",
        energy_final <= energy_initial + 1e-6,
    )
)
checks.append(
    (
        f"Max force decreased (from {max_f_initial:.4f} to {max_f_final:.4f})",
        max_f_final < max_f_initial,
    )
)
checks.append(("Energy is finite", np.isfinite(energy_final)))
checks.append(("Max force is finite", np.isfinite(max_f_final)))

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
