"""mdpy 1M9Z minimization benchmark — compares all four minimizers.

Tests SteepestDescent, ConjugateGradient, LBFGS, and FIRE on the same
1M9Z minimized structure, same number of steps. Reports energy (kJ/mol)
and max force (kJ/mol/A) for each, showing which converges fastest.

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
from mdpy.minimizer.conjugate_gradient import ConjugateGradientMinimizer
from mdpy.minimizer.lbfgs import LBFGSMinimizer
from mdpy.minimizer.fire import FIREMinimizer
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
NUM_STEPS = 500
E_M1 = None

# ---- Unit conversion ----
_TO_KJMOL = float(Quantity(1.0, default_energy_unit).convert_to(kilojoule_permol).value)
_TO_KJMOLA = float(
    Quantity(1.0, default_force_unit).convert_to(kilojoule_permol_over_angstrom).value
)


def build_system():
    psf = PSFParser(os.path.join(DATA_DIR, "1M9Z.psf"))
    pdb = PDBParser(os.path.join(DATA_DIR, "1M9Z_minimized.pdb"))
    toppar = CharmmTopparParser(
        os.path.join(DATA_DIR, "par_all36_prot.prm"),
        os.path.join(DATA_DIR, "toppar_water_ions.str"),
    )
    topology = psf.topology
    parameter_set = toppar.resolve_parameter_set(topology, psf.particle_type_names)
    pbc_matrix = pdb.pbc_matrix.astype(np.float64)

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

    return system, pdb


def run_minimizer(name, minimizer, system, initial_positions):
    """Run minimizer, return (name, e_init, e_final, mf_init, mf_final, elapsed)."""
    # Reset positions for fair comparison
    system.state.set_positions(initial_positions)
    system.update_neighbor_list(force_rebuild=True)
    system.compute_forces(compute_energy=True)

    e_init = float(cp.asnumpy(system.state.d_energy[0]))
    mf_init = minimizer.compute_max_force(system)

    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()

    for _ in range(NUM_STEPS):
        system.update_neighbor_list(sync_interval=10)
        system.compute_forces(compute_energy=False)
        minimizer.step(system)

    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - t0

    system.compute_forces(compute_energy=True)
    e_final = float(cp.asnumpy(system.state.d_energy[0]))
    mf_final = minimizer.compute_max_force(system)

    return name, e_init, e_final, mf_init, mf_final, elapsed


# ---- Build system once ----
print("Loading 1M9Z...")
system, pdb = build_system()
box_size = system.state.box_x

print(f"\n{'='*70}")
print(f"Minimizer comparison — 1M9Z ({system.num_particles} atoms, {box_size:.0f} A box)")
print(f"  steps: {NUM_STEPS}, step_size: {STEP_SIZE} (CG: 0.001)")
print(f"{'='*70}")

# ---- Run all four minimizers ----
minimizers = [
    ("SteepestDescent", SteepestDescentMinimizer(step_size=STEP_SIZE)),
    ("ConjugateGradient", ConjugateGradientMinimizer(step_size=0.001)),
    ("LBFGS", LBFGSMinimizer(step_size=STEP_SIZE)),
    ("FIRE", FIREMinimizer(time_step=STEP_SIZE, n_min=5)),
]

results = []
for name, minim in minimizers:
    print(f"\n  Running {name}...", end=" ", flush=True)
    result = run_minimizer(name, minim, system, pdb.positions)
    results.append(result)
    e_init, e_final, mf_init, mf_final, elapsed = result[1:]
    print(
        f"dE = {(e_final - e_init) * _TO_KJMOL:+.1f} kJ/mol, "
        f"maxF: {mf_init * _TO_KJMOLA:.1f} -> {mf_final * _TO_KJMOLA:.2f}, "
        f"{elapsed:.2f}s"
    )

# ---- Comparison table ----
print(f"\n\n{'='*90}")
print("Results")
print(f"{'='*90}")
print(
    f"  {'Minimizer':<20s}  {'E_init(kJ/mol)':>16s}  {'E_final(kJ/mol)':>16s}  "
    f"{'dE(kJ/mol)':>12s}  {'maxF_init':>10s}  {'maxF_final':>10s}  {'time':>7s}"
)
print(
    f"  {'-'*20:<20s}  {'-'*16:<16s}  {'-'*16:<16s}  "
    f"{'-'*12:<12s}  {'-'*10:<10s}  {'-'*10:<10s}  {'-'*7:<7s}"
)

for name, e_init, e_final, mf_init, mf_final, elapsed in results:
    print(
        f"  {name:<20s}  {e_init * _TO_KJMOL:16.1f}  {e_final * _TO_KJMOL:16.1f}  "
        f"{(e_final - e_init) * _TO_KJMOL:+12.1f}  "
        f"{mf_init * _TO_KJMOLA:10.2f}  {mf_final * _TO_KJMOLA:10.3f}  {elapsed:6.2f}s"
    )

# ---- Winner ----
best = min(results, key=lambda r: r[2])  # lowest final energy
print(f"\n  Lowest final energy: {best[0]} ({best[2] * _TO_KJMOL:.1f} kJ/mol)")
best_f = min(results, key=lambda r: r[4])  # lowest final max force
print(f"  Lowest final maxF:   {best_f[0]} ({best_f[4] * _TO_KJMOLA:.3f} kJ/(mol·A))")

# ---- Validation ----
print(f"\n{'='*90}")
print("Validation")
print(f"{'='*90}")

checks = []
for name, e_init, e_final, mf_init, mf_final, elapsed in results:
    checks.append(
        (
            f"{name}: energy decreased",
            e_final < e_init + 1e-6,
        )
    )
    checks.append(
        (
            f"{name}: maxF decreased",
            mf_final < mf_init,
        )
    )
    checks.append(
        (
            f"{name}: finite",
            np.isfinite(e_final) and np.isfinite(mf_final),
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
else:
    print(f"\n  Some checks FAILED")
    sys.exit(1)
