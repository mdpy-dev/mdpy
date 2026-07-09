"""mdpy 1M9Z minimization benchmark — FIRE minimize raw PDB, save result.

Reads 1M9Z.pdb (raw solvated structure at 100 A), runs FIRE minimization,
then saves the minimized positions to 1M9Z_minimized.pdb using PDBWriter.
Also compares all four minimizers (SD, CG, LBFGS, FIRE) for reference.

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
from mdpy.io.pdb_writer import PDBWriter
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
FIRE_STEPS = 3000
COMPARE_STEPS = 500

_TO_KJMOL = float(Quantity(1.0, default_energy_unit).convert_to(kilojoule_permol).value)
_TO_KJMOLA = float(
    Quantity(1.0, default_force_unit).convert_to(kilojoule_permol_over_angstrom).value
)

# ---- Load raw 1M9Z (100 A box, 95,567 atoms with water) ----
DATA_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

psf = PSFParser(os.path.join(DATA_LOCAL, "1M9Z.psf"))
pdb = PDBParser(os.path.join(DATA_LOCAL, "1M9Z.pdb"))
toppar = CharmmTopparParser(
    os.path.join(DATA_LOCAL, "par_all36_prot.prm"),
    os.path.join(DATA_LOCAL, "toppar_water_ions.str"),
)
topology = psf.topology
parameter_set = toppar.resolve_parameter_set(topology, psf.particle_type_names)

# Box from CRYST1
pbc_matrix = pdb.pbc_matrix.astype(np.float64)
box_size = pbc_matrix[0, 0]


def build_system():
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
    return system


# ---- Header ----
print(f"mdpy 1M9Z minimization")
print(f"  Source:  1M9Z.pdb (raw solvated)")
print(f"  Atoms:   {topology.num_particles}")
print(f"  Box:     {box_size:.0f} A (from CRYST1)")
print(f"  Cutoff:  {CUTOFF} A")

# =====================================================
# Phase 1: FIRE minimize + save
# =====================================================
print(f"\n{'='*70}")
print(f"Phase 1: FIRE minimize ({FIRE_STEPS} steps) → save to PDB")
print(f"{'='*70}")

system = build_system()
minimizer = FIREMinimizer(time_step=STEP_SIZE, n_min=5)

system.update_neighbor_list(force_rebuild=True)
system.compute_forces(compute_energy=True)
e0 = float(cp.asnumpy(system.state.d_energy[0]))
mf0 = minimizer.compute_max_force(system)
print(f"  init:  E={e0 * _TO_KJMOL:14.1f} kJ/mol, maxF={mf0 * _TO_KJMOLA:.2f} kJ/(mol·A)")

cp.cuda.Stream.null.synchronize()
t0 = time.perf_counter()

for i in range(FIRE_STEPS):
    system.update_neighbor_list(sync_interval=20)
    system.compute_forces(compute_energy=False)
    minimizer.step(system)
    if i % 1000 == 0 and i > 0:
        system.compute_forces(compute_energy=True)
        e = float(cp.asnumpy(system.state.d_energy[0]))
        print(f"  step {i:4d}: E={e * _TO_KJMOL:14.1f} kJ/mol")

cp.cuda.Stream.null.synchronize()
elapsed = time.perf_counter() - t0

system.compute_forces(compute_energy=True)
e1 = float(cp.asnumpy(system.state.d_energy[0]))
mf1 = minimizer.compute_max_force(system)
print(f"  final: E={e1 * _TO_KJMOL:14.1f} kJ/mol, maxF={mf1 * _TO_KJMOLA:.2f}")
print(f"  dE={(e1 - e0) * _TO_KJMOL:+.1f} kJ/mol, {elapsed:.1f}s ({FIRE_STEPS / elapsed:.0f} steps/s)")

# ---- Save using PDBWriter ----
pos = system.state.download_positions()
writer = PDBWriter(
    particle_ids=np.array(psf.particle_ids),
    particle_names=list(psf.particle_names),
    particle_molecule_ids=np.array(psf.particle_molecule_ids),
    particle_molecule_types=list(psf.particle_molecule_types),
    particle_chain_ids=list(psf.particle_chain_ids),
)
out_path = os.path.join(DATA_LOCAL, "1M9Z_minimized.pdb")
writer.write(out_path, positions=pos, pbc_matrix=pbc_matrix)

# Verify round-trip
pdb2 = PDBParser(out_path)
assert len(pdb2.positions) == len(pos)
assert np.allclose(pdb2.positions, pos, atol=1e-3)
print(f"  Saved + verified: {len(pos)} atoms → {out_path}")
print(f"  Box preserved: {pdb2.pbc_matrix[0,0]:.1f} A")

# =====================================================
# Phase 2: Four-minimizer comparison (for reference)
# =====================================================
print(f"\n{'='*70}")
print(f"Phase 2: Minimizer comparison ({COMPARE_STEPS} steps each)")
print(f"{'='*70}")


def run_minimizer(name, minimizer, system):
    system.state.set_positions(pdb.positions)
    system.update_neighbor_list(force_rebuild=True)
    system.compute_forces(compute_energy=True)
    e_init = float(cp.asnumpy(system.state.d_energy[0]))
    mf_init = minimizer.compute_max_force(system)

    t0 = time.perf_counter()
    for _ in range(COMPARE_STEPS):
        system.update_neighbor_list(sync_interval=10)
        system.compute_forces(compute_energy=False)
        minimizer.step(system)

    elapsed = time.perf_counter() - t0
    system.compute_forces(compute_energy=True)
    e_final = float(cp.asnumpy(system.state.d_energy[0]))
    mf_final = minimizer.compute_max_force(system)
    return e_init, e_final, mf_init, mf_final, elapsed


minimizers = [
    ("SteepestDescent", SteepestDescentMinimizer(step_size=STEP_SIZE)),
    ("ConjugateGradient", ConjugateGradientMinimizer(step_size=0.001)),
    ("LBFGS", LBFGSMinimizer(step_size=STEP_SIZE)),
    ("FIRE", FIREMinimizer(time_step=STEP_SIZE, n_min=5)),
]

results = []
for name, minim in minimizers:
    print(f"  Running {name}...", end=" ", flush=True)
    r = run_minimizer(name, minim, system)
    results.append((name, *r))
    print(f"dE={(r[1] - r[0]) * _TO_KJMOL:+.0f} kJ/mol, maxF: {r[2]*_TO_KJMOLA:.1f}->{r[3]*_TO_KJMOLA:.2f}, {r[4]:.2f}s")

print(f"\n  {'Minimizer':<20s}  {'E_final(kJ/mol)':>16s}  {'dE(kJ/mol)':>12s}  {'maxF_final':>10s}  {'time':>7s}")
print(f"  {'-'*20:<20s}  {'-'*16:<16s}  {'-'*12:<12s}  {'-'*10:<10s}  {'-'*7:<7s}")
for name, e_init, e_final, mf_init, mf_final, elapsed in results:
    print(
        f"  {name:<20s}  {e_final * _TO_KJMOL:16.1f}  "
        f"{(e_final - e_init) * _TO_KJMOL:+12.1f}  "
        f"{mf_final * _TO_KJMOLA:10.3f}  {elapsed:6.2f}s"
    )

best_e = min(results, key=lambda r: r[2])
best_f = min(results, key=lambda r: r[4])
print(f"\n  Lowest energy: {best_e[0]} ({best_e[2] * _TO_KJMOL:.0f} kJ/mol)")
print(f"  Lowest maxF:   {best_f[0]} ({best_f[4] * _TO_KJMOLA:.3f} kJ/(mol·A))")
