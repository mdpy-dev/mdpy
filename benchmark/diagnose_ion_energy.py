"""Single-point energy breakdown diagnostic for the ion system.

Computes forces ONCE at the initial (wrapped) configuration -- no integration --
and prints the per-term energy breakdown (kcal/mol), with and without the
position restraint, to determine whether the restraint dominates the total.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n md_analysis python benchmark/diagnose_ion_energy.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from mdpy.io.psf_parser import PSFParser
from mdpy.io.pdb_parser import PDBParser
from mdpy.io.charmm_toppar_parser import CharmmTopparParser
from mdpy.io.charmm_toppar_parser import create_parameter_table
from mdpy.force.factories.charmm import create_charmm_forces
from mdpy.force.bonded_force import BondedForce
from mdpy.force.expressions.position_restraint import position_restraint
from mdpy.system import System
from mdpy.constraint.constraint_scheme import create_constraints

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

BOX = np.array([75.450, 77.623, 69.668], dtype=np.float64)
CUTOFF = 12.0
KCAL_PER_INTERNAL = 1.0 / 4.1840286576e-4


def _load_inputs():
    psf = PSFParser(os.path.join(DATA_DIR, "ion.psf"))
    pdb = PDBParser(os.path.join(DATA_DIR, "ion.pdb"))
    toppar = CharmmTopparParser(
        os.path.join(DATA_DIR, "par_sin.prm"),
        os.path.join(DATA_DIR, "par_water.prm"),
    )
    topology = psf.topology
    parameter_table = create_parameter_table(topology, toppar)
    return topology, parameter_table, pdb


def _build_system(include_restraint):
    topology, parameter_table, pdb = _load_inputs()
    pbc_matrix = np.diag(BOX)
    forces = create_charmm_forces(topology, parameter_table, pbc_matrix, cutoff=CUTOFF)

    system = System(topology)
    system.upload_pbc(pbc_matrix)
    system.add_force_term(forces["bonded"])
    system.add_force_term(forces["nonbonded"])
    system.add_force_term(forces["pme"])

    if include_restraint:
        restraint = BondedForce(position_restraint)
        positions = pdb.positions
        sin_indices = [
            i for i, resname in enumerate(topology.molecule_types) if resname == "SIN"
        ]
        for i in sin_indices:
            restraint.add(
                [i], k=0.02,
                ref_x=float(positions[i][0]),
                ref_y=float(positions[i][1]),
                ref_z=float(positions[i][2]),
            )
        restraint.name = 'position_restraint'
        system.add_force_term(restraint)

    constraints = create_constraints(topology, parameter_table, scheme="h-bonds")
    for c in constraints:
        system.add_constraint(c)

    n = topology.num_particles
    system.upload_positions(pdb.positions)
    system.upload_velocities(np.zeros((n, 3), dtype=np.float32))
    system.update_neighbor_list(force_rebuild=True)
    return system


def _report(system, label):
    energies = system.dump_energy()
    forces = system.dump_forces()
    total = sum(energies.values())

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  {'term':>22s}  {'internal':>16s}  {'kcal/mol':>16s}")
    for name, val in energies.items():
        print(f"  {name:>22s}  {val:16.6e}  {val*KCAL_PER_INTERNAL:16.2f}")
    print(f"  {'-'*22}  {'-'*16}  {'-'*16}")
    print(f"  {'TOTAL':>22s}  {total:16.6e}  {total*KCAL_PER_INTERNAL:16.2f}")

    fmag = np.sqrt((forces ** 2).sum(axis=1))
    print(f"\n  per-particle |force| (internal units):")
    print(f"    min={fmag.min():.4e}  max={fmag.max():.4e}  "
          f"mean={fmag.mean():.4e}  std={fmag.std():.4e}")
    print(f"    has NaN: {bool(np.isnan(forces).any())}  "
          f"has Inf: {bool(np.isinf(forces).any())}")
    return energies, total


def main():
    topology, _, _ = _load_inputs()
    print("mdpy ion system single-point energy diagnostic")
    print(f"  Atoms:   {topology.num_particles}")
    print(f"  Box:     {BOX[0]:.3f} x {BOX[1]:.3f} x {BOX[2]:.3f} A")
    print(f"  Cutoff:  {CUTOFF} A")
    print(f"  KCAL_PER_INTERNAL = {KCAL_PER_INTERNAL:.4f}")

    e_with, total_with = _report(_build_system(include_restraint=True),
                                 "A. WITH position restraint (k=0.02 internal)")
    e_without, total_without = _report(_build_system(include_restraint=False),
                                       "B. WITHOUT position restraint")

    restraint_kcal = e_with.get("position_restraint", 0.0) * KCAL_PER_INTERNAL
    print(f"\n{'='*70}")
    print(f"  COMPARISON")
    print(f"{'='*70}")
    print(f"  total WITH    restraint: {total_with*KCAL_PER_INTERNAL:16.2f} kcal/mol")
    print(f"  total WITHOUT restraint: {total_without*KCAL_PER_INTERNAL:16.2f} kcal/mol")
    print(f"  restraint contribution : {restraint_kcal:16.2f} kcal/mol")
    if abs(total_with) > 0:
        frac = abs(e_with.get("position_restraint", 0.0)) / abs(total_with) * 100
        print(f"  restraint / total(WITH): {frac:6.2f} %")

    print(f"\n  Note: bulk periodic water+ion systems naturally have absolute")
    print(f"  PE on the order of 1e5 kcal/mol. A large magnitude alone is not")
    print(f"  a bug -- it must be cross-checked against OpenMM (see Step 2).")


if __name__ == "__main__":
    main()
