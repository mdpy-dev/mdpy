"""Generate OpenMM reference forces and energies for mdpy validation.

Uses OpenMM Reference platform for deterministic floating-point results.
Compares against mdpy's internal unit system (angstrom / dalton / fs / e / K).
"""

import os
import sys
import numpy as np

import openmm as mm
import openmm.app as app
from openmm import unit as omm_unit

from mdpy.unit import (
    Quantity, default_energy_unit, default_force_unit, default_length_unit,
    kilojoule_permol, kilojoule_permol_over_nanometer, nanometer,
)

ENERGY_CONV = Quantity(1.0, kilojoule_permol).convert_to(default_energy_unit).value
FORCE_CONV = Quantity(1.0, kilojoule_permol_over_nanometer).convert_to(default_force_unit).value
POSITION_CONV = Quantity(1.0, nanometer).convert_to(default_length_unit).value

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def generate_reference(psf_path, pdb_path, prm_path, output_path, cutoff_ang=12.0):
    psf = app.CharmmPsfFile(psf_path)
    pdb = app.PDBFile(pdb_path)
    params = app.CharmmParameterSet(prm_path)

    system = psf.createSystem(
        params,
        nonbondedMethod=app.NoCutoff,
    )

    for force in system.getForces():
        if force.__class__.__name__ == 'NonbondedForce':
            for i in range(force.getNumExceptions()):
                p1, p2, _, _, _ = force.getExceptionParameters(i)
                force.setExceptionParameters(
                    i, p1, p2,
                    0.0 * omm_unit.elementary_charge**2,
                    0.1 * omm_unit.nanometer,
                    0.0 * omm_unit.kilojoule_per_mole,
                )

    force_groups = {}
    for force in system.getForces():
        name = force.__class__.__name__
        if name not in force_groups:
            group = len(force_groups)
            force_groups[name] = group
        force.setForceGroup(force_groups[name])

    platform = mm.Platform.getPlatformByName('Reference')
    integrator = mm.VerletIntegrator(0.001 * omm_unit.picosecond)
    simulation = app.Simulation(psf.topology, system, integrator, platform=platform)
    simulation.context.setPositions(pdb.getPositions())

    state = simulation.context.getState(getPositions=True, getForces=True, getEnergy=True)

    positions = np.array(
        state.getPositions().value_in_unit(omm_unit.nanometer),
        dtype=np.float64,
    ) * POSITION_CONV

    forces = np.array(
        state.getForces().value_in_unit(omm_unit.kilojoule_per_mole / omm_unit.nanometer),
        dtype=np.float64,
    ) * FORCE_CONV

    total_energy = state.getPotentialEnergy().value_in_unit(omm_unit.kilojoule_per_mole) * ENERGY_CONV

    per_term = {}
    for name, group in force_groups.items():
        state_g = simulation.context.getState(getEnergy=True, groups={group})
        energy_kjmol = state_g.getPotentialEnergy().value_in_unit(omm_unit.kilojoule_per_mole)
        per_term[name] = energy_kjmol * ENERGY_CONV

    bonded_force_names = {
        'HarmonicBondForce', 'HarmonicAngleForce',
        'PeriodicTorsionForce', 'CustomTorsionForce',
    }
    nonbonded_force_names = {'NonbondedForce', 'CustomNonbondedForce'}

    ref_bonded = sum(e for n, e in per_term.items() if n in bonded_force_names)
    ref_nonbonded = sum(e for n, e in per_term.items() if n in nonbonded_force_names)
    ref_mdpy_total = ref_bonded + ref_nonbonded

    np.savez(
        output_path,
        positions=positions.astype(np.float64),
        forces=forces.astype(np.float64),
        total_energy=np.float64(total_energy),
        force_names=np.array(list(per_term.keys())),
        force_energies=np.array(list(per_term.values()), dtype=np.float64),
        ref_bonded_energy=np.float64(ref_bonded),
        ref_nonbonded_energy=np.float64(ref_nonbonded),
        ref_mdpy_total_energy=np.float64(ref_mdpy_total),
    )
    print(f"Saved: {output_path}")
    print(f"  Atoms: {len(positions)}")
    print(f"  Total energy (mdpy units): {total_energy:.8f}")
    print(f"  mdpy-comparable total: {ref_mdpy_total:.8f}")
    print(f"  Conversion factors: E={ENERGY_CONV:.6e}, F={FORCE_CONV:.6e}, L={POSITION_CONV:.1f}")
    for name, energy in per_term.items():
        print(f"  {name}: {energy:.8f}")


if __name__ == '__main__':
    generate_reference(
        os.path.join(DATA_DIR, '6PO6.psf'),
        os.path.join(DATA_DIR, '6PO6.pdb'),
        os.path.join(DATA_DIR, 'par_all36_prot.prm'),
        os.path.join(DATA_DIR, 'openmm_reference_6PO6.npz'),
    )
