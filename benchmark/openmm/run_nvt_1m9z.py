"""OpenMM NVT simulation of 1M9Z (95567 atoms) — 10 ns production run.

Used to validate that OpenMM can run this system stably, providing a
ground-truth trajectory for comparison with mdpy.

Configuration:
  - Ensemble: NVT (Langevin Middle integrator, 300 K, 1/ps friction)
  - Electrostatics: PME, 12 A cutoff
  - Constraints: HBonds (allows 2 fs timestep)
  - Timestep: 2 fs
  - Duration: 10 ns = 5,000,000 steps
  - Output: DCD trajectory + energy log + final state

Usage:
  conda run -n md_analysis python benchmark/openmm/run_nvt_1m9z.py

Output files (in benchmark/openmm/output/):
  - 1m9z_nvt.dcd         : trajectory (5000 frames, every 1000 steps = 2 ps)
  - 1m9z_nvt_energy.csv  : step, potential_energy_kJ, kinetic_energy_kJ, temperature_K
  - 1m9z_nvt_final.xml   : final serialized state (for mdpy comparison)
"""

import os
import sys
import time

import numpy as np

import openmm as mm
import openmm.app as app
from openmm import unit as u, Vec3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from benchmark._data_path import DATA_DIR
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

PSF_PATH = os.path.join(DATA_DIR, "1M9Z.psf")
PDB_PATH = os.path.join(DATA_DIR, "1M9Z.pdb")
PRM_PATH = os.path.join(DATA_DIR, "par_all36_prot.prm")
STR_PATH = os.path.join(DATA_DIR, "toppar_water_ions.str")

BOX_NM = 10.8
CUTOFF_NM = 1.2
TEMPERATURE_K = 300.0
FRICTION_PS = 1.0
DT_FS = 2.0
TOTAL_NS = 10.0
MINIMIZE_STEPS = 5000
WARMUP_PS = 100.0

STEPS_PER_NS = int(1e6 / DT_FS)
TOTAL_STEPS = int(TOTAL_NS * STEPS_PER_NS)
WARMUP_STEPS = int(WARMUP_PS * 1000 / DT_FS)
DCD_INTERVAL = 1000
ENERGY_INTERVAL = 500


def create_system():
    psf = app.CharmmPsfFile(PSF_PATH)
    params = app.CharmmParameterSet(PRM_PATH, STR_PATH)

    psf.box_vectors = [
        Vec3(BOX_NM, 0, 0),
        Vec3(0, BOX_NM, 0),
        Vec3(0, 0, BOX_NM),
    ] * u.nanometer

    system = psf.createSystem(
        params,
        nonbondedMethod=app.PME,
        nonbondedCutoff=CUTOFF_NM * u.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
    )

    return psf, system


def center_positions(pdb):
    positions_nm = np.array(
        [list(p) for p in pdb.getPositions().value_in_unit(u.nanometer)]
    )
    center = (positions_nm.max(axis=0) + positions_nm.min(axis=0)) / 2.0
    box_center = np.array([BOX_NM / 2.0] * 3)
    positions_nm += box_center - center
    return [Vec3(r[0], r[1], r[2]) for r in positions_nm] * u.nanometer


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  OpenMM NVT Simulation — 1M9Z")
    print("=" * 70)
    print(f"  Atoms:         95567")
    print(f"  Box:           {BOX_NM} nm cuboid")
    print(f"  Cutoff:        {CUTOFF_NM} nm, PME")
    print(f"  Temperature:   {TEMPERATURE_K} K (Langevin, friction={FRICTION_PS}/ps)")
    print(f"  Timestep:      {DT_FS} fs")
    print(f"  Total:         {TOTAL_NS} ns ({TOTAL_STEPS} steps)")
    print(f"  Output:        {OUTPUT_DIR}")
    print("=" * 70)

    pdb = app.PDBFile(PDB_PATH)
    psf, system = create_system()
    integrator = mm.LangevinMiddleIntegrator(
        TEMPERATURE_K * u.kelvin,
        FRICTION_PS / u.picosecond,
        DT_FS * u.femtoseconds,
    )
    platform = mm.Platform.getPlatformByName("CUDA")
    properties = {"DeviceIndex": "0", "Precision": "single"}
    simulation = app.Simulation(
        psf.topology,
        system,
        integrator,
        platform=platform,
        platformProperties=properties,
    )

    ctx_platform = simulation.context.getPlatform().getName()
    ctx_device = simulation.context.getPlatform().getPropertyValue(
        simulation.context, "DeviceIndex"
    )
    ctx_prec = simulation.context.getPlatform().getPropertyValue(
        simulation.context, "Precision"
    )
    print(f"  Platform:      {ctx_platform}, Device {ctx_device}, {ctx_prec} precision")

    centered_pos = center_positions(pdb)
    simulation.context.setPositions(centered_pos)
    simulation.context.setVelocitiesToTemperature(0 * u.kelvin)

    # --- Minimization ---
    print(f"\n[1/3] Energy minimization ({MINIMIZE_STEPS} steps)...", flush=True)
    t0 = time.perf_counter()
    simulation.minimizeEnergy(maxIterations=MINIMIZE_STEPS)
    t_min = time.perf_counter() - t0
    state = simulation.context.getState(getEnergy=True)
    e_min = state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    print(f"      Done in {t_min:.1f}s, E = {e_min:.1f} kJ/mol", flush=True)

    # --- Warmup ---
    simulation.context.setVelocitiesToTemperature(TEMPERATURE_K * u.kelvin)
    print(f"\n[2/3] Warmup ({WARMUP_PS} ps = {WARMUP_STEPS} steps)...", flush=True)
    t0 = time.perf_counter()
    simulation.step(WARMUP_STEPS)
    t_warm = time.perf_counter() - t0
    warm_ms = t_warm / WARMUP_STEPS * 1000
    print(f"      Done in {t_warm:.1f}s, {warm_ms:.2f} ms/step", flush=True)

    # --- Production ---
    dcd_path = os.path.join(OUTPUT_DIR, "1m9z_nvt.dcd")
    csv_path = os.path.join(OUTPUT_DIR, "1m9z_nvt_energy.csv")
    xml_path = os.path.join(OUTPUT_DIR, "1m9z_nvt_final.xml")

    simulation.reporters.append(app.DCDReporter(dcd_path, DCD_INTERVAL))
    simulation.reporters.append(
        app.StateDataReporter(
            csv_path,
            ENERGY_INTERVAL,
            step=True,
            potentialEnergy=True,
            kineticEnergy=True,
            temperature=True,
            speed=True,
        )
    )

    print(f"\n[3/3] Production ({TOTAL_NS} ns = {TOTAL_STEPS} steps)...", flush=True)
    print(
        f"      DCD: every {DCD_INTERVAL} steps ({DCD_INTERVAL * DT_FS / 1000:.1f} ps)"
    )
    print(f"      Energy: every {ENERGY_INTERVAL} steps")
    print(flush=True)

    t0 = time.perf_counter()
    CHECKPOINT_NS = 1.0
    checkpoint_steps = int(CHECKPOINT_NS * STEPS_PER_NS)
    next_checkpoint = checkpoint_steps

    for step_block in range(0, TOTAL_STEPS, checkpoint_steps):
        remaining = min(checkpoint_steps, TOTAL_STEPS - step_block)
        simulation.step(remaining)
        elapsed = time.perf_counter() - t0
        done = step_block + remaining
        frac = done / TOTAL_STEPS
        ms_per_step = elapsed / done * 1000
        ns_done = done * DT_FS / 1e6
        eta_s = (TOTAL_STEPS - done) * ms_per_step / 1000
        eta_h = eta_s / 3600
        print(
            f"  {ns_done:6.1f} / {TOTAL_NS} ns  "
            f"({frac*100:5.1f}%)  "
            f"{ms_per_step:.2f} ms/step  "
            f"ETA {eta_h:.1f}h",
            flush=True,
        )

    elapsed_total = time.perf_counter() - t0
    ms_per_step = elapsed_total / TOTAL_STEPS * 1000
    ns_day = 86400.0 / (ms_per_step * 1e-3) * DT_FS * 1e-6

    state = simulation.context.getState(getEnergy=True)
    e_final = state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)

    with open(xml_path, "w") as f:
        f.write(mm.XmlSerializer.serialize(state))

    print(f"\n{'=' * 70}")
    print(f"  Done!")
    print(f"  Wall time:   {elapsed_total:.0f}s ({elapsed_total/3600:.1f}h)")
    print(f"  Performance: {ms_per_step:.2f} ms/step, {ns_day:.1f} ns/day")
    print(f"  Final E_pot: {e_final:.1f} kJ/mol")
    print(f"  DCD:         {dcd_path}")
    print(f"  Energy log:  {csv_path}")
    print(f"  Final state: {xml_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run()
