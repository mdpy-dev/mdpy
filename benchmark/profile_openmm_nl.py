"""OpenMM neighbor list profiling workload for nsys/ncu.

Usage:
    CUDA_VISIBLE_DEVICES=0 /home/ubuntu/Programs/cuda/13.0/bin/nsys profile \
      --trace=cuda,nvtx,osrt --output=openmm_nl \
      conda run -n md_analysis python benchmark/profile_openmm_nl.py

    CUDA_VISIBLE_DEVICES=0 /home/ubuntu/Programs/cuda/13.0/bin/ncu \
      --set full --launch-skip 15 --launch-count 1 \
      -k "regex:findBlocksWithInteractions" \
      -o openmm_find_tiles_ncu \
      conda run -n md_analysis python benchmark/profile_openmm_nl.py
"""
import os
import openmm as mm
import openmm.app as app
from openmm import unit

from benchmark._data_path import DATA_DIR
BOX_SIZE = 10.8
CUTOFF = 1.2


def main():
    psf = app.CharmmPsfFile(os.path.join(DATA_DIR, '1M9Z.psf'))
    pdb = app.PDBFile(os.path.join(DATA_DIR, '1M9Z_minimized.pdb'))
    params = app.CharmmParameterSet(
        os.path.join(DATA_DIR, 'par_all36_prot.prm'),
        os.path.join(DATA_DIR, 'toppar_water_ions.str')
    )
    psf.setBox(BOX_SIZE, BOX_SIZE, BOX_SIZE)

    system = psf.createSystem(
        params,
        nonbondedMethod=app.CutoffNonPeriodic,
        nonbondedCutoff=CUTOFF * unit.nanometer,
    )
    integrator = mm.VerletIntegrator(0.002 * unit.picoseconds)

    platform = mm.Platform.getPlatformByName('CUDA')
    properties = {'Precision': 'single'}

    sim = app.Simulation(psf.topology, system, integrator, platform, properties)
    sim.context.setPositions(pdb.getPositions())
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin)

    print(f"Atoms: {psf.topology.getNumAtoms()}")

    for _ in range(10):
        sim.step(1)

    for _ in range(30):
        sim.step(1)

    print("Done.")


if __name__ == '__main__':
    main()
