"""nsys profiling workload for mdpy 1M9Z.

Usage:
    CUDA_VISIBLE_DEVICES=0 /home/ubuntu/Programs/cuda/13.0/bin/nsys profile \
        --trace=cuda,nvtx --output=benchmark/mdpy_timeline \
        conda run -n md_analysis python benchmark/profile_1m9z_nsys.py
"""

import os
import nvtx
import cupy as cp
import numpy as np
from benchmark._data_path import DATA_DIR

PSF_PATH = os.path.join(DATA_DIR, "1M9Z.psf")
PDB_PATH = os.path.join(DATA_DIR, "1M9Z_minimized.pdb")
PRM_PATH = os.path.join(DATA_DIR, "par_all36_prot.prm")
STR_PATH = os.path.join(DATA_DIR, "toppar_water_ions.str")

BOX_SIZE = 108.0
CUTOFF = 12.0
DT_FS = 0.5


def main():
    from mdpy.io.psf_parser import PSFParser
    from mdpy.io.pdb_parser import PDBParser
    from mdpy.io.charmm_toppar_parser import CharmmTopparParser
    from mdpy.io.charmm_toppar_parser import create_parameter_table
    from mdpy.force.bonded_force import BondedForce
    from mdpy.force.nonbonded_force import NonbondedForce
    from mdpy.force.expressions.lennard_jones import lennard_jones
    from mdpy.force.expressions.coulomb import coulomb
    from mdpy.integrator.verlet import VerletIntegrator
    from mdpy.system import System

    psf = PSFParser(PSF_PATH)
    pdb = PDBParser(PDB_PATH)
    toppar = CharmmTopparParser(PRM_PATH, STR_PATH)
    topology = psf.topology
    parameter_table = create_parameter_table(topology, toppar)

    pbc_matrix = np.eye(3, dtype=np.float64) * BOX_SIZE
    pbc_inv = np.linalg.inv(pbc_matrix)

    system = System(topology, pbc_matrix, cutoff=CUTOFF)
    system.add_force_term(BondedForce.charmm(topology, parameter_table))
    nb = NonbondedForce(lennard_jones + coulomb)
    nb.bind(topology, parameter_table, CUTOFF)
    system.add_force_term(nb)

    raw = pdb.positions.astype(np.float64)
    frac = raw @ pbc_inv
    frac -= np.floor(frac)
    wrapped = frac @ pbc_matrix
    system.particles.positions[:] = wrapped
    system.particles.velocities[:] = 0.0
    system.gpu.upload_positions(system.particles)
    system.gpu.upload_velocities(system.particles)

    integrator = VerletIntegrator(DT_FS)

    system.gpu.refresh_wrapped_positions()
    positions_soa = (
        system.gpu.d_wrapped_positions_x,
        system.gpu.d_wrapped_positions_y,
        system.gpu.d_wrapped_positions_z,
    )
    system._do_full_rebuild(positions_soa)

    bonded = system.force_terms[0]
    nonbonded = system.force_terms[1]
    bl = system.block_list
    gpu = system.gpu

    nvtx.push_range("warmup")
    for _ in range(10):
        system._emit_step_kernels(integrator)
    cp.cuda.Stream.null.synchronize()
    nvtx.pop_range()

    nvtx.push_range("profiled")
    for i in range(30):
        nvtx.push_range(f"step_{i}")

        nvtx.push_range("pbc_wrap")
        gpu.refresh_wrapped_positions()
        nvtx.pop_range()

        nvtx.push_range("check_rebuild")
        bl.check_rebuild_async(positions_soa)
        nvtx.pop_range()

        nvtx.push_range("zero_forces")
        gpu.zero_forces()
        nvtx.pop_range()

        nvtx.push_range("bonded_force")
        bonded.compute(gpu, bl)
        nvtx.pop_range()

        nvtx.push_range("nonbonded_force")
        nonbonded.compute(gpu, bl)
        nvtx.pop_range()

        nvtx.push_range("integrator")
        integrator.step(gpu)
        nvtx.pop_range()

        nvtx.pop_range()
    cp.cuda.Stream.null.synchronize()
    nvtx.pop_range()
    print("Profiling complete.")


if __name__ == "__main__":
    main()
