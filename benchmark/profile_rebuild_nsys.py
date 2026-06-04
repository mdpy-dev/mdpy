"""nsys profiling workload that captures rebuild cycles.

Usage:
    CUDA_VISIBLE_DEVICES=0 /home/ubuntu/Programs/cuda/13.0/bin/nsys profile \
        --trace=cuda,nvtx --output=benchmark/nsys/2026-06-03-013-1m9z-nograph \
        conda run -n md_analysis python benchmark/profile_rebuild_nsys.py
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
SKIN = 1.0


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

    rng = np.random.default_rng(42)
    velocities = rng.standard_normal((topology.num_particles, 3)).astype(np.float32) * 0.001

    system = System(topology, pbc_matrix, cutoff=CUTOFF, skin=SKIN,
                    rebuild_check_interval=10)
    system.add_force_term(BondedForce.charmm(topology, parameter_table))
    nb = NonbondedForce(lennard_jones + coulomb)
    nb.bind(topology, parameter_table, CUTOFF)
    system.add_force_term(nb)

    raw = pdb.positions.astype(np.float64)
    frac = raw @ pbc_inv
    frac -= np.floor(frac)
    wrapped = frac @ pbc_matrix
    system.particles.positions[:] = wrapped
    system.particles.velocities[:] = velocities
    system.gpu.upload_positions(system.particles)
    system.gpu.upload_velocities(system.particles)

    integrator = VerletIntegrator(DT_FS)

    system.upload_positions()
    system.upload_velocities()
    system.gpu.refresh_wrapped_positions()

    nvtx.push_range("warmup")
    for _ in range(50):
        system.update_neighbor_list(force_check=True)
        system.compute_forces()
        integrator.step(system)
        system.gpu.refresh_wrapped_positions()
    cp.cuda.Stream.null.synchronize()
    nvtx.pop_range()

    nvtx.push_range("profiled")
    for _ in range(250):
        system.update_neighbor_list(force_check=True)
        system.compute_forces()
        integrator.step(system)
        system.gpu.refresh_wrapped_positions()
    cp.cuda.Stream.null.synchronize()
    nvtx.pop_range()
    print("Profiling complete.")


if __name__ == "__main__":
    main()
