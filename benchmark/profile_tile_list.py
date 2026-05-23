"""mdpy tile list rebuild profiling workload for nsys/ncu.

Usage:
    CUDA_VISIBLE_DEVICES=0 /home/ubuntu/Programs/cuda/13.0/bin/nsys profile \
      --trace=cuda,nvtx,osrt --output=mdpy_tilelist \
      conda run -n md_analysis python benchmark/profile_tile_list.py

    CUDA_VISIBLE_DEVICES=0 /home/ubuntu/Programs/cuda/13.0/bin/ncu \
      --set full --launch-skip 5 --launch-count 1 \
      -k "regex:find_interacting_blocks" \
      -o mdpy_find_tiles_ncu \
      conda run -n md_analysis python benchmark/profile_tile_list.py
"""
import os
import cupy as cp
import numpy as np
import nvtx

from mdpy.forcefield.charmm_forcefield import CharmmForcefield
from mdpy.force.bonded_force import BondedForce
from mdpy.force.nonbonded_force import NonbondedForce
from mdpy.force.expressions.lennard_jones import lennard_jones
from mdpy.force.expressions.coulomb import coulomb
from mdpy.integrator.verlet import VerletIntegrator
from mdpy.system import System

from benchmark._data_path import DATA_DIR
BOX_SIZE = 108.0
CUTOFF = 12.0
DT_FS = 0.5


def main():
    ff = CharmmForcefield(
        os.path.join(DATA_DIR, '1M9Z.psf'),
        os.path.join(DATA_DIR, '1M9Z_minimized.pdb'),
        [os.path.join(DATA_DIR, 'par_all36_prot.prm'),
         os.path.join(DATA_DIR, 'toppar_water_ions.str')]
    )
    topology = ff.create_topology()
    parameter_table = ff.create_parameter_table()

    pbc_matrix = np.eye(3, dtype=np.float64) * BOX_SIZE
    pbc_inv = np.linalg.inv(pbc_matrix)

    system = System(topology, pbc_matrix, cutoff=CUTOFF)
    system.add_force_term(BondedForce.charmm(topology, parameter_table))
    nb = NonbondedForce(lennard_jones + coulomb)
    nb.bind(topology, parameter_table, CUTOFF)
    system.add_force_term(nb)

    raw = ff._pdb.positions.astype(np.float64)
    frac = raw @ pbc_inv
    frac -= np.floor(frac)
    wrapped = frac @ pbc_matrix
    system.particles.positions[:] = wrapped
    system.particles.velocities[:] = 0.0
    system.gpu.upload_positions(system.particles)
    system.gpu.upload_velocities(system.particles)

    integrator = VerletIntegrator(DT_FS)
    print(f"Atoms: {topology.num_particles}")

    nvtx.push_range("warmup")
    system.step(integrator, 5)
    cp.cuda.Stream.null.synchronize()
    nvtx.pop_range()

    nvtx.push_range("profiled_region")
    system.step(integrator, 25)
    cp.cuda.Stream.null.synchronize()
    nvtx.pop_range()

    print(f"Tiles: {system.tile_list.num_tiles}")
    print(f"Blocks: {system.tile_list.num_blocks}")
    print("Done.")


if __name__ == '__main__':
    main()
