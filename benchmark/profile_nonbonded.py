"""Profile mdpy nonbonded force with NVTX markers for nsys analysis."""
import os
import time
import nvtx
import cupy as cp
import numpy as np

from benchmark._data_path import DATA_DIR
PSF_PATH = os.path.join(DATA_DIR, '1M9Z.psf')
PDB_PATH = os.path.join(DATA_DIR, '1M9Z_minimized.pdb')
PRM_PATH = os.path.join(DATA_DIR, 'par_all36_prot.prm')
STR_PATH = os.path.join(DATA_DIR, 'toppar_water_ions.str')

BOX_SIZE = 108.0
CUTOFF = 12.0
DT_FS = 0.5

from mdpy.forcefield.charmm_forcefield import CharmmForcefield
from mdpy.force.bonded_force import BondedForce
from mdpy.force.nonbonded_force import NonbondedForce
from mdpy.force.expressions.lennard_jones import lennard_jones
from mdpy.force.expressions.coulomb import coulomb
from mdpy.integrator.verlet import VerletIntegrator
from mdpy.system import System

ff = CharmmForcefield(PSF_PATH, PDB_PATH, [PRM_PATH, STR_PATH])
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

print(f"Warmup 500 steps...")
cp.cuda.Stream.null.synchronize()
t0 = time.perf_counter()
system.step(integrator, 500)
cp.cuda.Stream.null.synchronize()
print(f"  {time.perf_counter() - t0:.1f}s")

print(f"Profiled region: 200 steps with NVTX markers...")
cp.cuda.Stream.null.synchronize()

for i in range(200):
    nvtx.push_range("step")
    positions_soa = system.gpu.get_positions_2d()

    nvtx.push_range("check_rebuild")
    need_rebuild = system.tile_list.check_rebuild(positions_soa)
    nvtx.pop_range()

    if need_rebuild:
        nvtx.push_range("tile_list_rebuild")
        system.tile_list.rebuild(
            positions_soa, topology, system.pbc_matrix, system.pbc_inv
        )
        nvtx.pop_range()

    nvtx.push_range("compute_forces")
    system.gpu.zero_forces()
    system.gpu.set_box_dims(
        abs(float(system.pbc_matrix[0, 0])),
        abs(float(system.pbc_matrix[1, 1])),
        abs(float(system.pbc_matrix[2, 2]))
    )
    for term_index, term in enumerate(system.force_terms):
        nvtx.push_range(term.name)
        system.gpu.zero_energy()
        term.compute(system.gpu, system.tile_list)
        system.gpu.accumulate_energy(term_index)
        nvtx.pop_range()
    nvtx.pop_range()

    nvtx.push_range("integrator")
    integrator.step(system.gpu)
    nvtx.pop_range()

    nvtx.pop_range()

cp.cuda.Stream.null.synchronize()
print("Done.")
