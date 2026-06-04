"""Profile block_pair_kernel memory access with ncu."""
import os
import sys
import subprocess
import json

NCU = "/home/ubuntu/Programs/cuda/13.0/bin/ncu"
WORKLOAD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile_block_pair_ncu_workload.py")

workload_code = r"""
import os, cupy as cp
import numpy as np
from mdpy.forcefield.charmm_forcefield import CharmmForcefield
from mdpy.force.bonded_force import BondedForce
from mdpy.force.nonbonded_force import NonbondedForce
from mdpy.force.expressions.lennard_jones import lennard_jones
from mdpy.force.expressions.coulomb import coulomb
from mdpy.integrator.langevin import LangevinBAOABIntegrator
from mdpy.system import System

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mdpy', 'test', 'data')
PSF_PATH = os.path.join(DATA_DIR, '1M9Z.psf')
PDB_PATH = os.path.join(DATA_DIR, '1M9Z_minimized.pdb')
PRM_PATH = os.path.join(DATA_DIR, 'par_all36_prot.prm')
STR_PATH = os.path.join(DATA_DIR, 'toppar_water_ions.str')

ff = CharmmForcefield(PSF_PATH, PDB_PATH, [PRM_PATH, STR_PATH])
topology = ff.create_topology()
parameter_table = ff.create_parameter_table()

pbc_matrix = np.eye(3, dtype=np.float64) * 108.0
system = System(topology, pbc_matrix, cutoff=12.0)
system.add_force_term(BondedForce(topology))
nb = NonbondedForce(lennard_jones + coulomb)
nb.bind(topology, parameter_table, 12.0)
system.add_force_term(nb)

raw = ff._pdb.positions.astype(np.float64)
pbc_inv = np.linalg.inv(pbc_matrix)
frac = raw @ pbc_inv
frac -= np.floor(frac)
wrapped = frac @ pbc_matrix
system.particles.positions[:] = wrapped
system.particles.velocities[:] = 0.0
system.gpu.upload_positions(system.particles)
system.gpu.upload_velocities(system.particles)

integrator = LangevinBAOABIntegrator(2.0, 300.0, 1.0)

for _ in range(20):
    system.step(integrator)

print("WARMUP_DONE")
for _ in range(3):
    system.step(integrator)
print("PROFILE_DONE")
"""

with open(WORKLOAD, 'w') as f:
    f.write(workload_code)

print("=== ncu Memory Profile: block_pair_kernel ===")
print()

metrics = [
    "gpu__time_duration.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
    "launch__shared_memory_per_block",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "memory_l1_wavefronts_shared",
    "memory_l1_hit_rate",
    "l1tex__avg_t_sectors_pipeline_lsu_miss_pct",
    "smsp__sass_inst_executed_op_global_ld.sum",
    "smsp__sass_inst_executed_op_global_st.sum",
    "smsp__sass_inst_executed_op_shared_ld.sum",
    "smsp__sass_inst_executed_op_shared_st.sum",
    "smsp__sass_inst_executed_op_atom_dot_global.sum",
    "smsp__sass_inst_executed_op_atom_dot_shared.sum",
    "sm__sass_inst_executed.sum",
    "lts__t_sectors_op_read.sum",
    "lts__t_sectors_op_write.sum",
    "lts__t_sectors_srcunit_tex.sum",
    "lts__t_sectors_srcunit_lts.sum",
]

metrics_str = ",".join(metrics)

cmd = [
    NCU,
    "--set", "full",
    "--launch-skip", "20",
    "--launch-count", "3",
    "-k", "regex:block_pair_kernel",
    "--csv",
    "-o", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ncu_block_pair_memory"),
    "--force-overwrite",
    "conda", "run", "-n", "md_analysis",
    "python", WORKLOAD,
]

print(f"Command: {' '.join(cmd)}")
print()

result = subprocess.run(
    cmd,
    capture_output=True,
    text=True,
    env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
    timeout=600,
)

print("STDOUT:", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
print("Return code:", result.returncode)
