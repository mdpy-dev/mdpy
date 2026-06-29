"""Isolated timing of BondedForce kernels on 1M9Z bonded terms.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n md_analysis python benchmark/time_bonded.py
"""
import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cupy as cp
import numpy as np

from mdpy.io.psf_parser import PSFParser
from mdpy.io.pdb_parser import PDBParser
from mdpy.io.charmm_toppar_parser import CharmmTopparParser
from mdpy.io.charmm_toppar_parser import create_parameter_table
from mdpy.force.factories.charmm import create_bonded_group
from mdpy.system import System

DATA = os.path.join(os.path.dirname(__file__), "..", "mdpy", "test", "data")
BOX = 108.0

psf = PSFParser(os.path.join(DATA, "1M9Z.psf"))
pdb = PDBParser(os.path.join(DATA, "1M9Z_minimized.pdb"))
tp = CharmmTopparParser(os.path.join(DATA, "par_all36_prot.prm"),
                        os.path.join(DATA, "toppar_water_ions.str"))
top = psf.topology
pt = create_parameter_table(top, tp)

system = System(top)
system.upload_pbc(np.eye(3, dtype=np.float64) * BOX)
bonded = create_bonded_group(top, pt)
system.add_force_term(bonded)

system.upload_positions(pdb.positions)
system.upload_velocities(np.zeros((top.num_particles, 3), dtype=np.float32))

# warmup
for _ in range(20):
    cp.cuda.Stream.null.synchronize()
    bonded.compute(system.gpu, None, compute_energy=True)
cp.cuda.Stream.null.synchronize()

# time
N = 200
cp.cuda.Stream.null.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    bonded.compute(system.gpu, None, compute_energy=True)
cp.cuda.Stream.null.synchronize()
elapsed = time.perf_counter() - t0
print(f"bonded compute: {elapsed/N*1000:.4f} ms/call over {N} calls")
