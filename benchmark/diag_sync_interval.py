"""Diagnose: does sync_interval drive step time on the ion system?
Isolates the SYNC cost from the rebuild count. Read-only: no engine changes."""
import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import cupy as cp

from mdpy.io.psf_parser import PSFParser
from mdpy.io.pdb_parser import PDBParser
from mdpy.io.charmm_toppar_parser import CharmmTopparParser, create_parameter_table
from mdpy.force.factories.charmm import create_charmm_forces
from mdpy.integrator.langevin import LangevinBAOABIntegrator
from mdpy.system import System
from mdpy.utils import generate_velocity_from_temperature
import mdpy.system as _sysmod

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BOX = np.array([75.450, 77.623, 69.668])
CUTOFF = 12.0
DT_FS = 2
STEPS = 400


def build():
    psf = PSFParser(os.path.join(DATA_DIR, "ion.psf"))
    pdb = PDBParser(os.path.join(DATA_DIR, "ion.pdb"))
    toppar = CharmmTopparParser(
        os.path.join(DATA_DIR, "par_sin.prm"),
        os.path.join(DATA_DIR, "par_water.prm"),
    )
    topology = psf.topology
    pt = create_parameter_table(topology, toppar)
    pbc = np.diag(BOX)
    forces = create_charmm_forces(topology, pt, pbc, cutoff=CUTOFF)
    s = System(topology)
    s.upload_pbc(pbc)
    s.add_force_term(forces["bonded"])
    s.add_force_term(forces["nonbonded"])
    s.add_force_term(forces["pme"], stream="pme")
    s.upload_positions(pdb.positions)
    s.upload_velocities(generate_velocity_from_temperature(300.0, topology.masses, seed=42))
    return s, LangevinBAOABIntegrator(DT_FS, 300.0, 1.0)


def run(s, integ, n, sync):
    for _ in range(n):
        s.update_neighbor_list(sync_interval=sync)
        s.compute_forces()
        integ.step(s)


def timeit(s, integ, n, sync):
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    run(s, integ, n, sync)
    cp.cuda.Stream.null.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0


_counts = {}
_orig = _sysmod.System._do_rebuild


def _counting(self, pos):
    _counts[id(self)] = _counts.get(id(self), 0) + 1
    _orig(self, pos)


_sysmod.System._do_rebuild = _counting

print(f"{'sync_interval':>14} {'ms/step':>10} {'ns/day':>10} {'rebuilds':>10}")
for sync in [1, 2, 5, 10, 20, 50, 100]:
    s, integ = build()
    _counts[id(s)] = 0
    run(s, integ, 50, 10)
    _counts[id(s)] = 0
    s.update_neighbor_list(force_rebuild=True)
    _counts[id(s)] = 0
    ms = timeit(s, integ, STEPS, sync)
    ns = 86400.0 / (ms * 1e-3) * DT_FS * 1e-6
    print(f"{sync:>14d} {ms:>10.3f} {ns:>10.1f} {_counts[id(s)]:>10d}")
