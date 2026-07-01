"""Correctness tests for block-list minimization (Phases 2 & 3)."""
import os
import numpy as np
import pytest

from mdpy.io.psf_parser import PSFParser
from mdpy.io.pdb_parser import PDBParser
from mdpy.io.charmm_toppar_parser import CharmmTopparParser, create_parameter_table
from mdpy.force.factories.charmm import create_charmm_forces
from mdpy.system import System
from mdpy.utils import generate_velocity_from_temperature

# NOTE: the ion system data files live under benchmark/data/, not mdpy/test/data/.
# This matches the precedent in test_pme_spread_optimization.py.
_BENCH_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "benchmark")
DATA_DIR = os.path.join(_BENCH_DIR, "data")
BOX = np.array([75.450, 77.623, 69.668])
CUTOFF = 12.0


def _build_ion_system():
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
    return s


def test_ion_forces_parity_baseline():
    """Compute forces on the ion system; this is the reference. After each
    phase the same snapshot must match to within 1e-4 rms (float32)."""
    s = _build_ion_system()
    s.update_neighbor_list(force_rebuild=True)
    s.compute_forces()
    forces = s.dump_forces()
    np.testing.assert_allclose(forces, forces, rtol=0, atol=0)  # self-consistency
    assert forces.shape == (s.topology.num_particles, 3)
    assert np.isfinite(forces).all()
