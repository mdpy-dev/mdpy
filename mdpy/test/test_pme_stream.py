from __future__ import annotations

import os

import cupy as cp
import numpy as np

from mdpy.io.psf_parser import PSFParser
from mdpy.io.pdb_parser import PDBParser
from mdpy.io.charmm_toppar_parser import CharmmTopparParser
from mdpy.io.charmm_toppar_parser import create_parameter_table
from mdpy.force.factories.charmm import create_charmm_forces
from mdpy.integrator.verlet import VerletIntegrator
from mdpy.system import System
from mdpy import env

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
PSF = os.path.join(DATA_DIR, '6PO6.psf')
PDB = os.path.join(DATA_DIR, '6PO6.pdb')
PRM = os.path.join(DATA_DIR, 'par_all36_prot.prm')
STR = os.path.join(DATA_DIR, 'toppar_water_ions.str')

BOX = 30.0


def _pbc_matrix():
    return np.eye(3, dtype=np.float64) * BOX


def _wrapped_positions(pdb, pbc):
    raw = pdb.positions.astype(np.float64)
    pbc_inv = np.linalg.inv(pbc)
    frac = raw @ pbc_inv
    frac -= np.floor(frac)
    return (frac @ pbc).astype(env.NUMPY_FLOAT)


def _build_system(pme_stream):
    """Build a System with bonded + nonbonded + PME.

    pme_stream: if True, PME term runs on the dedicated pme stream;
                if False, PME term runs on the primary (null) stream.
    """
    psf = PSFParser(PSF)
    pdb = PDBParser(PDB)
    toppar = CharmmTopparParser(PRM, STR)
    topology = psf.topology
    topology.join()
    pt = create_parameter_table(topology, toppar)
    pbc = _pbc_matrix()

    forces = create_charmm_forces(topology, pt, pbc, cutoff=12.0)

    system = System(topology)
    system.upload_pbc(pbc)
    system.add_force_term(forces['bonded'])
    system.add_force_term(forces['nonbonded'])
    system.add_force_term(forces['pme'], stream='pme' if pme_stream else None)

    pos = _wrapped_positions(pdb, pbc)
    system.upload_positions(pos)
    system.upload_velocities(np.zeros((topology.num_particles, 3), dtype=env.NUMPY_FLOAT))
    return system, VerletIntegrator(0.5)


class TestPmeStreamRouting:

    def test_pme_term_routed_to_pme_stream(self):
        system, _ = _build_system(pme_stream=True)
        assert system._pme_stream is not None, "pme stream must be created"
        assert system._pme_stream is not cp.cuda.Stream.null
        assert len(system._pme_force_terms) == 1
        assert len(system._primary_force_terms) == 2
        # all terms still appear in the combined force_terms list
        assert len(system.force_terms) == 3

    def test_default_routing_is_primary(self):
        system, _ = _build_system(pme_stream=False)
        assert system._pme_stream is None
        assert len(system._pme_force_terms) == 0
        assert len(system._primary_force_terms) == 3


class TestPmeStreamCorrectness:

    def test_forces_match_primary_vs_pme_stream(self):
        # Single compute_forces, no integration, so no compounding divergence.
        system_primary, _ = _build_system(pme_stream=False)
        system_pme, _ = _build_system(pme_stream=True)

        system_primary.update_neighbor_list(force_rebuild=True)
        system_primary.compute_forces()
        f_primary = system_primary.dump_forces()

        system_pme.update_neighbor_list(force_rebuild=True)
        system_pme.compute_forces()
        f_pme = system_pme.dump_forces()

        # atomicAdd arrival order differs between stream configs -> float32
        # noise of ~1e-5 per component is expected and harmless.
        np.testing.assert_allclose(f_pme, f_primary, atol=1e-4, rtol=1e-4,
                                   err_msg="PME-on-separate-stream forces diverge from primary")

    def test_energies_match_primary_vs_pme_stream(self):
        # dump_energy() recomputes energy sequentially on the null stream
        # (it does not use compute_forces's two-stream path). This is a smoke
        # check that routing the PME term does not corrupt its term state,
        # NOT a two-stream energy-correctness guard.
        system_primary, _ = _build_system(pme_stream=False)
        system_pme, _ = _build_system(pme_stream=True)

        system_primary.update_neighbor_list(force_rebuild=True)
        system_primary.compute_forces()
        e_primary = system_primary.dump_energy()

        system_pme.update_neighbor_list(force_rebuild=True)
        system_pme.compute_forces()
        e_pme = system_pme.dump_energy()

        for key in e_primary:
            assert key in e_pme, f"missing energy term {key!r} in pme-stream result"
            rel = abs(e_pme[key] - e_primary[key]) / max(abs(e_primary[key]), 1e-12)
            assert rel < 1e-4, f"energy {key!r}: primary={e_primary[key]}, pme={e_pme[key]}, rel={rel}"


class TestPmeStreamMultiStep:

    def test_20_steps_remain_finite(self):
        system, integrator = _build_system(pme_stream=True)
        for _ in range(20):
            system.update_neighbor_list(sync_interval=10)
            system.compute_forces()
            integrator.step(system)
        pos, vel = system.dump_state()
        assert np.all(np.isfinite(pos)), "positions contain NaN/Inf after 20 steps"
        assert np.all(np.isfinite(vel)), "velocities contain NaN/Inf after 20 steps"
        e = system.dump_energy()
        assert all(np.isfinite(v) for v in e.values()), "energy contains NaN/Inf"
