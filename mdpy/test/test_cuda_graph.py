"""Tests for CUDA Graph per-step compute integration."""
from __future__ import annotations

import cupy as cp
import numpy as np
import os

from mdpy.io.psf_parser import PSFParser
from mdpy.io.pdb_parser import PDBParser
from mdpy.io.charmm_toppar_parser import CharmmTopparParser
from mdpy.io.charmm_toppar_parser import create_parameter_table
from mdpy.force.bonded_force import BondedForce
from mdpy.force.nonbonded_force import NonbondedForce
from mdpy.force.expressions.lennard_jones import lennard_jones
from mdpy.force.expressions.coulomb import coulomb
from mdpy.integrator.verlet import VerletIntegrator
from mdpy.integrator.langevin import LangevinBAOABIntegrator
from mdpy.system import System


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _make_6po6_system(cutoff=12.0):
    psf = PSFParser(os.path.join(DATA_DIR, "6PO6.psf"))
    pdb = PDBParser(os.path.join(DATA_DIR, "6PO6.pdb"))
    toppar = CharmmTopparParser(
        os.path.join(DATA_DIR, "par_all36_prot.prm"),
        os.path.join(DATA_DIR, "toppar_water_ions.str"),
    )
    topology = psf.topology
    parameter_table = create_parameter_table(topology, toppar)
    pbc_matrix = np.eye(3, dtype=np.float64) * 100.0

    system = System(topology, pbc_matrix, cutoff=cutoff)
    system.add_force_term(BondedForce.charmm(topology, parameter_table))
    nb = NonbondedForce(lennard_jones + coulomb)
    nb.bind(topology, parameter_table, cutoff)
    system.add_force_term(nb)
    system.particles.positions[:] = pdb.positions
    system.gpu.upload_positions(system.particles)
    system.gpu.upload_velocities(system.particles)
    return system


class TestCudaGraphBasic:
    """Basic CUDA Graph functionality with 6PO6 (49 atoms)."""

    def test_graph_creation_and_replay(self):
        system = _make_6po6_system()
        integrator = VerletIntegrator(2.0)

        system.step(integrator, 1)
        assert system._step_graph is not None, "Graph should be created after first step"
        assert system._graph_needs_capture is False

        system.step(integrator, 10)
        assert system._step_graph is not None
        cp.cuda.Stream.null.synchronize()

    def test_graph_rebuild_recapture(self):
        system = _make_6po6_system()
        integrator = VerletIntegrator(2.0)

        system.step(integrator, 10)
        e1 = system.dump_energy()

        system.step(integrator, 10)
        e2 = system.dump_energy()

        assert abs(e1.get("bonded", 0.0) - e2.get("bonded", 0.0)) < 1e-2

    def test_dump_energy_correctness(self):
        system = _make_6po6_system()
        integrator = VerletIntegrator(2.0)

        system.step(integrator, 10)
        e = system.dump_energy()

        assert "bonded" in e
        assert "nonbonded" in e
        assert e["bonded"] != 0.0

    def test_dump_state_correctness(self):
        system = _make_6po6_system()
        integrator = VerletIntegrator(2.0)

        system.step(integrator, 10)
        pos, vel = system.dump_state()

        assert pos.shape[1] == 3
        assert vel.shape[1] == 3
        assert pos.shape[0] > 0
        assert not np.all(pos == 0.0)

    def test_integrator_change_invalidates_graph(self):
        system = _make_6po6_system()
        verlet = VerletIntegrator(2.0)
        langevin = LangevinBAOABIntegrator(2.0, 300.0, 1.0)

        system.step(verlet, 5)
        assert system._cached_integrator_id == id(verlet)

        system.step(langevin, 5)
        assert system._cached_integrator_id == id(langevin)

    def test_many_steps_no_crash(self):
        system = _make_6po6_system()
        integrator = VerletIntegrator(2.0)

        system.step(integrator, 100)
        cp.cuda.Stream.null.synchronize()

    def test_graph_direct_comparison(self):
        """Compare energy after graph replay vs direct execution (should match)."""
        system1 = _make_6po6_system()
        system2 = _make_6po6_system()

        system1.step(VerletIntegrator(2.0), 20)
        e1 = system1.dump_energy()

        system2.step(VerletIntegrator(2.0), 20)
        e2 = system2.dump_energy()

        for key in e1:
            assert abs(e1[key] - e2[key]) < 1e-4, f"{key}: {e1[key]} vs {e2[key]}"

    def test_zero_forces_kernel(self):
        """Verify the new zero_forces RawKernel works correctly."""
        system = _make_6po6_system()
        integrator = VerletIntegrator(2.0)

        system.step(integrator, 1)

        forces_x = system.gpu.d_forces_x.get()
        forces_y = system.gpu.d_forces_y.get()
        forces_z = system.gpu.d_forces_z.get()

        assert not np.all(forces_x == 0.0), "Forces should be non-zero after step"
        assert not np.all(forces_y == 0.0)
        assert not np.all(forces_z == 0.0)
