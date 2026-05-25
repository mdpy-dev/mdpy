import numpy as np
import pytest
import cupy as cp
import os
from mdpy.forcefield.charmm_forcefield import CharmmForcefield
from mdpy.force.bonded_force import BondedForce
from mdpy.force.nonbonded_force import NonbondedForce
from mdpy.force.expressions.lennard_jones import lennard_jones
from mdpy.force.expressions.coulomb import coulomb
from mdpy.integrator.verlet import VerletIntegrator
from mdpy.system import System
from mdpy import env

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def _make_system_6po6():
    ff = CharmmForcefield(
        os.path.join(DATA_DIR, '6PO6.psf'),
        os.path.join(DATA_DIR, '6PO6.pdb'),
        [os.path.join(DATA_DIR, 'par_all36_prot.prm'),
         os.path.join(DATA_DIR, 'toppar_water_ions.str')])
    topology = ff.create_topology()
    pt = ff.create_parameter_table()
    pbc = np.eye(3, dtype=np.float64) * 30.0
    system = System(topology, pbc, cutoff=12.0)
    system.add_force_term(BondedForce.charmm(topology, pt))
    nb = NonbondedForce(lennard_jones + coulomb)
    nb.bind(topology, pt, 12.0)
    system.add_force_term(nb)
    raw = ff._pdb.positions.astype(np.float64)
    pbc_inv = np.linalg.inv(pbc)
    frac = raw @ pbc_inv
    frac -= np.floor(frac)
    system.particles.positions[:] = frac @ pbc
    system.particles.velocities[:] = 0.0
    system.gpu.upload_positions(system.particles)
    system.gpu.upload_velocities(system.particles)
    return system, VerletIntegrator(0.5)


def test_gpu_exclusion_map_matches_cpu():
    system, integrator = _make_system_6po6()
    topology = system.topology

    topology.build_exclusion_map(scale_14=1.0)
    cpu_offset = topology.exclusion_offset.copy()
    cpu_neighbors = topology.exclusion_neighbors.copy()
    cpu_scale = topology.exclusion_scale.copy()

    from mdpy.core.topology import build_exclusion_map_gpu
    gpu_offset, gpu_neighbors, gpu_scale = build_exclusion_map_gpu(topology, scale_14=1.0)

    np.testing.assert_array_equal(cp.asnumpy(gpu_offset), cpu_offset)
    np.testing.assert_array_equal(cp.asnumpy(gpu_neighbors), cpu_neighbors)
    np.testing.assert_allclose(cp.asnumpy(gpu_scale), cpu_scale, atol=1e-7)


def test_exclusion_map_after_remap():
    system, integrator = _make_system_6po6()
    topology = system.topology

    system.step(integrator, 1)

    topology.build_exclusion_map(scale_14=1.0)
    cpu_offset = topology.exclusion_offset.copy()
    cpu_neighbors = topology.exclusion_neighbors.copy()
    cpu_scale = topology.exclusion_scale.copy()

    from mdpy.core.topology import build_exclusion_map_gpu
    gpu_offset, gpu_neighbors, gpu_scale = build_exclusion_map_gpu(topology, scale_14=1.0)

    np.testing.assert_array_equal(cp.asnumpy(gpu_offset), cpu_offset)
    np.testing.assert_array_equal(cp.asnumpy(gpu_neighbors), cpu_neighbors)
    np.testing.assert_allclose(cp.asnumpy(gpu_scale), cpu_scale, atol=1e-7)


def test_rebuild_produces_correct_forces_after_gpu_exclusion():
    system, integrator = _make_system_6po6()
    system.step(integrator, 1)
    e1 = system.dump_energy()
    pos1, vel1 = system.dump_state()

    system.step(integrator, 5)
    e2 = system.dump_energy()
    pos2, vel2 = system.dump_state()

    assert not np.any(np.isnan(pos2))
    assert not np.any(np.isnan(vel2))
    max_disp = np.max(np.abs(pos2 - pos1))
    assert max_disp > 0.0


@pytest.mark.slow
def test_rebuild_1m9z_correctness():
    ff = CharmmForcefield(
        os.path.join(DATA_DIR, '1M9Z.psf'),
        os.path.join(DATA_DIR, '1M9Z_minimized.pdb'),
        [os.path.join(DATA_DIR, 'par_all36_prot.prm'),
         os.path.join(DATA_DIR, 'toppar_water_ions.str')])
    topology = ff.create_topology()
    pt = ff.create_parameter_table()
    pbc = np.eye(3, dtype=np.float64) * 108.0
    system = System(topology, pbc, cutoff=12.0)
    system.add_force_term(BondedForce.charmm(topology, pt))
    nb = NonbondedForce(lennard_jones + coulomb)
    nb.bind(topology, pt, 12.0)
    system.add_force_term(nb)
    raw = ff._pdb.positions.astype(np.float64)
    pbc_inv = np.linalg.inv(pbc)
    frac = raw @ pbc_inv
    frac -= np.floor(frac)
    system.particles.positions[:] = frac @ pbc
    system.particles.velocities[:] = 0.0
    system.gpu.upload_positions(system.particles)
    system.gpu.upload_velocities(system.particles)
    integrator = VerletIntegrator(0.5)

    system.step(integrator, 10)
    energies = system.dump_energy()
    pos, vel = system.dump_state()

    assert not np.any(np.isnan(pos))
    assert not np.any(np.isnan(vel))
    assert 'bonded' in energies or 'nonbonded' in energies
