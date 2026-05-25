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

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def _make_system(box=30.0, cutoff=12.0):
    ff = CharmmForcefield(
        os.path.join(DATA_DIR, '6PO6.psf'),
        os.path.join(DATA_DIR, '6PO6.pdb'),
        [os.path.join(DATA_DIR, 'par_all36_prot.prm'),
         os.path.join(DATA_DIR, 'toppar_water_ions.str')])
    topology = ff.create_topology()
    pt = ff.create_parameter_table()
    pbc = np.eye(3, dtype=np.float64) * box
    system = System(topology, pbc, cutoff=cutoff)
    system.add_force_term(BondedForce.charmm(topology, pt))
    nb = NonbondedForce(lennard_jones + coulomb)
    nb.bind(topology, pt, cutoff)
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


def test_exclusion_data_preserved_across_rebuild():
    system, integrator = _make_system()
    system.step(integrator, 1)

    tl = system.tile_list
    assert tl._d_excl_offset is not None, "exclusion data should be set after first step"

    positions_soa = system.gpu.get_positions_2d()
    if tl.check_rebuild(positions_soa):
        pdb_to_sorted_gpu, _ = tl.rebuild(positions_soa, system.topology,
                                           system.pbc_matrix, system.pbc_inv)
        system._permute_all_arrays(pdb_to_sorted_gpu, None)
        tl.build_tiles(system.topology, system.pbc_matrix)

    assert tl._d_excl_offset is not None, "exclusion data should survive rebuild"

    system.step(integrator, 5)
    pos, vel = system.dump_state()
    assert not np.any(np.isnan(pos))
    assert not np.any(np.isnan(vel))


def test_exclusion_map_constant_sort_key():
    system, _ = _make_system()
    topology = system.topology
    from mdpy.core.topology import build_exclusion_map_gpu

    d_offset, d_neighbors, d_scale, d_unique_i = build_exclusion_map_gpu(topology, scale_14=1.0)

    topology.build_exclusion_map(scale_14=1.0)
    cpu_offset = topology.exclusion_offset
    cpu_neighbors = topology.exclusion_neighbors
    cpu_scale = topology.exclusion_scale

    np.testing.assert_array_equal(cp.asnumpy(d_offset), cpu_offset)
    np.testing.assert_array_equal(cp.asnumpy(d_neighbors), cpu_neighbors)
    np.testing.assert_allclose(cp.asnumpy(d_scale), cpu_scale, atol=1e-7)
