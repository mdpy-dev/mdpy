import numpy as np
import pytest
import cupy as cp
import os
from mdpy.io.psf_parser import PSFParser
from mdpy.io.pdb_parser import PDBParser
from mdpy.io.charmm_toppar_parser import CharmmTopparParser
from mdpy.io.charmm_toppar_parser import create_parameter_table
from mdpy.force.bonded_force import BondedForce
from mdpy.force.factories.charmm import create_bonded_group
from mdpy.force.nonbonded_force import NonbondedForce
from mdpy.force.expressions.lennard_jones import lennard_jones
from mdpy.force.expressions.coulomb import coulomb
from mdpy.integrator.verlet import VerletIntegrator
from mdpy.system import System
from mdpy import env

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def _run_steps(system, integrator, n):
    for i in range(n):
        system.update_neighbor_list(sync_interval=n)
        system.compute_forces()
        integrator.step(system)


def _ensure_ready(system):
    pass


def _make_system(box=30.0, cutoff=12.0):
    psf = PSFParser(os.path.join(DATA_DIR, '6PO6.psf'))
    pdb = PDBParser(os.path.join(DATA_DIR, '6PO6.pdb'))
    toppar = CharmmTopparParser(
        os.path.join(DATA_DIR, 'par_all36_prot.prm'),
        os.path.join(DATA_DIR, 'toppar_water_ions.str'))
    topology = psf.topology
    pt = create_parameter_table(topology, toppar)
    pbc = np.eye(3, dtype=np.float64) * box
    system = System(topology)
    system.set_pbc(pbc)
    system.add_force_term(create_bonded_group(topology, pt))
    nb = NonbondedForce(lennard_jones + coulomb, cutoff=cutoff)
    lj_pair = pt.type_pair_parameters['lj_pair']
    nb.set_pair_parameter('sigma', lj_pair[0::2].astype(env.NUMPY_FLOAT))
    nb.set_pair_parameter('epsilon', lj_pair[1::2].astype(env.NUMPY_FLOAT))
    system.add_force_term(nb)
    raw = pdb.positions.astype(np.float64)
    pbc_inv = np.linalg.inv(pbc)
    frac = raw @ pbc_inv
    frac -= np.floor(frac)
    system.set_positions((frac @ pbc).astype(env.NUMPY_FLOAT))
    system.set_velocities(np.zeros((topology.num_particles, 3), dtype=env.NUMPY_FLOAT))
    return system, VerletIntegrator(0.5)


def test_exclusion_data_preserved_across_rebuild():
    system, integrator = _make_system()
    _ensure_ready(system)
    _run_steps(system, integrator, 1)

    # Exclusions now live on Topology as a lazy cached GPU property.
    # Across a rebuild the SAME arrays must be returned (not recomputed).
    csr1 = system.topology.exclusion_csr
    system.update_neighbor_list(force_rebuild=True)
    csr2 = system.topology.exclusion_csr
    assert csr1[0] is csr2[0]
    assert csr1[1] is csr2[1]

    _run_steps(system, integrator, 5)
    pos, vel = system.dump_state()
    assert not np.any(np.isnan(pos))
    assert not np.any(np.isnan(vel))


def test_exclusion_csr_matches_brute_force():
    from mdpy.core.topology import _build_bond_graph_exclusion_pairs

    system, _ = _make_system()
    topology = system.topology

    pair_i, pair_j, total, _, _, _ = _build_bond_graph_exclusion_pairs(
        topology.bond_indices, topology.num_particles)
    N = topology.num_particles
    if total == 0:
        ref_offset = np.zeros(N + 1, dtype=np.int32)
        ref_neighbors = np.empty(0, dtype=np.int32)
    else:
        bi_i = np.concatenate([pair_i, pair_j])
        bi_j = np.concatenate([pair_j, pair_i])
        order = np.lexsort((bi_j, bi_i))
        bi_i, bi_j = bi_i[order], bi_j[order]
        keep = np.ones(len(bi_i), dtype=bool)
        keep[1:] = (bi_i[1:] != bi_i[:-1]) | (bi_j[1:] != bi_j[:-1])
        bi_i, bi_j = bi_i[keep], bi_j[keep]
        count = np.zeros(N + 1, dtype=np.int32)
        np.add.at(count, bi_i + 1, 1)
        ref_offset = np.cumsum(count, dtype=np.int32)
        ref_neighbors = bi_j.astype(np.int32)

    offset, neighbors = topology.exclusion_csr
    np.testing.assert_array_equal(cp.asnumpy(offset), ref_offset)
    # The GPU scatter kernel uses atomicAdd, so within-row neighbor order is
    # non-deterministic. Compare per-row sorted sets instead of exact arrays.
    gpu_neighbors = cp.asnumpy(neighbors)
    for a in range(N):
        s, e = ref_offset[a], ref_offset[a + 1]
        g = np.sort(gpu_neighbors[s:e])
        r = np.sort(ref_neighbors[s:e])
        np.testing.assert_array_equal(g, r)
