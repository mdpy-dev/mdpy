import numpy as np
import pytest
import cupy as cp
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
from mdpy.system import System

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def _run_steps(system, integrator, n):
    for _ in range(n):
        system.check_and_rebuild()
        system.compute_forces()
        integrator.step(system)
        system.gpu.refresh_wrapped_positions()


def _ensure_ready(system):
    system.ensure_uploaded()
    system.gpu.refresh_wrapped_positions()


def _make_system(box=30.0, cutoff=12.0):
    psf = PSFParser(os.path.join(DATA_DIR, '6PO6.psf'))
    pdb = PDBParser(os.path.join(DATA_DIR, '6PO6.pdb'))
    toppar = CharmmTopparParser(
        os.path.join(DATA_DIR, 'par_all36_prot.prm'),
        os.path.join(DATA_DIR, 'toppar_water_ions.str'))
    topology = psf.topology
    pt = create_parameter_table(topology, toppar)
    pbc = np.eye(3, dtype=np.float64) * box
    system = System(topology, pbc, cutoff=cutoff)
    system.add_force_term(BondedForce.charmm(topology, pt))
    nb = NonbondedForce(lennard_jones + coulomb)
    nb.bind(topology, pt, cutoff)
    system.add_force_term(nb)
    raw = pdb.positions.astype(np.float64)
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
    _ensure_ready(system)
    _run_steps(system, integrator, 1)

    bl = system.block_list
    assert bl._d_excl_offset is not None, "exclusion data should be set after first step"

    system.check_and_rebuild()

    assert bl._d_excl_offset is not None, "exclusion data should survive rebuild"

    _run_steps(system, integrator, 5)
    pos, vel = system.dump_state()
    assert not np.any(np.isnan(pos))
    assert not np.any(np.isnan(vel))


def test_exclusion_map_constant_sort_key():
    system, _ = _make_system()
    topology = system.topology
    from mdpy.core.topology import build_exclusion_map_gpu, Builder

    d_offset, d_neighbors, d_scale, d_unique_i = build_exclusion_map_gpu(topology, scale_14=1.0)

    builder = Builder()
    builder.set_particles(
        masses=topology.masses,
        charges=topology.charges,
        particle_types=topology.particle_types,
    )
    for i in range(topology.num_bonds):
        builder.add_bond(
            int(topology.bond_indices[i, 0]),
            int(topology.bond_indices[i, 1]), 0, 0)
    for i in range(topology.num_angles):
        builder.add_angle(
            int(topology.angle_indices[i, 0]),
            int(topology.angle_indices[i, 1]),
            int(topology.angle_indices[i, 2]), 0, 0)
    for i in range(topology.num_dihedrals):
        builder.add_dihedral(
            int(topology.dihedral_indices[i, 0]),
            int(topology.dihedral_indices[i, 1]),
            int(topology.dihedral_indices[i, 2]),
            int(topology.dihedral_indices[i, 3]), 0, 0, 0)
    for i in range(topology.num_impropers):
        builder.add_improper(
            int(topology.improper_indices[i, 0]),
            int(topology.improper_indices[i, 1]),
            int(topology.improper_indices[i, 2]),
            int(topology.improper_indices[i, 3]), 0, 0)
    builder.build_exclusion_map(scale_14=1.0)
    ref_topo, _ = builder.build()

    np.testing.assert_array_equal(cp.asnumpy(d_offset), ref_topo.exclusion_offset)
    np.testing.assert_array_equal(cp.asnumpy(d_neighbors), ref_topo.exclusion_neighbors)
    np.testing.assert_allclose(cp.asnumpy(d_scale), ref_topo.exclusion_scale, atol=1e-7)
