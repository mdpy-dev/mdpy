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
from mdpy.core.topology import Builder

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


def _build_reference_exclusion_map(topology, scale_14=1.0):
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
    builder.build_exclusion_map(scale_14=scale_14)
    ref_topo, _ = builder.build()
    return ref_topo.exclusion_offset, ref_topo.exclusion_neighbors, ref_topo.exclusion_scale


def test_gpu_exclusion_map_matches_cpu():
    system, integrator = _make_system_6po6()
    topology = system.topology

    ref_offset, ref_neighbors, ref_scale = _build_reference_exclusion_map(topology, scale_14=1.0)

    from mdpy.core.topology import build_exclusion_map_gpu
    gpu_offset, gpu_neighbors, gpu_scale, _ = build_exclusion_map_gpu(topology, scale_14=1.0)

    np.testing.assert_array_equal(cp.asnumpy(gpu_offset), ref_offset)
    np.testing.assert_array_equal(cp.asnumpy(gpu_neighbors), ref_neighbors)
    np.testing.assert_allclose(cp.asnumpy(gpu_scale), ref_scale, atol=1e-7)


def test_exclusion_map_after_remap():
    system, integrator = _make_system_6po6()
    topology = system.topology

    system.step(integrator, 1)

    ref_offset, ref_neighbors, ref_scale = _build_reference_exclusion_map(topology, scale_14=1.0)

    from mdpy.core.topology import build_exclusion_map_gpu
    gpu_offset, gpu_neighbors, gpu_scale, _ = build_exclusion_map_gpu(topology, scale_14=1.0)

    np.testing.assert_array_equal(cp.asnumpy(gpu_offset), ref_offset)
    np.testing.assert_array_equal(cp.asnumpy(gpu_neighbors), ref_neighbors)
    np.testing.assert_allclose(cp.asnumpy(gpu_scale), ref_scale, atol=1e-7)


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


def test_permute_fast_path_matches_full_rebuild():
    system, integrator = _make_system_6po6()
    topology = system.topology

    from mdpy.core.topology import build_exclusion_map_gpu, permute_exclusion_pairs_gpu
    import cupy as cp

    gpu_offset, gpu_neighbors, gpu_scale, gpu_unique_i = build_exclusion_map_gpu(topology, scale_14=1.0)

    rng = np.random.default_rng(42)
    perm = np.arange(topology.num_particles, dtype=np.int32)
    rng.shuffle(perm)
    d_perm = cp.asarray(perm)

    d_composed_perm = cp.empty(topology.num_particles, dtype=cp.int32)
    d_composed_perm[d_perm] = cp.arange(topology.num_particles, dtype=cp.int32)

    result = permute_exclusion_pairs_gpu(
        gpu_unique_i, gpu_neighbors, gpu_scale,
        d_composed_perm, topology.num_particles)
    d_offset, d_neighbors, d_scale = result[0], result[1], result[2]

    remap = cp.asnumpy(d_composed_perm)
    topology_remapped = topology
    topology_remapped.bond_indices = remap[topology_remapped.bond_indices]
    topology_remapped.angle_indices = remap[topology_remapped.angle_indices]
    topology_remapped.dihedral_indices = remap[topology_remapped.dihedral_indices]
    topology_remapped.improper_indices = remap[topology_remapped.improper_indices]

    ref_offset, ref_neighbors, ref_scale = _build_reference_exclusion_map(topology_remapped, scale_14=1.0)
    gt_offset, gt_neighbors, gt_scale, _ = build_exclusion_map_gpu(topology_remapped, scale_14=1.0)

    np.testing.assert_array_equal(cp.asnumpy(d_offset), cp.asnumpy(gt_offset))
    np.testing.assert_array_equal(cp.asnumpy(d_neighbors), cp.asnumpy(gt_neighbors))
    np.testing.assert_allclose(cp.asnumpy(d_scale), cp.asnumpy(gt_scale), atol=1e-7)
    np.testing.assert_array_equal(ref_offset, cp.asnumpy(gt_offset))
    np.testing.assert_array_equal(ref_neighbors, cp.asnumpy(gt_neighbors))
    np.testing.assert_allclose(ref_scale, cp.asnumpy(gt_scale), atol=1e-7)
