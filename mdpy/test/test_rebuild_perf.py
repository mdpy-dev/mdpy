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


def test_parallel_csr_matches_sequential():
    system, integrator = _make_system_6po6()
    topology = system.topology

    from mdpy.core.topology import _get_gpu_kernels
    import cupy as cp

    topology.build_exclusion_map(scale_14=1.0)
    cpu_offset = topology.exclusion_offset.copy()

    num_particles = topology.num_particles
    total_pairs = topology.num_bonds + topology.num_angles + topology.num_dihedrals + topology.num_impropers

    kernels = _get_gpu_kernels()

    d_bond_idx = cp.asarray(topology.bond_indices.ravel().astype(np.int32))
    d_angle_idx = cp.asarray(topology.angle_indices.ravel().astype(np.int32))
    d_dihedral_idx = cp.asarray(topology.dihedral_indices.ravel().astype(np.int32))
    d_improper_idx = cp.asarray(topology.improper_indices.ravel().astype(np.int32))

    d_pair_i = cp.empty(total_pairs, dtype=cp.int32)
    d_pair_j = cp.empty(total_pairs, dtype=cp.int32)
    d_pair_scale = cp.empty(total_pairs, dtype=cp.float32)

    block = 256
    grid = (total_pairs + block - 1) // block
    kernels['generate'](
        (grid,), (block,),
        (d_bond_idx, np.int32(topology.num_bonds),
         d_angle_idx, np.int32(topology.num_angles),
         d_dihedral_idx, np.int32(topology.num_dihedrals),
         d_improper_idx, np.int32(topology.num_impropers),
         np.float32(1.0),
         d_pair_i, d_pair_j, d_pair_scale,
         np.int32(total_pairs)))

    max_j = int(cp.max(d_pair_j)) + 1
    sort_key = (d_pair_i.astype(cp.int64) * np.int64(max_j * 2 + 2)
                + d_pair_j.astype(cp.int64) * np.int64(2)
                + (d_pair_scale > 0.0).astype(cp.int64))
    order = cp.argsort(sort_key)
    d_pair_i = d_pair_i[order]
    d_pair_j = d_pair_j[order]
    d_pair_scale = d_pair_scale[order]

    d_unique_i = cp.empty(total_pairs, dtype=cp.int32)
    d_unique_j = cp.empty(total_pairs, dtype=cp.int32)
    d_unique_scale = cp.empty(total_pairs, dtype=cp.float32)
    d_unique_count = cp.empty(1, dtype=cp.int32)

    kernels['dedup'](
        (1,), (1,),
        (d_pair_i, d_pair_j, d_pair_scale,
         d_unique_i, d_unique_j, d_unique_scale,
         d_unique_count,
         np.int32(total_pairs)))

    unique_count = int(d_unique_count[0])
    d_unique_i = d_unique_i[:unique_count]

    d_offset_seq = cp.zeros(num_particles + 1, dtype=cp.int32)
    kernels['csr'](
        (1,), (1,),
        (d_unique_i, np.int32(unique_count),
         np.int32(num_particles), d_offset_seq))
    sequential_offset = cp.asnumpy(d_offset_seq)

    d_offset_parallel = cp.full(num_particles + 1, -1, dtype=cp.int32)
    tpb = 256
    grid_csr = ((unique_count + 1 + tpb - 1) // tpb,)
    kernels['parallel_csr'](
        grid_csr, (tpb,),
        (d_unique_i, np.int32(unique_count),
         np.int32(num_particles), d_offset_parallel))

    tpb_fill = 256
    grid_fill = ((num_particles + tpb_fill - 1) // tpb_fill,)
    kernels['fill_csr_gaps'](
        grid_fill, (tpb_fill,),
        (d_offset_parallel, np.int32(num_particles)))

    parallel_offset = cp.asnumpy(d_offset_parallel)
    np.testing.assert_array_equal(parallel_offset, sequential_offset)
    np.testing.assert_array_equal(parallel_offset, cpu_offset)


def test_parallel_dedup_matches_sequential():
    system, integrator = _make_system_6po6()
    topology = system.topology

    from mdpy.core.topology import _get_gpu_kernels
    import cupy as cp

    topology.build_exclusion_map(scale_14=1.0)
    cpu_neighbors = topology.exclusion_neighbors.copy()
    cpu_scale = topology.exclusion_scale.copy()

    total_pairs = topology.num_bonds + topology.num_angles + topology.num_dihedrals + topology.num_impropers

    kernels = _get_gpu_kernels()
    d_bond_idx = cp.asarray(topology.bond_indices.ravel().astype(np.int32))
    d_angle_idx = cp.asarray(topology.angle_indices.ravel().astype(np.int32))
    d_dihedral_idx = cp.asarray(topology.dihedral_indices.ravel().astype(np.int32))
    d_improper_idx = cp.asarray(topology.improper_indices.ravel().astype(np.int32))

    d_pair_i = cp.empty(total_pairs, dtype=cp.int32)
    d_pair_j = cp.empty(total_pairs, dtype=cp.int32)
    d_pair_scale = cp.empty(total_pairs, dtype=cp.float32)

    block = 256
    grid = (total_pairs + block - 1) // block
    kernels['generate'](
        (grid,), (block,),
        (d_bond_idx, np.int32(topology.num_bonds),
         d_angle_idx, np.int32(topology.num_angles),
         d_dihedral_idx, np.int32(topology.num_dihedrals),
         d_improper_idx, np.int32(topology.num_impropers),
         np.float32(1.0),
         d_pair_i, d_pair_j, d_pair_scale,
         np.int32(total_pairs)))

    max_j = int(cp.max(d_pair_j)) + 1
    sort_key = (d_pair_i.astype(cp.int64) * np.int64(max_j * 2 + 2)
                + d_pair_j.astype(cp.int64) * np.int64(2)
                + (d_pair_scale > 0.0).astype(cp.int64))
    order = cp.argsort(sort_key)
    d_sorted_i = d_pair_i[order]
    d_sorted_j = d_pair_j[order]
    d_sorted_scale = d_pair_scale[order]

    d_seq_i = cp.empty(total_pairs, dtype=cp.int32)
    d_seq_j = cp.empty(total_pairs, dtype=cp.int32)
    d_seq_scale = cp.empty(total_pairs, dtype=cp.float32)
    d_seq_count = cp.empty(1, dtype=cp.int32)
    kernels['dedup'](
        (1,), (1,),
        (d_sorted_i, d_sorted_j, d_sorted_scale,
         d_seq_i, d_seq_j, d_seq_scale,
         d_seq_count,
         np.int32(total_pairs)))
    seq_count = int(d_seq_count[0])
    seq_neighbors = cp.asnumpy(d_seq_j[:seq_count])
    seq_scale = cp.asnumpy(d_seq_scale[:seq_count])

    d_flags = cp.zeros(total_pairs, dtype=cp.int32)
    tpb = 256
    grid_d = ((total_pairs + tpb - 1) // tpb,)
    kernels['parallel_dedup'](grid_d, (tpb,),
        (d_sorted_i, d_sorted_j, d_sorted_scale,
         np.int32(total_pairs), d_flags))

    scatter_idx = cp.cumsum(d_flags) - 1
    unique_count = int(scatter_idx[total_pairs - 1]) + 1
    assert unique_count == seq_count

    d_par_j = cp.full(unique_count, -1, dtype=cp.int32)
    d_par_scale = cp.zeros(unique_count, dtype=cp.float32)
    d_par_j[scatter_idx] = d_sorted_j
    d_par_scale[scatter_idx] = d_sorted_scale

    par_neighbors = cp.asnumpy(d_par_j[:unique_count])
    par_scale = cp.asnumpy(d_par_scale[:unique_count])

    np.testing.assert_array_equal(par_neighbors, seq_neighbors)
    np.testing.assert_allclose(par_scale, seq_scale, atol=1e-7)
    np.testing.assert_array_equal(par_neighbors, cpu_neighbors)
    np.testing.assert_allclose(par_scale, cpu_scale, atol=1e-7)


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
