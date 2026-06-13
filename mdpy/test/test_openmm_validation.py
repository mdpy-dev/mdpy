"""Validate mdpy GPU forces against OpenMM Reference platform.

Uses NoCutoff to eliminate cutoff/switching differences between mdpy and OpenMM.
OpenMM CHARMM splits nonbonded into NonbondedForce (Coulomb + 1-4 LJ) and
CustomNonbondedForce (regular LJ). The reference sums both for comparison.

Known remaining differences:
  1. CMAPTorsionForce: OpenMM includes it (~5e-6), mdpy does not.
  2. mdpy uses a 100A cutoff (vs NoCutoff) — negligible for 49-atom system.
  3. Float32 (mdpy GPU) vs float64 (OpenMM Reference).
"""

import os
import numpy as np
import pytest

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def _setup_mdpy_system(psf_path, pdb_path, prm_path, cutoff=12.0):
    from mdpy.io.psf_parser import PSFParser
    from mdpy.io.pdb_parser import PDBParser
    from mdpy.io.charmm_toppar_parser import CharmmTopparParser
    from mdpy.io.charmm_toppar_parser import create_parameter_table
    from mdpy.force.bonded_force import BondedForce
    from mdpy.force.nonbonded_force import NonbondedForce
    from mdpy.force.expressions.lennard_jones import lennard_jones
    from mdpy.force.expressions.coulomb import coulomb
    from mdpy.system import System

    psf = PSFParser(psf_path)
    pdb = PDBParser(pdb_path)
    toppar = CharmmTopparParser(prm_path)
    topology = psf.topology
    parameter_table = create_parameter_table(topology, toppar)
    pbc_matrix = np.eye(3, dtype=np.float64) * 100.0
    pbc_inv = np.linalg.inv(pbc_matrix)

    system = System(topology, pbc_matrix, cutoff=cutoff)
    system.add_force_term(BondedForce.charmm(topology, parameter_table))
    nb = NonbondedForce(lennard_jones + coulomb, cutoff=cutoff)
    lj_pair = parameter_table.type_pair_parameters['lj_pair']
    nb.set_pair_parameter('sigma', lj_pair[0::2].astype(np.float32))
    nb.set_pair_parameter('epsilon', lj_pair[1::2].astype(np.float32))
    system.add_force_term(nb)

    raw_positions = pdb.positions.astype(np.float64)
    frac = raw_positions @ pbc_inv
    frac -= np.floor(frac)
    wrapped = frac @ pbc_matrix

    system.upload_positions(wrapped.astype(np.float32))
    system.upload_velocities(np.zeros((topology.num_particles, 3), dtype=np.float32))
    positions_2d = (
        system.gpu.d_positions_x,
        system.gpu.d_positions_y,
        system.gpu.d_positions_z,
    )
    system.block_list.rebuild(
        positions_2d, topology,
        system.pbc_matrix, system.pbc_inv,
    )
    system._permute_all_arrays()
    system.block_list.build_block_pairs(topology, system.pbc_matrix)
    for term in system.force_terms:
        if hasattr(term, 'bind_sorted'):
            term.bind_sorted(topology, system.block_list, system.gpu)
    system.compute_forces()
    bl = system.block_list
    gpu = system.gpu
    sorted_to_pdb = bl.d_sorted_to_pdb
    frc = np.stack([
        gpu.permute_from_sorted(sorted_to_pdb, gpu.d_forces_x).get(),
        gpu.permute_from_sorted(sorted_to_pdb, gpu.d_forces_y).get(),
        gpu.permute_from_sorted(sorted_to_pdb, gpu.d_forces_z).get(),
    ], axis=1)
    system._validation_forces = frc
    system._validation_positions = wrapped.astype(np.float32)
    return system


def _load_reference(name):
    path = os.path.join(DATA_DIR, f'openmm_reference_{name}.npz')
    if not os.path.exists(path):
        pytest.skip(f'Reference file not found: {path}. Run generate_reference.py first.')
    return dict(np.load(path, allow_pickle=True))


def _rel_err(a, b):
    denom = max(abs(a), abs(b))
    if denom < 1e-14:
        return abs(a - b)
    return abs(a - b) / denom


@pytest.fixture(scope='module')
def ref_6po6():
    return _load_reference('6PO6')


@pytest.fixture(scope='module')
def mdpy_6po6():
    return _setup_mdpy_system(
        os.path.join(DATA_DIR, '6PO6.psf'),
        os.path.join(DATA_DIR, '6PO6.pdb'),
        os.path.join(DATA_DIR, 'par_all36_prot.prm'),
        cutoff=15.0,
    )


class TestOpenMMValidation6PO6:

    def test_positions_match(self, mdpy_6po6, ref_6po6):
        ref_pos = ref_6po6['positions']
        mdpy_pos = mdpy_6po6._validation_positions
        pbc_matrix = np.eye(3, dtype=np.float64) * 100.0
        pbc_inv = np.linalg.inv(pbc_matrix)
        frac = ref_pos.astype(np.float64) @ pbc_inv
        frac -= np.floor(frac)
        wrapped_ref = frac @ pbc_matrix
        np.testing.assert_allclose(mdpy_pos, wrapped_ref, atol=0.5)

    def test_bonded_energy(self, mdpy_6po6, ref_6po6):
        mdpy_bonded = mdpy_6po6.dump_energy().get('bonded', 0.0)
        ref_bonded = float(ref_6po6['ref_bonded_energy'])
        err = _rel_err(mdpy_bonded, ref_bonded)
        assert err < 0.05, (
            f'Bonded energy: mdpy={mdpy_bonded:.8f}, ref={ref_bonded:.8f}, '
            f'rel_err={err:.6e}'
        )

    def test_nonbonded_energy(self, mdpy_6po6, ref_6po6):
        mdpy_nb = mdpy_6po6.dump_energy().get('nonbonded', 0.0)
        ref_nb = float(ref_6po6['ref_nonbonded_energy'])
        err = _rel_err(mdpy_nb, ref_nb)
        assert err < 0.01, (
            f'Nonbonded energy: mdpy={mdpy_nb:.8f}, ref={ref_nb:.8f}, '
            f'rel_err={err:.6e}'
        )

    def test_total_energy(self, mdpy_6po6, ref_6po6):
        mdpy_total = sum(mdpy_6po6.dump_energy().values())
        ref_total = float(ref_6po6['ref_mdpy_total_energy'])
        abs_err = abs(mdpy_total - ref_total)
        rel_err = _rel_err(mdpy_total, ref_total)
        assert abs_err < 0.005 or rel_err < 0.05, (
            f'Total energy: mdpy={mdpy_total:.8f}, ref={ref_total:.8f}, '
            f'abs_err={abs_err:.6e}, rel_err={rel_err:.6e}'
        )

    def test_force_direction_correlation(self, mdpy_6po6, ref_6po6):
        ref_f = ref_6po6['forces'].flatten()
        mdpy_f = mdpy_6po6._validation_forces.flatten()
        mask = np.abs(ref_f) > 1e-8
        if not np.any(mask):
            return
        correlation = np.corrcoef(ref_f[mask], mdpy_f[mask])[0, 1]
        assert correlation > 0.99, f'Force direction correlation: {correlation:.6f}'

    def test_forces_magnitude_order(self, mdpy_6po6, ref_6po6):
        ref_forces = ref_6po6['forces']
        mdpy_forces = mdpy_6po6._validation_forces
        ref_norms = np.linalg.norm(ref_forces, axis=1)
        mdpy_norms = np.linalg.norm(mdpy_forces, axis=1)
        correlation = np.corrcoef(ref_norms, mdpy_norms)[0, 1]
        assert correlation > 0.99, f'Force magnitude correlation: {correlation:.6f}'
