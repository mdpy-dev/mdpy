import os
import numpy as np
import cupy as cp
import pytest
from mdpy.io.psf_parser import PSFParser
from mdpy.io.pdb_parser import PDBParser
from mdpy.io.charmm_toppar_parser import CharmmTopparParser, create_parameter_table
from mdpy.force.bonded_force import BondedForce
from mdpy.force.factories.charmm import create_bonded_group
from mdpy.force.nonbonded_force import NonbondedForce
from mdpy.force.expressions.lennard_jones import lennard_jones
from mdpy.force.expressions.screened_coulomb import screened_coulomb
from mdpy.force.pme_reciprocal_force import PMEReciprocalForce
from mdpy.integrator.verlet import VerletIntegrator
from mdpy.system import System

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
PSF = os.path.join(DATA_DIR, '6PO6.psf')
PDB = os.path.join(DATA_DIR, '6PO6.pdb')
PRM = os.path.join(DATA_DIR, 'par_all36_prot.prm')
BOX = 100.0
CUTOFF = 10.0


def _build_system():
    psf = PSFParser(PSF)
    pdb = PDBParser(PDB)
    toppar = CharmmTopparParser(PRM)
    topology = psf.topology
    parameter_table = create_parameter_table(topology, toppar)
    pbc_matrix = np.eye(3, dtype=np.float32) * BOX

    system = System(topology, pbc_matrix, cutoff=CUTOFF)
    system.add_force_term(create_bonded_group(topology, parameter_table))

    nb = NonbondedForce(lennard_jones + screened_coulomb, cutoff=CUTOFF)
    lj_pair = parameter_table.type_pair_parameters['lj_pair']
    nb.set_pair_parameter('sigma', lj_pair[0::2].astype(np.float32))
    nb.set_pair_parameter('epsilon', lj_pair[1::2].astype(np.float32))
    system.add_force_term(nb)

    pme = PMEReciprocalForce(CUTOFF)
    pme.bind(topology, parameter_table, pbc_matrix=pbc_matrix)
    system.add_force_term(pme)

    positions = pdb.positions.astype(np.float32)
    pbc_inv = np.linalg.inv(pbc_matrix.astype(np.float64))
    frac = positions.astype(np.float64) @ pbc_inv
    frac -= np.floor(frac)
    wrapped = (frac @ pbc_matrix).astype(np.float32)

    system.upload_positions(wrapped)
    system.upload_velocities(np.zeros((topology.num_particles, 3), dtype=np.float32))
    return system, pme


def test_pme_charges_sorted_after_rebuild():
    system, pme = _build_system()

    charges_before = cp.asnumpy(pme._d_charges).copy()

    system.update_neighbor_list(sync_interval=10)
    system.compute_forces()

    bl = system.block_list
    assert bl.d_raw_order.size > 0, "Rebuild did not produce sort order"

    charges_after = cp.asnumpy(pme._d_charges)

    perm = cp.asnumpy(bl.d_raw_order)
    expected_sorted = charges_before[perm]
    np.testing.assert_array_almost_equal(
        charges_after, expected_sorted, decimal=5,
        err_msg="PME charges not correctly permuted to sorted order"
    )


def test_pme_exclusion_pairs_sorted_after_rebuild():
    system, pme = _build_system()

    if pme._num_exclusion_pairs == 0:
        pytest.skip("No exclusion pairs in 6PO6")

    pair_i_before = cp.asnumpy(pme._d_pair_i).copy()
    pair_j_before = cp.asnumpy(pme._d_pair_j).copy()

    system.update_neighbor_list(sync_interval=10)
    system.compute_forces()

    bl = system.block_list
    pdb_to_sorted = cp.asnumpy(bl.d_pdb_to_sorted)

    expected_i = pdb_to_sorted[pair_i_before]
    expected_j = pdb_to_sorted[pair_j_before]

    pair_i_after = cp.asnumpy(pme._d_pair_i)
    pair_j_after = cp.asnumpy(pme._d_pair_j)

    np.testing.assert_array_equal(pair_i_after, expected_i,
        err_msg="PME exclusion pair_i not remapped to sorted order")
    np.testing.assert_array_equal(pair_j_after, expected_j,
        err_msg="PME exclusion pair_j not remapped to sorted order")


def test_pme_energy_no_divergence():
    system, pme = _build_system()
    integrator = VerletIntegrator(0.5)

    energies = []
    for step in range(500):
        system.update_neighbor_list(sync_interval=10)
        system.compute_forces()
        integrator.step(system)
        if step % 50 == 0:
            e = system.dump_energy()
            total = sum(e.values())
            energies.append(total)

    for i, e in enumerate(energies):
        assert np.isfinite(e), f"Non-finite energy at step {i * 50}: {e}"

    e0 = energies[0]
    max_abs = max(abs(e) for e in energies)
    for i, e in enumerate(energies):
        assert abs(e - e0) < max_abs * 2.0 + 1.0, (
            f"Energy diverged at step {i * 50}: E={e:.4f}, E0={e0:.4f}"
        )
