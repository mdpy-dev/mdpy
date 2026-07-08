"""Force term isolation tests for trajectory stability.

Runs multi-step simulations with ONLY bonded forces, ONLY nonbonded forces,
and both combined, to determine which force term causes trajectory instability.

Uses 1M9Z (95,567 atoms) with time_step=0.5 fs Verlet integration.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n md_analysis pytest \\
        mdpy/test/test_force_isolation.py -sv -m slow
"""

import os
import numpy as np
import pytest

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

PSF_PATH = os.path.join(DATA_DIR, '1M9Z.psf')
PDB_PATH = os.path.join(DATA_DIR, '1M9Z_minimized.pdb')
PRM_PATH = os.path.join(DATA_DIR, 'par_all36_prot.prm')
STR_PATH = os.path.join(DATA_DIR, 'toppar_water_ions.str')

BOX_SIZE = 108.0
CUTOFF = 12.0
KCAL = 1.0 / 4.1840286576e-4


def _setup_system(include_bonded, include_nonbonded):
    from mdpy.io.psf_parser import PSFParser
    from mdpy.io.pdb_parser import PDBParser
    from mdpy.io.charmm_toppar_parser import CharmmTopparParser
    from mdpy.io.charmm_toppar_parser import create_parameter_table
    from mdpy.force.bonded_force import BondedForce
    from mdpy.force.factories.charmm import create_bonded_group
    from mdpy.force.nonbonded_force import NonbondedForce
    from mdpy.force.expressions.lennard_jones import lennard_jones
    from mdpy.force.expressions.coulomb import coulomb
    from mdpy.system import System

    psf = PSFParser(PSF_PATH)
    pdb = PDBParser(PDB_PATH)
    toppar = CharmmTopparParser(PRM_PATH, STR_PATH)
    topology = psf.topology
    parameter_table = create_parameter_table(topology, toppar)

    pbc_matrix = np.eye(3, dtype=np.float64) * BOX_SIZE
    pbc_inv = np.linalg.inv(pbc_matrix)

    system = System(topology)

    system.set_pbc(pbc_matrix)
    system._cutoff = CUTOFF

    if include_bonded:
        system.add_force_term(create_bonded_group(topology, parameter_table))

    if include_nonbonded:
        nb = NonbondedForce(lennard_jones + coulomb, cutoff=CUTOFF)
        lj_pair = parameter_table.type_pair_parameters['lj_pair']
        nb.set_pair_parameter('sigma', lj_pair[0::2].astype(np.float32))
        nb.set_pair_parameter('epsilon', lj_pair[1::2].astype(np.float32))
        system.add_force_term(nb)

    raw = pdb.positions.astype(np.float64)
    frac = raw @ pbc_inv
    frac -= np.floor(frac)
    wrapped = frac @ pbc_matrix
    system.set_positions(wrapped.astype(np.float32))
    system.set_velocities(
        np.zeros((topology.num_particles, 3), dtype=np.float32),
    )

    system.update_neighbor_list(force_rebuild=True)

    return system


def _run_simulation(include_bonded, include_nonbonded, num_steps):
    from mdpy.integrator.verlet import VerletIntegrator

    system = _setup_system(include_bonded, include_nonbonded)
    integrator = VerletIntegrator(0.5)

    energies_list = []
    for step in range(num_steps):
        system.update_neighbor_list(sync_interval=10)
        system.compute_forces()
        energies = system.dump_energy()
        total = sum(energies.values()) * KCAL
        energies_list.append(total)
        integrator.step(system)

    return np.array(energies_list)


@pytest.mark.slow
def test_bonded_only_stability():
    energies = _run_simulation(
        include_bonded=True, include_nonbonded=False, num_steps=100,
    )

    assert np.all(np.isfinite(energies)), (
        f"NaN/Inf in bonded-only energies at steps "
        f"{np.where(~np.isfinite(energies))[0]}"
    )

    first, last = energies[0], energies[-1]
    drift = abs(last - first) / max(abs(first), 1e-10)
    assert drift < 0.5, (
        f"Bonded-only energy drift {drift:.4f} (50%) over 100 steps: "
        f"first={first:.2f}, last={last:.2f}"
    )


@pytest.mark.slow
def test_nonbonded_only_stability():
    energies = _run_simulation(
        include_bonded=False, include_nonbonded=True, num_steps=100,
    )

    assert np.all(np.isfinite(energies)), (
        f"NaN/Inf in nonbonded-only energies at steps "
        f"{np.where(~np.isfinite(energies))[0]}"
    )

    first, last = energies[0], energies[-1]
    assert abs(last) < 10 * max(abs(first), 1e-10), (
        f"Nonbonded-only energy blew up: first={first:.2f}, last={last:.2f}"
    )


@pytest.mark.slow
def test_combined_stability():
    energies = _run_simulation(
        include_bonded=True, include_nonbonded=True, num_steps=100,
    )

    assert np.all(np.isfinite(energies)), (
        f"NaN/Inf in combined energies at steps "
        f"{np.where(~np.isfinite(energies))[0]}"
    )

    first, last = energies[0], energies[-1]
    assert abs(last) < 10 * max(abs(first), 1e-10), (
        f"Combined energy blew up: first={first:.2f}, last={last:.2f}"
    )
