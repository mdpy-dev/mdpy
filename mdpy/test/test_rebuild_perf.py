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
from mdpy.core.state import State
from mdpy import env

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def _run_steps(system, integrator, n):
    for i in range(n):
        system.update_neighbor_list(sync_interval=n)
        system.compute_forces()
        integrator.step(system)


def _ensure_ready(system):
    pass


def _make_system_6po6():
    psf = PSFParser(os.path.join(DATA_DIR, '6PO6.psf'))
    pdb = PDBParser(os.path.join(DATA_DIR, '6PO6.pdb'))
    toppar = CharmmTopparParser(
        os.path.join(DATA_DIR, 'par_all36_prot.prm'),
        os.path.join(DATA_DIR, 'toppar_water_ions.str'))
    topology = psf.topology
    pt = create_parameter_table(topology, toppar, type_names=psf.particle_type_names)
    pbc = np.eye(3, dtype=np.float64) * 30.0
    state = State(topology.num_particles)
    state.set_masses(psf.masses)
    state.set_charges(psf.charges)
    state.set_type_indices(psf.particle_type_indices)
    system = System(topology, state)
    system.set_pbc(pbc)
    system.add_force_term(create_bonded_group(topology, pt))
    nb = NonbondedForce(lennard_jones + coulomb, cutoff=12.0)
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


def _make_system_1m9z():
    psf = PSFParser(os.path.join(DATA_DIR, '1M9Z.psf'))
    pdb = PDBParser(os.path.join(DATA_DIR, '1M9Z_minimized.pdb'))
    toppar = CharmmTopparParser(
        os.path.join(DATA_DIR, 'par_all36_prot.prm'),
        os.path.join(DATA_DIR, 'toppar_water_ions.str'))
    topology = psf.topology
    pt = create_parameter_table(topology, toppar, type_names=psf.particle_type_names)
    pbc = np.eye(3, dtype=np.float64) * 108.0
    state = State(topology.num_particles)
    state.set_masses(psf.masses)
    state.set_charges(psf.charges)
    state.set_type_indices(psf.particle_type_indices)
    system = System(topology, state)
    system.set_pbc(pbc)
    system.add_force_term(create_bonded_group(topology, pt))
    nb = NonbondedForce(lennard_jones + coulomb, cutoff=12.0)
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


def test_rebuild_produces_correct_forces_after_gpu_exclusion():
    system, integrator = _make_system_6po6()
    _ensure_ready(system)
    _run_steps(system, integrator, 1)
    e1 = system.dump_energy()
    pos1, vel1 = system.dump_state()

    _run_steps(system, integrator, 5)
    e2 = system.dump_energy()
    pos2, vel2 = system.dump_state()

    assert not np.any(np.isnan(pos2))
    assert not np.any(np.isnan(vel2))
    max_disp = np.max(np.abs(pos2 - pos1))
    assert max_disp > 0.0


@pytest.mark.slow
def test_rebuild_1m9z_correctness():
    system, integrator = _make_system_1m9z()

    _ensure_ready(system)
    _run_steps(system, integrator, 10)
    energies = system.dump_energy()
    pos, vel = system.dump_state()

    assert not np.any(np.isnan(pos))
    assert not np.any(np.isnan(vel))
    assert 'bonded' in energies or 'nonbonded' in energies
