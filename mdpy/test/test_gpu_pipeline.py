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
PSF = os.path.join(DATA_DIR, '6PO6.psf')
PDB = os.path.join(DATA_DIR, '6PO6.pdb')
PRM = os.path.join(DATA_DIR, 'par_all36_prot.prm')
STR = os.path.join(DATA_DIR, 'toppar_water_ions.str')


def _run_steps(system, integrator, n):
    for i in range(n):
        system.update_neighbor_list(sync_interval=n)
        system.compute_forces()
        integrator.step(system)


def _ensure_ready(system):
    pass


def _make_system():
    psf = PSFParser(PSF)
    pdb = PDBParser(PDB)
    toppar = CharmmTopparParser(PRM, STR)
    topology = psf.topology
    pt = create_parameter_table(topology, toppar)
    pbc = np.eye(3, dtype=np.float64) * 30.0
    system = System(topology)
    system.upload_pbc(pbc)
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
    system.upload_positions((frac @ pbc).astype(env.NUMPY_FLOAT))
    system.upload_velocities(np.zeros((topology.num_particles, 3), dtype=env.NUMPY_FLOAT))
    return system, VerletIntegrator(0.5)


def test_gpu_block_pair_classification():
    system, integrator = _make_system()
    _ensure_ready(system)
    _run_steps(system, integrator, 1)
    bl = system.block_list
    # Phase 2 unified mask path: all pairs run through the exclusion kernel.
    # num_main_block_pairs covers every block pair; excl count is 0 by design.
    assert bl.num_main_block_pairs == bl.num_block_pairs
    assert bl.num_exclusion_block_pairs == 0
    assert bl.d_main_block_pairs.shape[0] >= bl.num_main_block_pairs
    assert bl.d_main_interacting_atoms.shape[0] >= bl.num_main_block_pairs * 32
    # masks cover every main pair (zero mask = no exclusion = full force)
    assert bl.d_excl_exclusion_masks.shape[0] >= bl.num_main_block_pairs * 32
