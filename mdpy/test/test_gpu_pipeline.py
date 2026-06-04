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
        system.gpu.refresh_wrapped_positions()


def _ensure_ready(system):
    system.gpu.refresh_wrapped_positions()


def _make_system():
    psf = PSFParser(PSF)
    pdb = PDBParser(PDB)
    toppar = CharmmTopparParser(PRM, STR)
    topology = psf.topology
    pt = create_parameter_table(topology, toppar)
    pbc = np.eye(3, dtype=np.float64) * 30.0
    system = System(topology, pbc, cutoff=12.0)
    system.add_force_term(BondedForce.charmm(topology, pt))
    nb = NonbondedForce(lennard_jones + coulomb)
    nb.bind(topology, pt, 12.0)
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
    assert bl.num_exclusion_block_pairs + bl.num_main_block_pairs == bl.num_block_pairs
    assert bl.num_exclusion_block_pairs > 0
    assert bl.d_excl_block_pairs.shape[0] >= bl.num_exclusion_block_pairs
    assert bl.d_main_block_pairs.shape[0] >= bl.num_main_block_pairs
    assert bl.d_excl_interacting_atoms.shape[0] >= bl.num_exclusion_block_pairs * 32
    assert bl.d_excl_exclusion_masks.shape[0] >= bl.num_exclusion_block_pairs * 32
    assert bl.d_excl_scaling_masks.shape[0] >= bl.num_exclusion_block_pairs * 32
    assert bl.d_main_interacting_atoms.shape[0] >= bl.num_main_block_pairs * 32
