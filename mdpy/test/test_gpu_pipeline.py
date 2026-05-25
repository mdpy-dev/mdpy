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

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
PSF = os.path.join(DATA_DIR, '6PO6.psf')
PDB = os.path.join(DATA_DIR, '6PO6.pdb')
PRM = os.path.join(DATA_DIR, 'par_all36_prot.prm')
STR = os.path.join(DATA_DIR, 'toppar_water_ions.str')


def _make_system():
    ff = CharmmForcefield(PSF, PDB, [PRM, STR])
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


def test_gpu_tile_classification():
    system, integrator = _make_system()
    system.step(integrator, 1)
    tl = system.tile_list
    assert tl.num_exclusion_tiles + tl.num_main_tiles == tl.num_tiles
    assert tl.num_exclusion_tiles > 0
    assert tl.d_excl_tiles.shape[0] == tl.num_exclusion_tiles
    assert tl.d_main_tiles.shape[0] == tl.num_main_tiles
    if tl.num_exclusion_tiles > 0:
        assert tl.d_excl_interacting_atoms.shape[0] == tl.num_exclusion_tiles * 32
        assert tl.d_excl_exclusion_masks.shape[0] == tl.num_exclusion_tiles * 32
        assert tl.d_excl_scaling_masks.shape[0] == tl.num_exclusion_tiles * 32
    if tl.num_main_tiles > 0:
        assert tl.d_main_interacting_atoms.shape[0] == tl.num_main_tiles * 32
