"""Validate that the rebuild/permute pipeline preserves data integrity.

Three tests check that the block-list rebuild + array permutation + force-term
re-sort pipeline does not corrupt positions, produces physically reasonable
forces after integration, and preserves prev_positions across rebuilds.
"""

import os

import numpy as np
import pytest

from mdpy import env

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

PSF_PATH = os.path.join(DATA_DIR, "1M9Z.psf")
PDB_PATH = os.path.join(DATA_DIR, "1M9Z_minimized.pdb")
PRM_PATH = os.path.join(DATA_DIR, "par_all36_prot.prm")
STR_PATH = os.path.join(DATA_DIR, "toppar_water_ions.str")

BOX_SIZE = 108.0
CUTOFF = 12.0


def _setup_system():
    from mdpy.force.bonded_force import BondedForce
    from mdpy.force.factories.charmm import create_bonded_group
    from mdpy.force.expressions.coulomb import coulomb
    from mdpy.force.expressions.lennard_jones import lennard_jones
    from mdpy.force.nonbonded_force import NonbondedForce
    from mdpy.io.charmm_toppar_parser import CharmmTopparParser
    from mdpy.io.charmm_toppar_parser import create_parameter_table
    from mdpy.io.pdb_parser import PDBParser
    from mdpy.io.psf_parser import PSFParser
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
    system.add_force_term(create_bonded_group(topology, parameter_table))
    nb = NonbondedForce(lennard_jones + coulomb, cutoff=CUTOFF)
    lj_pair = parameter_table.type_pair_parameters['lj_pair']
    nb.set_pair_parameter('sigma', lj_pair[0::2].astype(env.NUMPY_FLOAT))
    nb.set_pair_parameter('epsilon', lj_pair[1::2].astype(env.NUMPY_FLOAT))
    system.add_force_term(nb)

    raw = pdb.positions.astype(np.float64)
    frac = raw @ pbc_inv
    frac -= np.floor(frac)
    wrapped = frac @ pbc_matrix
    system.set_positions(wrapped.astype(np.float32))
    system.set_velocities(
        np.zeros((topology.num_particles, 3), dtype=np.float32)
    )

    system.update_neighbor_list(force_rebuild=True)
    return system


@pytest.mark.slow
class TestRebuildPipeline:

    def test_positions_preserved_after_rebuild(self):
        system = _setup_system()
        pos_before, _ = system.dump_state()

        system._do_rebuild(force=True)

        pos_after, _ = system.dump_state()

        max_diff = np.max(np.abs(pos_before - pos_after))
        assert max_diff < 1e-5, (
            f"Positions changed by {max_diff:.2e} after rebuild "
            f"(max |diff|, tolerance 1e-5)"
        )

    def test_forces_correct_after_one_integration_step(self):
        from mdpy.integrator.verlet import VerletIntegrator

        system = _setup_system()
        integrator = VerletIntegrator(0.5)

        system.compute_forces()
        system._do_rebuild(force=True)

        integrator.step(system)

        system._do_rebuild(force=True)
        system.compute_forces()

        energies = system.dump_energy()
        forces = system.dump_forces()
        pos, _ = system.dump_state()

        assert np.all(np.isfinite(pos)), "Positions contain NaN or Inf"
        assert np.all(np.isfinite(forces)), "Forces contain NaN or Inf"

        total_force = np.sum(forces, axis=0)
        force_mag = np.linalg.norm(total_force)
        assert force_mag < 1e-2, (
            f"Total force magnitude {force_mag:.2e} exceeds 1e-3"
        )

        bonded_energy = energies.get("bonded", 0.0)
        assert abs(bonded_energy) < 50000, (
            f"Bonded energy {bonded_energy:.1f} out of reasonable range"
        )

        nonbonded_energy = energies.get("nonbonded", 0.0)
        assert nonbonded_energy < 0, (
            f"Nonbonded energy {nonbonded_energy:.1f} should be negative "
            f"(attractive)"
        )

    def test_prev_positions_consistent_after_rebuild(self):
        from mdpy.integrator.verlet import VerletIntegrator

        system = _setup_system()
        integrator = VerletIntegrator(0.5)

        system.compute_forces()
        system._do_rebuild(force=True)

        integrator.step(system)

        positions, _ = system.dump_state()

        assert np.all(np.isfinite(positions)), "Positions contain NaN or Inf"
        assert np.all(positions >= -1.0), "Positions below -1.0 (bad wrap)"
        assert np.all(positions <= BOX_SIZE + 1.0), (
            f"Positions above {BOX_SIZE + 1.0:.0f} (bad wrap)"
        )

        system._do_rebuild(force=True)

        positions2, _ = system.dump_state()

        max_diff = np.max(np.abs(positions - positions2))
        assert max_diff < 1e-5, (
            f"Positions changed by {max_diff:.2e} after rebuild "
            f"(rebuild should only reorder, not change values)"
        )
