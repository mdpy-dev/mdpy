"""Verify PME spread/gather optimization produces identical energies and forces."""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BENCH_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "benchmark")
REF_PATH = os.path.join(BENCH_DIR, "pme_reference.npz")


@pytest.fixture(scope="module")
def ion_system():
    from mdpy.core.state import State
    from mdpy.system import System
    from mdpy.io.psf_parser import PSFParser
    from mdpy.io.pdb_parser import PDBParser
    from mdpy.io.charmm_toppar_parser import CharmmTopparParser
    from mdpy.force.factories.charmm import create_charmm_forces

    data_dir = os.path.join(BENCH_DIR, "data")
    psf = PSFParser(os.path.join(data_dir, "ion.psf"))
    pdb = PDBParser(os.path.join(data_dir, "ion.pdb"))
    toppar = CharmmTopparParser(
        os.path.join(data_dir, "par_sin.prm"),
        os.path.join(data_dir, "par_water.prm"),
    )
    pbc = np.diag([75.450, 77.623, 69.668])

    topology = psf.topology
    parameter_set = toppar.resolve_parameter_set(topology, psf.particle_type_names)
    state = State(topology.num_particles)
    state.set_pbc(pbc)
    state.set_positions(pdb.positions)
    state.set_charges(psf.charges)
    state.set_masses(psf.masses)
    state.set_type_indices(parameter_set.particle_type_indices)
    forces = create_charmm_forces(topology, parameter_set, pbc, cutoff=12.0)
    system = System(topology, state)
    for f in forces['bonded']:
        system.add_force_term(f)
    system.add_force_term(forces['nonbonded'])
    system.add_force_term(forces['pme'], stream='pme')
    system.set_velocities(np.zeros((system.num_particles, 3), dtype=np.float32))
    system.update_neighbor_list(force_rebuild=True)
    system.compute_forces()
    return system


def test_pme_energy_matches_reference(ion_system):
    """PME reciprocal energy must match pre-optimization reference within float32 precision."""
    system = ion_system
    ref = np.load(REF_PATH, allow_pickle=True)
    ref_energy = float(ref["ion_energy"])
    energies = system.dump_energy()
    pme_energy = energies.get("pme_reciprocal", 0.0)
    rel_err = abs(pme_energy - ref_energy) / max(abs(ref_energy), 1e-12)
    assert rel_err < 1e-5, f"PME energy rel err {rel_err:.2e} exceeds 1e-5"


def test_pme_forces_match_reference(ion_system):
    """PME forces must match pre-optimization reference within float32 precision."""
    system = ion_system
    ref = np.load(REF_PATH, allow_pickle=True)
    ref_forces = ref["ion_forces"]
    forces_arr = system.dump_forces()
    max_diff = np.max(np.abs(forces_arr - ref_forces))
    max_force = np.max(np.abs(ref_forces))
    rel_err = max_diff / max(max_force, 1e-12)
    assert rel_err < 1e-4, f"Max force rel err {rel_err:.2e} exceeds 1e-4"
