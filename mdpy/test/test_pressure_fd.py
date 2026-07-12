import numpy as np
import pytest
import cupy as cp
from mdpy.core.state import State
from mdpy.core.topology import Topology
from mdpy.system import System


def _make_water_system(N_mol=4, box=30.0):
    """Build a small system of 'water-like' 3-atom molecules."""
    N = N_mol * 3
    topo = Topology(); topo.num_particles = N
    state = State(N)
    state.set_pbc(np.diag([box, box, box]).astype(np.float32))
    positions = np.zeros((N, 3), dtype=np.float32)
    mol_ids = np.zeros(N, dtype=np.int32)
    for m in range(N_mol):
        cx = (m + 0.5) * box / N_mol
        positions[m*3] = [cx, box/2, box/2]
        positions[m*3+1] = [cx + 0.96, box/2, box/2]
        positions[m*3+2] = [cx + 0.31, box/2 + 0.92, box/2]
        mol_ids[m*3:m*3+3] = m
    state.set_positions(positions)
    state.set_particle_charges(np.zeros(N, dtype=np.float32))
    state.set_particle_masses(np.ones(N, dtype=np.float32))
    state.set_particle_type_indices(np.zeros(N, dtype=np.int32))
    state.set_particle_molecule_ids(mol_ids)
    state.set_velocities(np.zeros((N, 3), dtype=np.float32))
    system = System(topo, state=state)
    return system


def test_molecule_csr_built():
    """System should build molecule CSR from particle_molecule_ids."""
    system = _make_water_system(N_mol=4)
    system._ensure_molecule_csr()
    assert system._num_molecules == 4
    assert len(system._mol_atoms) == 12
    assert len(system._mol_start_index) == 5
