import numpy as np
import pytest
import cupy as cp
from mdpy.core.state import State
from mdpy.core.topology import Topology
from mdpy.system import System
from mdpy.force.bonded_force import BondedForce
from mdpy.force.bonded_transpiler import bonded_expression
from mdpy.force.nonbonded_force import NonbondedForce
from mdpy.force.nonbonded_transpiler import nonbonded_expression
from mdpy.force.markers import param
from mdpy.core.block_list import BlockList


@bonded_expression(body=2)
def harmonic_bond(p1, p2, k=0.0, r0=0.0):
    r = distance(p1, p2)
    return k * (r - r0) ** 2


@nonbonded_expression
def lj_pair(pos1, pos2, epsilon=param, sigma=param):
    r = distance(pos1, pos2)
    return 4.0 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)


def _build_small_system():
    N = 4
    topo = Topology(); topo.num_particles = N
    state = State(N)
    state.set_pbc(np.diag([20.0, 20.0, 20.0]).astype(np.float32))
    state.set_positions(np.array([
        [0, 0, 0], [2.5, 0, 0], [0, 2.5, 0], [2.5, 2.5, 0]
    ], dtype=np.float32))
    state.set_particle_charges(np.zeros(N, dtype=np.float32))
    state.set_particle_masses(np.ones(N, dtype=np.float32))
    state.set_particle_type_indices(np.zeros(N, dtype=np.int32))
    state.set_velocities(np.zeros((N, 3), dtype=np.float32))
    system = System(topo, state=state)

    bond = BondedForce(harmonic_bond)
    bond.add([0, 1], k=50.0, r0=2.0)
    bond.add([1, 2], k=50.0, r0=2.0)
    system.add_force_term(bond)

    nb = NonbondedForce(lj_pair, cutoff=5.0)
    nb.set_pair_parameter('epsilon', np.array([0.1], dtype=np.float32))
    nb.set_pair_parameter('sigma', np.array([1.0], dtype=np.float32))
    system.add_force_term(nb)

    system.update_neighbor_list(force_rebuild=True)
    return system


def test_dump_virial_returns_per_term_virials():
    system = _build_small_system()
    virials = system.dump_virial()
    assert 'bonded' in virials, "bonded virial missing"
    assert 'nonbonded' in virials, "nonbonded virial missing"
    assert virials['bonded'].shape == (3, 3)
    assert virials['nonbonded'].shape == (3, 3)
    assert not np.allclose(virials['bonded'], 0), "bonded virial is zero"
    assert not np.allclose(virials['nonbonded'], 0), "nonbonded virial is zero"


def test_dump_energy_and_virial_combined():
    system = _build_small_system()
    energies, virials = system.dump_energy_and_virial()
    assert 'bonded' in energies
    assert 'nonbonded' in energies
    assert 'bonded' in virials
    assert 'nonbonded' in virials
    assert isinstance(energies['bonded'], float)
    assert virials['bonded'].shape == (3, 3)


def test_dump_virial_tensors_are_symmetric():
    system = _build_small_system()
    virials = system.dump_virial()
    for name, W in virials.items():
        asymmetry = np.max(np.abs(W - W.T))
        assert asymmetry < 1e-3, f"{name} virial not symmetric: max asymmetry {asymmetry}"
