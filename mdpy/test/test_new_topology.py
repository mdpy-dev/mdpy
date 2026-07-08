import numpy as np
import cupy as cp
from mdpy.core.topology import Topology


def _get_neighbors(topology, particle_index):
    offset, neighbors = topology.exclusion_csr
    start = int(offset[particle_index].get())
    end = int(offset[particle_index + 1].get())
    return cp.asnumpy(neighbors[start:end])


def _simple_topology() -> Topology:
    topology = Topology()
    topology.num_particles = 4
    return topology


def test_add_bond():
    topology = _simple_topology()
    topology.add_bond(0, 1)
    assert topology.num_bonds == 1
    assert topology.bond_indices[0, 0] == 0
    assert topology.bond_indices[0, 1] == 1


def test_add_angle():
    topology = _simple_topology()
    topology.add_angle(0, 1, 2)
    assert topology.num_angles == 1
    assert list(topology.angle_indices[0]) == [0, 1, 2]


def test_add_dihedral():
    topology = _simple_topology()
    topology.add_dihedral(0, 1, 2, 3)
    assert topology.num_dihedrals == 1


def test_add_improper():
    topology = _simple_topology()
    topology.add_improper(0, 1, 2, 3)
    assert topology.num_impropers == 1


def test_build_topology():
    topology = _simple_topology()
    topology.add_bond(0, 1)
    topology.add_bond(1, 2)
    topology.add_angle(0, 1, 2)
    topology.add_dihedral(0, 1, 2, 3)
    assert topology.num_particles == 4
    assert topology.num_bonds == 2
    assert topology.num_angles == 1
    assert topology.num_dihedrals == 1
    assert topology.num_impropers == 0


def test_topology_arrays_dtype():
    topology = _simple_topology()
    topology.add_bond(0, 1)
    from mdpy import precision
    assert topology.bond_indices.dtype == precision.INT


def test_topology_empty_terms():
    topology = _simple_topology()
    assert topology.num_bonds == 0
    assert topology.bond_indices.shape == (0, 2)
    assert topology.num_angles == 0
    assert topology.num_dihedrals == 0
    assert topology.num_impropers == 0


def test_exclusion_map_basic():
    topology = _simple_topology()
    topology.add_bond(0, 1)
    topology.add_bond(1, 2)
    topology.add_bond(2, 3)
    topology.add_angle(0, 1, 2)
    topology.add_angle(1, 2, 3)
    topology.add_dihedral(0, 1, 2, 3)

    neighbors_0 = _get_neighbors(topology, 0)
    assert 1 in neighbors_0
    assert 2 in neighbors_0
    assert 3 in neighbors_0


def test_exclusion_map_symmetry():
    topology = _simple_topology()
    topology.add_bond(0, 1)

    neighbors_0 = _get_neighbors(topology, 0)
    neighbors_1 = _get_neighbors(topology, 1)
    assert 1 in neighbors_0
    assert 0 in neighbors_1


def test_exclusion_map_no_interactions():
    topology = _simple_topology()
    for particle_index in range(4):
        neighbors = _get_neighbors(topology, particle_index)
        assert len(neighbors) == 0


def test_repr():
    topology = _simple_topology()
    topology.add_bond(0, 1)
    text = repr(topology)
    assert '4 particles' in text
    assert '1 bonds' in text


def test_numpy_cache_invalidation():
    topology = _simple_topology()
    topology.add_bond(0, 1)
    arr1 = topology.bond_indices
    topology.add_bond(1, 2)
    arr2 = topology.bond_indices
    assert arr1 is not arr2
    assert topology.num_bonds == 2
    assert arr2.shape == (2, 2)
