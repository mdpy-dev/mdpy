import numpy as np
import cupy as cp
from mdpy.core.topology import Topology, Builder


def _get_neighbors(topology, particle_index):
    offset, neighbors, scale = topology.exclusion_csr
    start = int(offset[particle_index].get())
    end = int(offset[particle_index + 1].get())
    return (
        cp.asnumpy(neighbors[start:end]),
        cp.asnumpy(scale[start:end]),
    )


def _simple_builder() -> Builder:
    builder = Builder()
    builder.set_particles(
        masses=np.ones(4, dtype=np.float32),
        charges=np.zeros(4, dtype=np.float32),
        particle_type_indices=np.array([0, 1, 1, 0], dtype=np.int32),
    )
    return builder


def test_builder_set_particles():
    builder = _simple_builder()
    assert builder._num_particles == 4
    assert len(builder._masses) == 4


def test_builder_add_bond():
    builder = _simple_builder()
    builder.add_bond(0, 1, k=305.0, r0=1.5)
    assert len(builder._bonds) == 1
    assert builder._bonds[0] == [0, 1, 305.0, 1.5]


def test_builder_add_angle():
    builder = _simple_builder()
    builder.add_angle(0, 1, 2, force_constant=50.0, equilibrium_angle=1.9)
    assert len(builder._angles) == 1
    assert builder._angles[0][:3] == [0, 1, 2]


def test_builder_add_dihedral():
    builder = _simple_builder()
    builder.add_dihedral(0, 1, 2, 3, force_constant=0.5, periodicity=3, phase=0.0)
    assert len(builder._dihedrals) == 1


def test_builder_add_improper():
    builder = _simple_builder()
    builder.add_improper(0, 1, 2, 3, force_constant=10.0, equilibrium_angle=0.0)
    assert len(builder._impropers) == 1


def test_build_topology():
    builder = _simple_builder()
    builder.add_bond(0, 1, k=305.0, r0=1.5)
    builder.add_bond(1, 2, k=310.0, r0=1.4)
    builder.add_angle(0, 1, 2, force_constant=50.0, equilibrium_angle=1.9)
    builder.add_dihedral(0, 1, 2, 3, force_constant=0.5, periodicity=3, phase=0.0)
    topology, _ = builder.build()
    assert topology.num_particles == 4
    assert topology.num_bonds == 2
    assert topology.num_angles == 1
    assert topology.num_dihedrals == 1
    assert topology.num_impropers == 0


def test_topology_arrays_dtype():
    builder = _simple_builder()
    builder.add_bond(0, 1, k=305.0, r0=1.5)
    topology, term_params = builder.build()
    from mdpy import env
    assert topology.bond_indices.dtype == env.NUMPY_INT
    assert term_params['bond'].dtype == env.NUMPY_FLOAT


def test_topology_empty_terms():
    builder = _simple_builder()
    topology, _ = builder.build()
    assert topology.num_bonds == 0
    assert topology.bond_indices.shape == (0, 2)
    assert topology.num_angles == 0
    assert topology.num_dihedrals == 0
    assert topology.num_impropers == 0


def test_exclusion_map_basic():
    builder = _simple_builder()
    builder.add_bond(0, 1, k=305.0, r0=1.5)
    builder.add_bond(1, 2, k=310.0, r0=1.4)
    builder.add_bond(2, 3, k=300.0, r0=1.5)
    builder.add_angle(0, 1, 2, force_constant=50.0, equilibrium_angle=1.9)
    builder.add_angle(1, 2, 3, force_constant=50.0, equilibrium_angle=1.9)
    builder.add_dihedral(0, 1, 2, 3, force_constant=0.5, periodicity=3, phase=0.0)
    topology, _ = builder.build()

    neighbors_0, scales_0 = _get_neighbors(topology,0)
    assert 1 in neighbors_0
    assert 2 in neighbors_0
    assert 3 in neighbors_0

    idx_01 = np.where(neighbors_0 == 1)[0][0]
    idx_02 = np.where(neighbors_0 == 2)[0][0]
    idx_03 = np.where(neighbors_0 == 3)[0][0]
    assert scales_0[idx_01] == 0.0
    assert scales_0[idx_02] == 0.0
    assert scales_0[idx_03] == 0.0


def test_exclusion_map_symmetry():
    builder = _simple_builder()
    builder.add_bond(0, 1, k=305.0, r0=1.5)
    topology, _ = builder.build()

    neighbors_0, _ = _get_neighbors(topology,0)
    neighbors_1, _ = _get_neighbors(topology,1)
    assert 1 in neighbors_0
    assert 0 in neighbors_1


def test_exclusion_map_no_interactions():
    builder = _simple_builder()
    topology, _ = builder.build()
    for particle_index in range(4):
        neighbors, _ = _get_neighbors(topology,particle_index)
        assert len(neighbors) == 0


def test_add_batch_indices():
    builder = _simple_builder()
    indices = np.array([[0, 1], [1, 2]], dtype=np.int32)
    parameters = np.array([[305.0, 1.5], [310.0, 1.4]], dtype=np.float32)
    builder.add_bond_indices(indices, parameters)
    assert len(builder._bonds) == 2
    assert builder._bonds[0] == [0, 1, 305.0, 1.5]


def test_repr():
    builder = _simple_builder()
    builder.add_bond(0, 1, k=305.0, r0=1.5)
    topology, _ = builder.build()
    text = repr(topology)
    assert '4 particles' in text
    assert '1 bonds' in text


def test_build_without_particles_raises():
    builder = Builder()
    try:
        builder.build()
        assert False, 'should raise'
    except ValueError:
        pass
