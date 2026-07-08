import numpy as np
import cupy as cp
from mdpy.core.topology import Topology, Builder


def _topo_with_bonds():
    builder = Builder()
    builder.set_particles(
        masses=np.ones(6, dtype=np.float32),
        charges=np.zeros(6, dtype=np.float32),
        particle_type_indices=np.zeros(6, dtype=np.int32),
    )
    builder.add_bond(0, 1, 0.0, 0.0)
    builder.add_bond(1, 2, 0.0, 0.0)
    builder.add_bond(2, 3, 0.0, 0.0)
    builder.add_angle(0, 1, 2, 0.0, 0.0)   # -> 0-3 is a 1-3 exclusion
    topology, _ = builder.build()
    return topology


def test_lazy_not_built_until_read():
    topo = _topo_with_bonds()
    assert topo._exclusion_dirty is True
    assert topo._exclusion_csr is None  # not built yet


def test_first_read_builds_and_caches():
    topo = _topo_with_bonds()
    offset, neighbors = topo.exclusion_csr
    assert topo._exclusion_dirty is False
    assert offset.shape[0] == topo.num_particles + 1
    # cached: reading again returns the SAME arrays (no recompute)
    offset2, _ = topo.exclusion_csr
    assert offset2 is offset


def test_invalidate_forces_rebuild():
    topo = _topo_with_bonds()
    offset, _ = topo.exclusion_csr
    topo.invalidate_exclusions()
    assert topo._exclusion_dirty is True
    offset2, _ = topo.exclusion_csr      # rebuilds
    assert topo._exclusion_dirty is False
    # values identical (bond graph unchanged)
    np.testing.assert_array_equal(cp.asnumpy(offset), cp.asnumpy(offset2))


def test_reverse_csr_available():
    topo = _topo_with_bonds()
    rev_offset, rev_neighbors = topo.exclusion_reverse_csr
    assert rev_offset.shape[0] == topo.num_particles + 1
