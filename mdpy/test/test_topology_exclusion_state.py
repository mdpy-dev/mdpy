import numpy as np
import cupy as cp
from mdpy.core.topology import Topology


def _topo_with_bonds():
    topology = Topology()
    topology.num_particles = 6
    topology.add_bond(0, 1)
    topology.add_bond(1, 2)
    topology.add_bond(2, 3)
    topology.add_angle(0, 1, 2)
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
