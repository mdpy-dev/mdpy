"""Tests for mdpy.barostat — Monte Carlo barostat and supporting infrastructure."""

import numpy as np
import cupy as cp
import pytest

from mdpy import precision
from mdpy.core.topology import Topology
from mdpy.core.state import State
from mdpy.core.parameter_set import ParameterSet
from mdpy.force.bonded_force import BondedForce
from mdpy.force.nonbonded_transpiler import nonbonded_expression
from mdpy.force.markers import param
from mdpy.force.expressions.geometry import distance
from mdpy.force.nonbonded_force import NonbondedForce
from mdpy.force.pme_reciprocal_force import PMEReciprocalForce, precompute_bk_factors
from mdpy.system import System


@pytest.fixture(autouse=True)
def _sync_gpu():
    yield
    cp.cuda.Stream.null.synchronize()


def _make_pme(cutoff=12.0, fourier_spacing=1.0):
    """Create a minimal PMEReciprocalForce for testing."""
    topology = Topology()
    topology.num_particles = 2
    topology.add_bond(0, 1)
    pt = ParameterSet()
    pbc = np.eye(3, dtype=np.float64) * 50.0
    pme = PMEReciprocalForce(cutoff, fourier_spacing=fourier_spacing)
    pme.initialize_grid(topology, pt, pbc_matrix=pbc)
    return pme


class TestPMEUpdateBox:
    def test_bk_factors_change_after_update_box(self):
        pme = _make_pme()
        original_bk = pme._d_bk_factors.copy()

        pme.update_box(60.0, 60.0, 60.0, block_list=None)

        new_bk = pme._d_bk_factors
        assert not cp.allclose(original_bk, new_bk), \
            "bk_factors must change when box dimensions change"

    def test_bk_factors_match_manual_recompute(self):
        pme = _make_pme()
        new_box_x, new_box_y, new_box_z = 55.0, 55.0, 55.0

        pme.update_box(new_box_x, new_box_y, new_box_z, block_list=None)
        expected_bk = precompute_bk_factors(
            pme.alpha, pme.grid_x, pme.grid_y, pme.grid_z,
            pme.order, new_box_x, new_box_y, new_box_z)

        assert cp.allclose(pme._d_bk_factors, cp.asarray(expected_bk))

    def test_subgrid_not_invalidated_when_num_cells_unchanged(self):
        """When block_list.num_cells hasn't changed, subgrid cache stays valid."""
        pme = _make_pme()

        # Build a fake block_list-like object with stable num_cells
        class FakeBlockList:
            num_cells_x = 5
            num_cells_y = 5
            num_cells_z = 5

        bl = FakeBlockList()

        # Simulate subgrid initialization by running through the lazy init path
        pme._subgrid_dx = -(-pme.grid_x // 5) + 2 * pme.order
        pme._subgrid_dy = -(-pme.grid_y // 5) + 2 * pme.order
        pme._subgrid_dz = -(-pme.grid_z // 5) + 2 * pme.order
        pme._subgrid_initialized = True

        pme.update_box(66.0, 66.0, 66.0, block_list=bl)

        assert pme._subgrid_initialized is True, \
            "subgrid should NOT be invalidated when num_cells hasn't changed"

    def test_subgrid_invalidated_when_num_cells_changes(self):
        """When block_list.num_cells changes, subgrid cache must be invalidated."""
        pme = _make_pme()

        class FakeBlockList:
            num_cells_x = 3  # Different from original (was 5 in setup)
            num_cells_y = 3
            num_cells_z = 3

        bl = FakeBlockList()

        # Set up initial subgrid as if computed with num_cells=5
        pme._subgrid_dx = -(-pme.grid_x // 5) + 2 * pme.order
        pme._subgrid_dy = -(-pme.grid_y // 5) + 2 * pme.order
        pme._subgrid_dz = -(-pme.grid_z // 5) + 2 * pme.order
        pme._subgrid_initialized = True

        # update_box with block_list that has num_cells=3 — should invalidate
        pme.update_box(40.0, 40.0, 40.0, block_list=bl)

        assert pme._subgrid_initialized is False, \
            "subgrid MUST be invalidated when num_cells changed"
