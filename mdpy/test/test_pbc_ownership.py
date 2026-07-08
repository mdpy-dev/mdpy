import numpy as np
import pytest

from mdpy.core.state import State
from mdpy.core.topology import Topology


def _make_topology(n=4):
    topology = Topology()
    topology.num_particles = n
    return topology


def test_state_exposes_public_pbc_properties():
    topo = _make_topology(4)
    ctx = State(topo.num_particles)
    ctx.set_pbc(np.diag(np.array([10.0, 20.0, 30.0], dtype=np.float32)).flatten())

    assert ctx.box_x == pytest.approx(10.0)
    assert ctx.box_y == pytest.approx(20.0)
    assert ctx.box_z == pytest.approx(30.0)
    assert ctx.inv_box_x == pytest.approx(0.1)
    assert ctx.inv_box_y == pytest.approx(0.05)
    assert ctx.inv_box_z == pytest.approx(1.0 / 30.0)


def test_state_pbc_properties_update_after_set_pbc():
    topo = _make_topology(4)
    ctx = State(topo.num_particles)
    ctx.set_pbc(np.diag(np.array([10.0, 20.0, 30.0], dtype=np.float32)).flatten())
    ctx.set_pbc(np.diag(np.array([40.0, 50.0, 60.0], dtype=np.float32)).flatten())

    assert ctx.box_x == pytest.approx(40.0)
    assert ctx.box_y == pytest.approx(50.0)
    assert ctx.box_z == pytest.approx(60.0)


def test_block_list_uses_current_pbc_after_box_change():
    """After explicit re-rebuild with a larger box, BlockList cell count must
    reflect the new box (cutoff fixed).

    Regression guard for the PBC ownership refactor (Task 5): BlockList will
    stop owning PBC and read d_pbc_matrix from State. This test must keep
    passing after the signature change.
    """
    from mdpy.core.block_list import BlockList

    topo = _make_topology(64)
    rng = np.random.default_rng(42)
    positions = rng.uniform(0, 10, (64, 3)).astype(np.float32)

    ctx = State(topo.num_particles)
    ctx.set_pbc((np.eye(3, dtype=np.float32) * 10.0).flatten())
    ctx.set_positions(positions)
    ctx.set_velocities(np.zeros((64, 3), dtype=np.float32))

    bl = BlockList(cutoff=4.0, skin=1.0, rebuild_check_interval=1)
    bl.rebuild(
        topo,
        ctx,
        force=True,
    )
    nc_x_before = bl.num_cells_x

    # Now upload a new box that is 2x larger in each dimension.
    ctx.set_pbc((np.eye(3, dtype=np.float32) * 20.0).flatten())

    # Re-rebuild reading PBC from State. The signature change must not
    # break the physical behavior: BlockList reads the current box.
    bl.rebuild(
        topo,
        ctx,
        force=True,
    )
    nc_x_after = bl.num_cells_x

    # Cell count scales with box length (cutoff fixed); doubling the box
    # should roughly double nc_x. If this assertion fails after Task 5,
    # BlockList is reading stale PBC.
    assert nc_x_after > nc_x_before, (
        f"BlockList did not pick up the new PBC: nc_x {nc_x_before} -> {nc_x_after}"
    )


def test_system_update_neighbor_list_raises_if_set_pbc_not_called():
    """Regression: System must raise if set_pbc() was never called.

    Before the PBC ownership refactor, the gate checked a host-side cache
    that was only set inside set_pbc. After the refactor, State
    initializes d_pbc_matrix to a non-None identity placeholder, so the
    gate must use an explicit flag.
    """
    from mdpy.system import System

    topo = _make_topology(8)

    system = System(topo)
    # Note: do NOT call system.set_pbc(...)

    # set positions+velocities so the _ensure_ready gate passes.
    rng = np.random.default_rng(0)
    system.set_positions(rng.uniform(0, 5, (8, 3)).astype(np.float32))
    system.set_velocities(np.zeros((8, 3), dtype=np.float32))

    # Need a force term with a cutoff so block_list can be constructed.
    # Build a minimal stand-in force term that has a _cutoff attribute.
    class _FakeForce:
        _cutoff = 4.0
        name = "fake"

    system.add_force_term(_FakeForce())

    with pytest.raises(RuntimeError, match="PBC not set"):
        system.update_neighbor_list()


def test_state_lazy_pbc():
    """State constructed without PBC; has_pbc False until set_pbc."""
    topo = _make_topology(4)
    ctx = State(topo.num_particles)
    assert ctx.has_pbc is False
    assert ctx.d_pbc_matrix is None
    assert ctx.d_pbc_inv is None
    ctx.set_pbc(np.diag([10.0, 10.0, 10.0]).astype(np.float32))
    assert ctx.has_pbc is True
    assert ctx.d_pbc_matrix is not None
    buf_ptr = ctx.d_pbc_matrix.data.ptr
    ctx.set_pbc(np.diag([20.0, 20.0, 20.0]).astype(np.float32))
    assert ctx.d_pbc_matrix.data.ptr == buf_ptr  # in-place overwrite
