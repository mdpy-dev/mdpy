import numpy as np
import pytest

from mdpy.core.gpu_context import GPUContext
from mdpy.core.topology import Builder


def _make_topology(n=4):
    return Builder().set_particles(
        np.ones(n, dtype=np.float32),
        np.zeros(n, dtype=np.int32),
        np.zeros(n, dtype=np.int32),
    ).build()[0]


def test_gpu_context_exposes_public_pbc_properties():
    topo = _make_topology(4)
    ctx = GPUContext()
    ctx.initialize(topo, np.diag(np.array([10.0, 20.0, 30.0], dtype=np.float32)).flatten())

    assert ctx.box_x == pytest.approx(10.0)
    assert ctx.box_y == pytest.approx(20.0)
    assert ctx.box_z == pytest.approx(30.0)
    assert ctx.inv_box_x == pytest.approx(0.1)
    assert ctx.inv_box_y == pytest.approx(0.05)
    assert ctx.inv_box_z == pytest.approx(1.0 / 30.0)


def test_gpu_context_pbc_properties_update_after_upload_pbc():
    topo = _make_topology(4)
    ctx = GPUContext()
    ctx.initialize(topo, np.diag(np.array([10.0, 20.0, 30.0], dtype=np.float32)).flatten())
    ctx.upload_pbc(np.diag(np.array([40.0, 50.0, 60.0], dtype=np.float32)).flatten())

    assert ctx.box_x == pytest.approx(40.0)
    assert ctx.box_y == pytest.approx(50.0)
    assert ctx.box_z == pytest.approx(60.0)


def test_block_list_uses_current_pbc_after_box_change():
    """After explicit re-rebuild with a larger box, BlockList cell count must
    reflect the new box (cutoff fixed).

    Regression guard for the PBC ownership refactor (Task 5): BlockList will
    stop owning PBC and read d_pbc_matrix from GPUContext. This test must keep
    passing after the signature change.
    """
    from mdpy.core.block_list import BlockList

    topo = _make_topology(64)
    rng = np.random.default_rng(42)
    positions = rng.uniform(0, 10, (64, 3)).astype(np.float32)

    ctx = GPUContext()
    ctx.initialize(topo, (np.eye(3, dtype=np.float32) * 10.0).flatten())
    ctx.upload_positions(positions)
    ctx.upload_velocities(np.zeros((64, 3), dtype=np.float32))

    bl = BlockList(cutoff=4.0, skin=1.0, rebuild_check_interval=1)
    bl.rebuild(
        (ctx.d_positions_x, ctx.d_positions_y, ctx.d_positions_z),
        topo,
        ctx,
        force=True,
    )
    nc_x_before = bl.nc_x

    # Now upload a new box that is 2x larger in each dimension.
    ctx.upload_pbc((np.eye(3, dtype=np.float32) * 20.0).flatten())

    # Re-rebuild reading PBC from GPUContext. The signature change must not
    # break the physical behavior: BlockList reads the current box.
    bl.rebuild(
        (ctx.d_positions_x, ctx.d_positions_y, ctx.d_positions_z),
        topo,
        ctx,
        force=True,
    )
    nc_x_after = bl.nc_x

    # Cell count scales with box length (cutoff fixed); doubling the box
    # should roughly double nc_x. If this assertion fails after Task 5,
    # BlockList is reading stale PBC.
    assert nc_x_after > nc_x_before, (
        f"BlockList did not pick up the new PBC: nc_x {nc_x_before} -> {nc_x_after}"
    )
