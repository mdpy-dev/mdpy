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
