import numpy as np
from mdpy.core.pbc import (
    check_pbc_matrix, wrap_positions, compute_pbc_inv,
)


def test_check_pbc_matrix_cubic():
    box = np.eye(3) * 10.0
    result = check_pbc_matrix(box)
    assert result.dtype in (np.float32, np.float64)


def test_check_pbc_matrix_triclinic():
    box = np.array([[10, 0, 0], [2, 8, 0], [1, 1, 12]], dtype=float)
    result = check_pbc_matrix(box)
    assert result.shape == (3, 3)


def test_check_pbc_matrix_rejects_singular():
    box = np.array([[10, 0, 0], [10, 0, 0], [0, 0, 10]], dtype=float)
    try:
        check_pbc_matrix(box)
        assert False, 'should raise'
    except ValueError:
        pass


def test_check_pbc_matrix_rejects_wrong_shape():
    try:
        check_pbc_matrix(np.ones((2, 3)))
        assert False, 'should raise'
    except ValueError:
        pass


def test_wrap_positions_cubic():
    box = np.eye(3) * 10.0
    box_inv = np.linalg.inv(box)
    positions = np.array([[11.0, -1.0, 5.0]])
    wrapped = wrap_positions(positions, box, box_inv)
    np.testing.assert_allclose(wrapped, [[1.0, -1.0, 5.0]], atol=1e-5)


def test_wrap_positions_multiple():
    box = np.diag([10.0, 10.0, 10.0])
    box_inv = np.linalg.inv(box)
    positions = np.array([
        [3.0, 3.0, 3.0],
        [12.0, 5.0, -2.0],
    ])
    wrapped = wrap_positions(positions, box, box_inv)
    np.testing.assert_allclose(wrapped[0], [3, 3, 3], atol=1e-5)
    np.testing.assert_allclose(wrapped[1], [2, 5, -2], atol=1e-5)


def test_compute_pbc_inv():
    box = np.array([[10, 0, 0], [2, 8, 0], [1, 1, 12]], dtype=np.float64)
    inv = compute_pbc_inv(box)
    identity = box @ inv
    np.testing.assert_allclose(identity, np.eye(3), atol=1e-4)

