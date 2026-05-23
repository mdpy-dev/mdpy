import numpy as np
from mdpy import env


def check_pbc_matrix(pbc_matrix: np.ndarray) -> np.ndarray:
    row, col = pbc_matrix.shape
    if row != 3 or col != 3:
        raise ValueError(
            'pbc_matrix should have shape [3, 3], got [%d, %d]' % (row, col)
        )
    if np.linalg.det(pbc_matrix) == 0:
        raise ValueError(
            'pbc_matrix is singular (column vectors are linearly dependent)'
        )
    return pbc_matrix.astype(env.NUMPY_FLOAT)


def wrap_positions(
    positions: np.ndarray,
    pbc_matrix: np.ndarray,
    pbc_inv: np.ndarray,
) -> np.ndarray:
    move_vector = -np.round(positions @ pbc_inv)
    if np.max(np.abs(move_vector)) >= 2:
        particle_indices = np.unique(
            [row[0] for row in np.argwhere(np.abs(move_vector) >= 2)]
        )
        raise RuntimeError(
            'Particle(s) with id %s moved beyond 2 PBC images' % particle_indices
        )
    return positions + move_vector @ pbc_matrix


def compute_pbc_inv(pbc_matrix: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        np.linalg.inv(pbc_matrix), dtype=env.NUMPY_FLOAT
    )


def minimum_image(
    delta: np.ndarray, pbc_matrix: np.ndarray, pbc_inv: np.ndarray
) -> np.ndarray:
    scaled = delta @ pbc_inv
    rounded = np.round(scaled)
    scaled -= rounded
    return scaled @ pbc_matrix
