import numpy as np
import cupy as cp
import pytest
from mdpy.core.topology import Builder
from mdpy.core.gpu_context import GPUContext
from mdpy.constraint.lincs import LincsConstraint


def _make_ethane_system():
    masses = np.array([12.0, 1.0, 1.0, 1.0, 12.0, 1.0, 1.0, 1.0], dtype=np.float32)
    positions = np.array([
        [5.0, 5.0, 5.0],
        [5.0, 5.0, 6.09],
        [5.0, 6.09, 5.0],
        [6.09, 5.0, 5.0],
        [6.54, 5.0, 5.0],
        [6.54, 5.0, 6.09],
        [6.54, 6.09, 5.0],
        [7.63, 5.0, 5.0],
    ], dtype=np.float32)
    constraint_pairs = [
        (0, 1), (0, 2), (0, 3),
        (4, 5), (4, 6), (4, 7),
        (0, 4),
    ]
    target_lengths = [1.09, 1.09, 1.09, 1.09, 1.09, 1.09, 1.54]
    mol_ids = np.zeros(8, dtype=np.int32)
    pbc_matrix = np.diag([20.0, 20.0, 20.0]).astype(np.float32)
    return constraint_pairs, target_lengths, masses, positions, mol_ids, pbc_matrix


def test_lincs_preserves_bond_lengths():
    np.random.seed(42)
    pairs, lengths, masses, positions, mol_ids, pbc = _make_ethane_system()
    lincs = LincsConstraint(pairs, lengths, masses, expansion_order=4, num_iterations=1)
    gpu = GPUContext()
    topology = Builder().set_particles(
        masses,
        np.zeros(len(masses), dtype=np.float32),
        np.zeros(len(masses), dtype=np.int32),
        mol_ids,
    ).build()[0]
    gpu.initialize(topology, pbc.flatten())
    gpu.upload_positions(positions)
    gpu.upload_prev_positions(positions.copy())
    perturbed = positions + np.random.randn(*positions.shape).astype(np.float32) * 0.005
    gpu.d_positions_x[:] = cp.asarray(perturbed[:, 0])
    gpu.d_positions_y[:] = cp.asarray(perturbed[:, 1])
    gpu.d_positions_z[:] = cp.asarray(perturbed[:, 2])
    lincs.apply(gpu, 0.002)
    corrected = gpu.download_positions()
    for c, (i, j) in enumerate(pairs):
        d = np.linalg.norm(corrected[i] - corrected[j])
        assert abs(d - lengths[c]) < 0.01, f"Bond {c} ({i}-{j}): {d} != {lengths[c]}"


def test_lincs_no_change_if_already_correct():
    np.random.seed(42)
    pairs, lengths, masses, positions, mol_ids, pbc = _make_ethane_system()
    lincs = LincsConstraint(pairs, lengths, masses, expansion_order=4, num_iterations=1)
    gpu = GPUContext()
    topology = Builder().set_particles(
        masses,
        np.zeros(len(masses), dtype=np.float32),
        np.zeros(len(masses), dtype=np.int32),
        mol_ids,
    ).build()[0]
    gpu.initialize(topology, pbc.flatten())
    gpu.upload_positions(positions)
    gpu.upload_prev_positions(positions.copy())
    lincs.apply(gpu, 0.002)
    corrected = gpu.download_positions()
    np.testing.assert_allclose(corrected, positions, atol=1e-3)
