import numpy as np
import cupy as cp
import pytest
from mdpy.core.topology import Builder
from mdpy.core.gpu_context import GPUContext
from mdpy.constraint.settle import SettleConstraint


def _make_water_system(n_waters=10, box_size=30.0, dOH=1.0, dHH=1.63298):
    masses = []
    mol_ids = []
    water_triplets = []
    positions = np.zeros((n_waters * 3, 3), dtype=np.float32)
    idx = 0
    for w in range(n_waters):
        ow = idx
        hw1 = idx + 1
        hw2 = idx + 2
        cx = np.random.uniform(3.0, box_size - 3.0)
        cy = np.random.uniform(3.0, box_size - 3.0)
        cz = np.random.uniform(3.0, box_size - 3.0)
        half_hh = dHH / 2.0
        height = np.sqrt(dOH**2 - half_hh**2)
        angle = np.random.uniform(0, 2 * np.pi)
        positions[ow] = [cx, cy, cz]
        positions[hw1] = [cx + half_hh * np.cos(angle), cy + height,
                          cz + half_hh * np.sin(angle)]
        positions[hw2] = [cx - half_hh * np.cos(angle), cy + height,
                          cz - half_hh * np.sin(angle)]
        masses.extend([15.999, 1.008, 1.008])
        mol_ids.extend([w, w, w])
        water_triplets.append((ow, hw1, hw2))
        idx += 3
    masses = np.array(masses, dtype=np.float32)
    mol_ids = np.array(mol_ids, dtype=np.int32)
    pbc_matrix = np.diag([box_size, box_size, box_size]).astype(np.float32)
    return water_triplets, masses, mol_ids, positions, pbc_matrix


def test_settle_preserves_bond_lengths():
    np.random.seed(42)
    dOH, dHH = 1.0, 1.63298
    water_triplets, masses, mol_ids, positions, pbc_matrix = _make_water_system(50, 30.0, dOH, dHH)
    settle = SettleConstraint(water_triplets, masses, dOH, dHH)
    gpu = GPUContext()
    topology = Builder().set_particles(
        masses,
        np.zeros(len(masses), dtype=np.float32),
        np.zeros(len(masses), dtype=np.int32),
        mol_ids,
    ).build()[0]
    gpu.initialize(topology, pbc_matrix.flatten())
    gpu.upload_positions(positions)
    gpu.upload_prev_positions(positions.copy())
    perturbed = positions + np.random.randn(*positions.shape).astype(np.float32) * 0.01
    gpu.d_positions_x[:] = cp.asarray(perturbed[:, 0])
    gpu.d_positions_y[:] = cp.asarray(perturbed[:, 1])
    gpu.d_positions_z[:] = cp.asarray(perturbed[:, 2])
    settle.apply(gpu, 0.002)
    corrected = gpu.download_positions()
    for ow, hw1, hw2 in water_triplets:
        d_oh1 = np.linalg.norm(corrected[ow] - corrected[hw1])
        d_oh2 = np.linalg.norm(corrected[ow] - corrected[hw2])
        d_hh = np.linalg.norm(corrected[hw1] - corrected[hw2])
        assert abs(d_oh1 - dOH) < 1e-3, f"O-H1 distance {d_oh1} != {dOH}"
        assert abs(d_oh2 - dOH) < 1e-3, f"O-H2 distance {d_oh2} != {dOH}"
        assert abs(d_hh - dHH) < 1e-3, f"H-H distance {d_hh} != {dHH}"


def test_settle_no_change_if_already_correct():
    np.random.seed(42)
    dOH, dHH = 1.0, 1.63298
    water_triplets, masses, mol_ids, positions, pbc_matrix = _make_water_system(10, 30.0, dOH, dHH)
    settle = SettleConstraint(water_triplets, masses, dOH, dHH)
    gpu = GPUContext()
    topology = Builder().set_particles(
        masses,
        np.zeros(len(masses), dtype=np.float32),
        np.zeros(len(masses), dtype=np.int32),
        mol_ids,
    ).build()[0]
    gpu.initialize(topology, pbc_matrix.flatten())
    gpu.upload_positions(positions)
    gpu.upload_prev_positions(positions.copy())
    settle.apply(gpu, 0.002)
    corrected = gpu.download_positions()
    np.testing.assert_allclose(corrected, positions, atol=1e-4)
