from __future__ import annotations

import numpy as np
import cupy as cp
from mdpy import env

_PBC_WRAP_KERNEL = r"""
extern "C" __global__
void pbc_wrap_kernel(
    float* positions,
    const float* pbc_matrix,
    const float* pbc_inv,
    int number_particles
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= number_particles) return;

    float px = positions[index * 3 + 0];
    float py = positions[index * 3 + 1];
    float pz = positions[index * 3 + 2];

    float fx = px * pbc_inv[0] + py * pbc_inv[3] + pz * pbc_inv[6];
    float fy = px * pbc_inv[1] + py * pbc_inv[4] + pz * pbc_inv[7];
    float fz = px * pbc_inv[2] + py * pbc_inv[5] + pz * pbc_inv[8];

    fx = fx - floorf(fx);
    fy = fy - floorf(fy);
    fz = fz - floorf(fz);

    positions[index * 3 + 0] = fx * pbc_matrix[0] + fy * pbc_matrix[3] + fz * pbc_matrix[6];
    positions[index * 3 + 1] = fx * pbc_matrix[1] + fy * pbc_matrix[4] + fz * pbc_matrix[7];
    positions[index * 3 + 2] = fx * pbc_matrix[2] + fy * pbc_matrix[5] + fz * pbc_matrix[8];
}
"""


class GPUContext:

    _pbc_wrap_kernel = None

    def __init__(self):
        self.number_particles = 0

        self.d_positions = None
        self.d_velocities = None
        self.d_forces = None
        self.d_prev_positions = None
        self.d_masses = None
        self.d_types = None
        self.d_energy = None

        self.d_pbc_matrix = None
        self.d_pbc_inv = None

        self.d_box_dims = None

        self._box_x = 0.0
        self._box_y = 0.0
        self._box_z = 0.0
        self._inv_box_x = 0.0
        self._inv_box_y = 0.0
        self._inv_box_z = 0.0

    @classmethod
    def _ensure_kernels(cls):
        if cls._pbc_wrap_kernel is None:
            cls._pbc_wrap_kernel = cp.RawKernel(_PBC_WRAP_KERNEL, 'pbc_wrap_kernel')

    def initialize(self, topology, pbc_matrix):
        self.number_particles = topology.num_particles
        number = self.number_particles

        float_dtype = np.float32
        int_dtype = np.int32

        self.d_positions = cp.zeros(number * 3, dtype=float_dtype)
        self.d_velocities = cp.zeros(number * 3, dtype=float_dtype)
        self.d_forces = cp.zeros(number * 3, dtype=float_dtype)
        self.d_prev_positions = cp.zeros(number * 3, dtype=float_dtype)
        self.d_masses = cp.asarray(
            topology.masses.astype(float_dtype)
        )
        self.d_types = cp.asarray(
            topology.particle_types.astype(int_dtype)
        )
        self.d_energy = cp.zeros(1, dtype=float_dtype)
        self.d_energy_accumulator = None

        pbc_flat = np.ascontiguousarray(pbc_matrix, dtype=float_dtype).ravel()
        self.d_pbc_matrix = cp.asarray(pbc_flat)
        pbc_inv = np.linalg.inv(pbc_flat.reshape(3, 3))
        self.d_pbc_inv = cp.asarray(
            np.ascontiguousarray(pbc_inv, dtype=float_dtype).ravel()
        )

        self.d_box_dims = cp.zeros(6, dtype=float_dtype)

        self._ensure_kernels()

    def wrap_positions(self):
        number = self.number_particles
        tpb = 256
        self._pbc_wrap_kernel(
            ((number + tpb - 1) // tpb,), (tpb,),
            (self.d_positions, self.d_pbc_matrix, self.d_pbc_inv, np.int32(number)),
        )

    def upload_positions(self, particle_table):
        data = np.ascontiguousarray(
            particle_table.positions.astype(np.float32).ravel()
        )
        self.d_positions[:] = cp.asarray(data)

    def upload_velocities(self, particle_table):
        data = np.ascontiguousarray(
            particle_table.velocities.astype(np.float32).ravel()
        )
        self.d_velocities[:] = cp.asarray(data)

    def download_positions(self, particle_table):
        particle_table.positions[:] = self.d_positions.get().reshape(-1, 3)

    def download_velocities(self, particle_table):
        particle_table.velocities[:] = self.d_velocities.get().reshape(-1, 3)

    def download_forces(self, particle_table):
        particle_table.forces[:] = self.d_forces.get().reshape(-1, 3)

    def zero_forces(self):
        self.d_forces[:] = 0

    def zero_energy(self):
        self.d_energy[:] = 0

    def allocate_energy_accumulator(self, num_terms):
        self.d_energy_accumulator = cp.zeros(num_terms, dtype=np.float32)

    def accumulate_energy(self, term_index):
        self.d_energy_accumulator[term_index] = self.d_energy[0]

    def get_positions_2d(self):
        return self.d_positions.reshape(-1, 3)

    def set_box_dims(self, box_x, box_y, box_z):
        self._box_x = float(box_x)
        self._box_y = float(box_y)
        self._box_z = float(box_z)
        self._inv_box_x = 1.0 / self._box_x
        self._inv_box_y = 1.0 / self._box_y
        self._inv_box_z = 1.0 / self._box_z
        self.d_box_dims[:] = cp.array([
            self._box_x, self._box_y, self._box_z,
            self._inv_box_x, self._inv_box_y, self._inv_box_z
        ], dtype=np.float32)
