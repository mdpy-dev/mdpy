from __future__ import annotations

import numpy as np
import cupy as cp
from mdpy import env

_PBC_WRAP_KERNEL = r"""
extern "C" __global__
void pbc_wrap_kernel(
    float* __restrict__ pos_x,
    float* __restrict__ pos_y,
    float* __restrict__ pos_z,
    const float* __restrict__ pbc_matrix,
    const float* __restrict__ pbc_inv,
    int number_particles
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= number_particles) return;

    float px = pos_x[index];
    float py = pos_y[index];
    float pz = pos_z[index];

    float fx = px * pbc_inv[0] + py * pbc_inv[3] + pz * pbc_inv[6];
    float fy = px * pbc_inv[1] + py * pbc_inv[4] + pz * pbc_inv[7];
    float fz = px * pbc_inv[2] + py * pbc_inv[5] + pz * pbc_inv[8];

    fx = fx - floorf(fx);
    fy = fy - floorf(fy);
    fz = fz - floorf(fz);

    pos_x[index] = fx * pbc_matrix[0] + fy * pbc_matrix[3] + fz * pbc_matrix[6];
    pos_y[index] = fx * pbc_matrix[1] + fy * pbc_matrix[4] + fz * pbc_matrix[7];
    pos_z[index] = fx * pbc_matrix[2] + fy * pbc_matrix[5] + fz * pbc_matrix[8];
}
"""


class GPUContext:

    _pbc_wrap_kernel = None

    def __init__(self):
        self.number_particles = 0

        self.d_positions_x = None
        self.d_positions_y = None
        self.d_positions_z = None

        self.d_velocities_x = None
        self.d_velocities_y = None
        self.d_velocities_z = None

        self.d_forces_x = None
        self.d_forces_y = None
        self.d_forces_z = None

        self.d_prev_positions_x = None
        self.d_prev_positions_y = None
        self.d_prev_positions_z = None

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

        self.d_positions_x = cp.zeros(number, dtype=float_dtype)
        self.d_positions_y = cp.zeros(number, dtype=float_dtype)
        self.d_positions_z = cp.zeros(number, dtype=float_dtype)

        self.d_velocities_x = cp.zeros(number, dtype=float_dtype)
        self.d_velocities_y = cp.zeros(number, dtype=float_dtype)
        self.d_velocities_z = cp.zeros(number, dtype=float_dtype)

        self.d_forces_x = cp.zeros(number, dtype=float_dtype)
        self.d_forces_y = cp.zeros(number, dtype=float_dtype)
        self.d_forces_z = cp.zeros(number, dtype=float_dtype)

        self.d_prev_positions_x = cp.zeros(number, dtype=float_dtype)
        self.d_prev_positions_y = cp.zeros(number, dtype=float_dtype)
        self.d_prev_positions_z = cp.zeros(number, dtype=float_dtype)

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
            (self.d_positions_x, self.d_positions_y, self.d_positions_z,
             self.d_pbc_matrix, self.d_pbc_inv, np.int32(number)),
        )

    def upload_positions(self, particle_table):
        data = np.ascontiguousarray(
            particle_table.positions.astype(np.float32)
        )
        self.d_positions_x[:] = cp.asarray(data[:, 0])
        self.d_positions_y[:] = cp.asarray(data[:, 1])
        self.d_positions_z[:] = cp.asarray(data[:, 2])

    def upload_velocities(self, particle_table):
        data = np.ascontiguousarray(
            particle_table.velocities.astype(np.float32)
        )
        self.d_velocities_x[:] = cp.asarray(data[:, 0])
        self.d_velocities_y[:] = cp.asarray(data[:, 1])
        self.d_velocities_z[:] = cp.asarray(data[:, 2])

    def download_positions(self, particle_table):
        pos = np.stack([
            self.d_positions_x.get(),
            self.d_positions_y.get(),
            self.d_positions_z.get(),
        ], axis=1)
        particle_table.positions[:] = pos

    def download_velocities(self, particle_table):
        vel = np.stack([
            self.d_velocities_x.get(),
            self.d_velocities_y.get(),
            self.d_velocities_z.get(),
        ], axis=1)
        particle_table.velocities[:] = vel

    def download_forces(self, particle_table):
        frc = np.stack([
            self.d_forces_x.get(),
            self.d_forces_y.get(),
            self.d_forces_z.get(),
        ], axis=1)
        particle_table.forces[:] = frc

    def zero_forces(self):
        self.d_forces_x[:] = 0
        self.d_forces_y[:] = 0
        self.d_forces_z[:] = 0

    def zero_energy(self):
        self.d_energy[:] = 0

    def allocate_energy_accumulator(self, num_terms):
        self.d_energy_accumulator = cp.zeros(num_terms, dtype=np.float32)

    def accumulate_energy(self, term_index):
        self.d_energy_accumulator[term_index] = self.d_energy[0]

    def get_positions_2d(self):
        return (self.d_positions_x, self.d_positions_y, self.d_positions_z)

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
