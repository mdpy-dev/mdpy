from __future__ import annotations

import numpy as np
import cupy as cp
from mdpy import precision

_PBC_WRAP_INPLACE_KERNEL = r"""
extern "C" __global__
void pbc_wrap_inplace_kernel(
    float* __restrict__ pos_x,
    float* __restrict__ pos_y,
    float* __restrict__ pos_z,
    const float* __restrict__ pbc_matrix,
    const float* __restrict__ pbc_inv,
    int num_particles
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_particles) return;
    float px = pos_x[index];
    float py = pos_y[index];
    float pz = pos_z[index];
    float fx = px*pbc_inv[0] + py*pbc_inv[3] + pz*pbc_inv[6];
    float fy = px*pbc_inv[1] + py*pbc_inv[4] + pz*pbc_inv[7];
    float fz = px*pbc_inv[2] + py*pbc_inv[5] + pz*pbc_inv[8];
    fx -= floorf(fx); fy -= floorf(fy); fz -= floorf(fz);
    pos_x[index] = fx*pbc_matrix[0] + fy*pbc_matrix[3] + fz*pbc_matrix[6];
    pos_y[index] = fx*pbc_matrix[1] + fy*pbc_matrix[4] + fz*pbc_matrix[7];
    pos_z[index] = fx*pbc_matrix[2] + fy*pbc_matrix[5] + fz*pbc_matrix[8];
}
"""

_WRAP_CORRECT_KERNEL = r"""
extern "C" __global__
void wrap_correct_kernel(
    float* __restrict__ pos_x,
    float* __restrict__ pos_y,
    float* __restrict__ pos_z,
    float* __restrict__ prev_pos_x,
    float* __restrict__ prev_pos_y,
    float* __restrict__ prev_pos_z,
    const float* __restrict__ pbc_matrix,
    const float* __restrict__ pbc_inv,
    int num_particles
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_particles) return;
    float px = pos_x[index];
    float py = pos_y[index];
    float pz = pos_z[index];
    float fx = px*pbc_inv[0] + py*pbc_inv[3] + pz*pbc_inv[6];
    float fy = px*pbc_inv[1] + py*pbc_inv[4] + pz*pbc_inv[7];
    float fz = px*pbc_inv[2] + py*pbc_inv[5] + pz*pbc_inv[8];
    float fx_w = fx - floorf(fx);
    float fy_w = fy - floorf(fy);
    float fz_w = fz - floorf(fz);
    float dx = (fx_w - fx)*pbc_matrix[0] + (fy_w - fy)*pbc_matrix[3] + (fz_w - fz)*pbc_matrix[6];
    float dy = (fx_w - fx)*pbc_matrix[1] + (fy_w - fy)*pbc_matrix[4] + (fz_w - fz)*pbc_matrix[7];
    float dz = (fx_w - fx)*pbc_matrix[2] + (fy_w - fy)*pbc_matrix[5] + (fz_w - fz)*pbc_matrix[8];
    pos_x[index] += dx;
    pos_y[index] += dy;
    pos_z[index] += dz;
    prev_pos_x[index] += dx;
    prev_pos_y[index] += dy;
    prev_pos_z[index] += dz;
}
"""

_ZERO_FORCES_KERNEL = r"""
extern "C" __global__
void zero_forces_kernel(
    float* __restrict__ fx,
    float* __restrict__ fy,
    float* __restrict__ fz,
    int num_particles
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_particles) return;
    fx[index] = 0.0f;
    fy[index] = 0.0f;
    fz[index] = 0.0f;
}
"""


class State:

    def __init__(self, num_particles):
        self.num_particles = num_particles

        # Per-particle state arrays (zero-filled; populated by set_* methods).
        self.d_positions_x = cp.zeros(self.num_particles, dtype=np.float32)
        self.d_positions_y = cp.zeros(self.num_particles, dtype=np.float32)
        self.d_positions_z = cp.zeros(self.num_particles, dtype=np.float32)

        self.d_velocities_x = cp.zeros(self.num_particles, dtype=np.float32)
        self.d_velocities_y = cp.zeros(self.num_particles, dtype=np.float32)
        self.d_velocities_z = cp.zeros(self.num_particles, dtype=np.float32)

        self.d_forces_x = cp.zeros(self.num_particles, dtype=np.float32)
        self.d_forces_y = cp.zeros(self.num_particles, dtype=np.float32)
        self.d_forces_z = cp.zeros(self.num_particles, dtype=np.float32)

        self.d_prev_positions_x = cp.zeros(self.num_particles, dtype=np.float32)
        self.d_prev_positions_y = cp.zeros(self.num_particles, dtype=np.float32)
        self.d_prev_positions_z = cp.zeros(self.num_particles, dtype=np.float32)

        # Per-particle properties (zero-filled; populated by set_* methods).
        self.d_particle_masses = cp.zeros(self.num_particles, dtype=np.float32)
        self.d_particle_type_indices = cp.zeros(self.num_particles, dtype=np.int32)
        self.d_particle_charges = cp.zeros(self.num_particles, dtype=np.float32)

        self.d_energy = cp.zeros(1, dtype=np.float32)
        self.d_energy_accumulator = None

        # PBC: lazy-allocated on first set_pbc. None until then.
        self.d_pbc_matrix = None
        self.d_pbc_inv = None
        self._box_x = 0.0
        self._box_y = 0.0
        self._box_z = 0.0
        self._inv_box_x = 0.0
        self._inv_box_y = 0.0
        self._inv_box_z = 0.0

        # State-set flags (queried via has_* properties / is_ready).
        self._has_positions = False
        self._has_velocities = False
        self._has_charges = False
        self._has_masses = False
        self._has_type_indices = False
        self._has_pbc = False

        # Lazy kernel caches.
        self._zero_forces_kernel = None
        self._wrap_kernel = None
        self._wrap_correct_kernel = None

    def _ensure_wrap_kernel(self):
        if self._wrap_kernel is not None:
            return
        self._wrap_kernel = cp.RawKernel(
            _PBC_WRAP_INPLACE_KERNEL, "pbc_wrap_inplace_kernel"
        )

    def _ensure_wrap_correct_kernel(self):
        if self._wrap_correct_kernel is not None:
            return
        self._wrap_correct_kernel = cp.RawKernel(
            _WRAP_CORRECT_KERNEL, "wrap_correct_kernel"
        )

    def wrap_positions_with_prev_correction(self):
        self._ensure_wrap_correct_kernel()
        N = self.num_particles
        threads_per_block = 256
        grid = ((N + threads_per_block - 1) // threads_per_block,)
        self._wrap_correct_kernel(
            grid,
            (threads_per_block,),
            (
                self.d_positions_x,
                self.d_positions_y,
                self.d_positions_z,
                self.d_prev_positions_x,
                self.d_prev_positions_y,
                self.d_prev_positions_z,
                self.d_pbc_matrix,
                self.d_pbc_inv,
                np.int32(N),
            ),
        )

    def _wrap_positions_inplace(self):
        self._ensure_wrap_kernel()
        N = self.num_particles
        threads_per_block = 256
        grid = ((N + threads_per_block - 1) // threads_per_block,)
        self._wrap_kernel(
            grid,
            (threads_per_block,),
            (
                self.d_positions_x,
                self.d_positions_y,
                self.d_positions_z,
                self.d_pbc_matrix,
                self.d_pbc_inv,
                np.int32(N),
            ),
        )

    def _ensure_zero_forces_kernel(self):
        if self._zero_forces_kernel is not None:
            return
        self._zero_forces_kernel = cp.RawKernel(
            _ZERO_FORCES_KERNEL, "zero_forces_kernel"
        )

    def set_pbc(self, pbc_matrix):
        """Set the periodic box matrix. Allocates d_pbc_matrix/d_pbc_inv on
        first call; subsequent calls overwrite in place. Does NOT trigger a
        neighbor-list rebuild."""
        pbc_flat = np.ascontiguousarray(
            np.asarray(pbc_matrix, dtype=np.float32)
        ).ravel()
        if self.d_pbc_matrix is None:
            self.d_pbc_matrix = cp.empty(9, dtype=np.float32)
            self.d_pbc_inv = cp.empty(9, dtype=np.float32)
        self.d_pbc_matrix[:] = cp.asarray(pbc_flat)
        pbc_inv = np.linalg.inv(pbc_flat.reshape(3, 3))
        self.d_pbc_inv[:] = cp.asarray(
            np.ascontiguousarray(pbc_inv, dtype=np.float32).ravel()
        )
        pbc_2d = np.asarray(pbc_matrix).reshape(3, 3)
        self.set_box_dims(
            abs(float(pbc_2d[0, 0])),
            abs(float(pbc_2d[1, 1])),
            abs(float(pbc_2d[2, 2])),
        )
        self._has_pbc = True

    def set_positions(self, positions):
        data = np.ascontiguousarray(np.asarray(positions, dtype=np.float32))
        self.d_positions_x[:] = cp.asarray(data[:, 0])
        self.d_positions_y[:] = cp.asarray(data[:, 1])
        self.d_positions_z[:] = cp.asarray(data[:, 2])
        if self._has_pbc:
            self._wrap_positions_inplace()
        self._has_positions = True

    def set_velocities(self, velocities):
        data = np.ascontiguousarray(np.asarray(velocities, dtype=np.float32))
        self.d_velocities_x[:] = cp.asarray(data[:, 0])
        self.d_velocities_y[:] = cp.asarray(data[:, 1])
        self.d_velocities_z[:] = cp.asarray(data[:, 2])
        self._has_velocities = True

    def set_prev_positions(self, positions):
        data = np.ascontiguousarray(np.asarray(positions, dtype=np.float32))
        self.d_prev_positions_x[:] = cp.asarray(data[:, 0])
        self.d_prev_positions_y[:] = cp.asarray(data[:, 1])
        self.d_prev_positions_z[:] = cp.asarray(data[:, 2])

    def set_particle_charges(self, particle_charges):
        data = np.ascontiguousarray(np.asarray(particle_charges, dtype=np.float32))
        self.d_particle_charges[:] = cp.asarray(data)
        self._has_charges = True

    def set_particle_masses(self, particle_masses):
        data = np.ascontiguousarray(np.asarray(particle_masses, dtype=np.float32))
        self.d_particle_masses[:] = cp.asarray(data)
        self._has_masses = True

    def set_particle_type_indices(self, particle_type_indices):
        data = np.ascontiguousarray(np.asarray(particle_type_indices, dtype=np.int32))
        self.d_particle_type_indices[:] = cp.asarray(data)
        self._has_type_indices = True

    def download_positions(self):
        return np.stack(
            [
                self.d_positions_x.get(),
                self.d_positions_y.get(),
                self.d_positions_z.get(),
            ],
            axis=1,
        )

    def download_velocities(self):
        return np.stack(
            [
                self.d_velocities_x.get(),
                self.d_velocities_y.get(),
                self.d_velocities_z.get(),
            ],
            axis=1,
        )

    def download_forces(self):
        return np.stack(
            [self.d_forces_x.get(), self.d_forces_y.get(), self.d_forces_z.get()],
            axis=1,
        )

    def zero_forces(self):
        self._ensure_zero_forces_kernel()
        N = self.num_particles
        threads_per_block = 256
        grid = ((N + threads_per_block - 1) // threads_per_block,)
        self._zero_forces_kernel(
            grid,
            (threads_per_block,),
            (
                self.d_forces_x,
                self.d_forces_y,
                self.d_forces_z,
                np.int32(N),
            ),
        )

    def zero_energy(self):
        self.d_energy[:] = 0

    def allocate_energy_accumulator(self, num_terms):
        self.d_energy_accumulator = cp.zeros(num_terms, dtype=np.float32)

    def set_energy_slot(self, term_index):
        self.d_energy_accumulator[term_index] = self.d_energy[0]

    def set_box_dims(self, box_x, box_y, box_z):
        self._box_x = float(box_x)
        self._box_y = float(box_y)
        self._box_z = float(box_z)
        self._inv_box_x = 1.0 / self._box_x
        self._inv_box_y = 1.0 / self._box_y
        self._inv_box_z = 1.0 / self._box_z

    @property
    def box_x(self):
        return self._box_x

    @property
    def box_y(self):
        return self._box_y

    @property
    def box_z(self):
        return self._box_z

    @property
    def inv_box_x(self):
        return self._inv_box_x

    @property
    def inv_box_y(self):
        return self._inv_box_y

    @property
    def inv_box_z(self):
        return self._inv_box_z

    @property
    def has_pbc(self):
        return self._has_pbc

    @property
    def has_positions(self):
        return self._has_positions

    @property
    def has_velocities(self):
        return self._has_velocities

    @property
    def is_ready(self):
        return (self._has_positions and self._has_velocities
                and self._has_charges and self._has_masses
                and self._has_type_indices and self._has_pbc)
