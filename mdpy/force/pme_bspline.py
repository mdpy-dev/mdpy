from __future__ import annotations

import math

import cupy as cp
import numpy as np


def compute_bspline_weights(fractional: float, order: int = 4) -> tuple[np.ndarray, np.ndarray]:
    w = fractional - math.floor(fractional)

    data = [0.0] * order
    data[order - 1] = 0.0
    data[1] = w
    data[0] = 1.0 - w

    for j in range(3, order):
        div = 1.0 / (j - 1)
        data[j - 1] = div * w * data[j - 2]
        for k in range(1, j - 1):
            data[j - k - 1] = div * ((w + k) * data[j - k - 2] + (j - k - w) * data[j - k - 1])
        data[0] = div * (1.0 - w) * data[0]

    ddata = [0.0] * order
    ddata[0] = -data[0]
    for k in range(1, order):
        ddata[k] = data[k - 1] - data[k]

    scale = 1.0 / (order - 1)
    data[order - 1] = scale * w * data[order - 2]
    for j in range(1, order - 1):
        data[order - j - 1] = scale * ((w + j) * data[order - j - 2] + (order - j - w) * data[order - j - 1])
    data[0] = scale * (1.0 - w) * data[0]

    theta = np.array(data, dtype=np.float64)
    dtheta = np.array(ddata, dtype=np.float64)
    return theta, dtheta


_BSPLINE_KERNEL_SOURCE = r"""
extern "C" __global__
void bspline_kernel(
    const float* __restrict__ positions_x,
    const float* __restrict__ positions_y,
    const float* __restrict__ positions_z,
    int num_particles,
    float recip_box_x,
    float recip_box_y,
    float recip_box_z,
    int grid_x,
    int grid_y,
    int grid_z,
    int order,
    float* __restrict__ theta,
    float* __restrict__ dtheta,
    int* __restrict__ grid_idx
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    float px = positions_x[i];
    float py = positions_y[i];
    float pz = positions_z[i];

    float fx = px * recip_box_x * grid_x;
    float fy = py * recip_box_y * grid_y;
    float fz = pz * recip_box_z * grid_z;

    float u_arr[3];
    u_arr[0] = fx - floorf(fx);
    u_arr[1] = fy - floorf(fy);
    u_arr[2] = fz - floorf(fz);

    grid_idx[i * 3 + 0] = ((int) floorf(fx)) % grid_x;
    grid_idx[i * 3 + 1] = ((int) floorf(fy)) % grid_y;
    grid_idx[i * 3 + 2] = ((int) floorf(fz)) % grid_z;

    int neg[3];
    neg[0] = grid_idx[i * 3 + 0] < 0 ? grid_x : 0;
    neg[1] = grid_idx[i * 3 + 1] < 0 ? grid_y : 0;
    neg[2] = grid_idx[i * 3 + 2] < 0 ? grid_z : 0;
    grid_idx[i * 3 + 0] += neg[0];
    grid_idx[i * 3 + 1] += neg[1];
    grid_idx[i * 3 + 2] += neg[2];

    int stride = order * 3;

    for (int dim = 0; dim < 3; dim++) {
        float u = u_arr[dim];

        float data[16];
        for (int k = 0; k < order; k++) data[k] = 0.0f;

        data[order - 1] = 0.0f;
        data[1] = u;
        data[0] = 1.0f - u;

        for (int j = 3; j < order; j++) {
            float div = 1.0f / (float)(j - 1);
            data[j - 1] = div * u * data[j - 2];
            for (int k = 1; k < j - 1; k++) {
                data[j - k - 1] = div * ((u + (float)k) * data[j - k - 2] + ((float)(j - k) - u) * data[j - k - 1]);
            }
            data[0] = div * (1.0f - u) * data[0];
        }

        float ddata[16];
        ddata[0] = -data[0];
        for (int k = 1; k < order; k++) {
            ddata[k] = data[k - 1] - data[k];
        }

        float scale = 1.0f / (float)(order - 1);
        data[order - 1] = scale * u * data[order - 2];
        for (int j = 1; j < order - 1; j++) {
            data[order - j - 1] = scale * ((u + (float)j) * data[order - j - 2] + ((float)(order - j) - u) * data[order - j - 1]);
        }
        data[0] = scale * (1.0f - u) * data[0];

        int base = i * stride + dim * order;
        for (int k = 0; k < order; k++) {
            theta[base + k] = data[k];
            dtheta[base + k] = ddata[k];
        }
    }
}
"""

_SPREAD_KERNEL_SOURCE = r"""
extern "C" __global__
void spread_kernel(
    const float* __restrict__ charges,
    const int* __restrict__ grid_idx,
    const float* __restrict__ theta,
    int num_particles,
    int grid_x, int grid_y, int grid_z,
    int order,
    float* __restrict__ charge_grid
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    float q = charges[i];
    int ix = grid_idx[i * 3 + 0];
    int iy = grid_idx[i * 3 + 1];
    int iz = grid_idx[i * 3 + 2];

    for (int kx = 0; kx < order; kx++) {
        int gx = (ix + kx) % grid_x;
        if (gx < 0) gx += grid_x;
        float tx = theta[i * order * 3 + 0 * order + kx];

        for (int ky = 0; ky < order; ky++) {
            int gy = (iy + ky) % grid_y;
            if (gy < 0) gy += grid_y;
            float ty = theta[i * order * 3 + 1 * order + ky];

            for (int kz = 0; kz < order; kz++) {
                int gz = (iz + kz) % grid_z;
                if (gz < 0) gz += grid_z;
                float tz = theta[i * order * 3 + 2 * order + kz];

                float contribution = q * tx * ty * tz;
                int idx = (gx * grid_y + gy) * grid_z + gz;
                atomicAdd(&charge_grid[idx], contribution);
            }
        }
    }
}
"""

def _compute_bspline_moduli(grid_dim: int, order: int) -> np.ndarray:
    data = [0.0] * order
    data[order - 1] = 0.0
    data[1] = 0.0
    data[0] = 1.0
    for k in range(3, order):
        div = 1.0 / (k - 1.0)
        data[k - 1] = 0.0
        for l in range(1, k - 1):
            data[k - l - 1] = div * (l * data[k - l - 2] + (k - l) * data[k - l - 1])
        data[0] = div * data[0]

    ddata = [0.0] * order
    ddata[0] = -data[0]
    for k in range(1, order):
        ddata[k] = data[k - 1] - data[k]

    div = 1.0 / (order - 1)
    data[order - 1] = 0.0
    for l in range(1, order - 1):
        data[order - l - 1] = div * (l * data[order - l - 2] + (order - l) * data[order - l - 1])
    data[0] = div * data[0]

    bsplines_data = np.zeros(grid_dim, dtype=np.float64)
    for i in range(1, min(order + 1, grid_dim)):
        bsplines_data[i] = data[i - 1]

    dft = np.fft.fft(bsplines_data)
    moduli = np.abs(dft) ** 2

    for i in range(grid_dim):
        if moduli[i] < 1e-7:
            moduli[i] = (moduli[(i - 1 + grid_dim) % grid_dim] + moduli[(i + 1) % grid_dim]) * 0.5

    return moduli


def precompute_bk_factors(alpha: float, grid_x: int, grid_y: int, grid_z: int,
                          order: int, box_x: float, box_y: float, box_z: float) -> np.ndarray:
    moduli_x = _compute_bspline_moduli(grid_x, order)
    moduli_y = _compute_bspline_moduli(grid_y, order)
    moduli_z = _compute_bspline_moduli(grid_z, order)

    volume = box_x * box_y * box_z
    scale_factor = math.pi * volume
    recip_exp_factor = math.pi ** 2 / (alpha ** 2)

    recip_x = 1.0 / box_x
    recip_y = 1.0 / box_y
    recip_z = 1.0 / box_z

    nz_half = grid_z // 2 + 1
    bk = np.zeros((grid_x, grid_y, nz_half), dtype=np.float32)

    firstz = 1
    for kx in range(grid_x):
        mx = kx if kx < (grid_x + 1) // 2 else kx - grid_x
        mhx = mx * recip_x
        bx = scale_factor * moduli_x[kx]

        for ky in range(grid_y):
            my = ky if ky < (grid_y + 1) // 2 else ky - grid_y
            mhy = my * recip_y
            mhx2y2 = mhx * mhx + mhy * mhy
            bxby = bx * moduli_y[ky]

            for kz in range(firstz, nz_half):
                mz = kz if kz < (grid_z + 1) // 2 else kz - grid_z
                mhz = mz * recip_z
                bz = moduli_z[kz]
                m2 = mhx2y2 + mhz * mhz
                denom = m2 * bxby * bz
                bk[kx, ky, kz] = math.exp(-recip_exp_factor * m2) / denom

            firstz = 0

    return bk


_bspline_kernel = None
_spread_kernel = None


def get_bspline_kernel():
    global _bspline_kernel
    if _bspline_kernel is None:
        _bspline_kernel = cp.RawKernel(_BSPLINE_KERNEL_SOURCE, "bspline_kernel")
    return _bspline_kernel


def get_spread_kernel():
    global _spread_kernel
    if _spread_kernel is None:
        _spread_kernel = cp.RawKernel(_SPREAD_KERNEL_SOURCE, "spread_kernel")
    return _spread_kernel
