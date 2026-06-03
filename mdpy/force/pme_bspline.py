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



_SPREAD_KERNEL_SOURCE = r"""
extern "C" __global__
void spread_kernel(
    const float* __restrict__ positions_x,
    const float* __restrict__ positions_y,
    const float* __restrict__ positions_z,
    const float* __restrict__ charges,
    int num_particles,
    float recip_box_x, float recip_box_y, float recip_box_z,
    int grid_x, int grid_y, int grid_z,
    int order,
    float* __restrict__ charge_grid
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

    int grid_start[3];
    grid_start[0] = ((int)floorf(fx)) % grid_x;
    grid_start[1] = ((int)floorf(fy)) % grid_y;
    grid_start[2] = ((int)floorf(fz)) % grid_z;
    if (grid_start[0] < 0) grid_start[0] += grid_x;
    if (grid_start[1] < 0) grid_start[1] += grid_y;
    if (grid_start[2] < 0) grid_start[2] += grid_z;

    float theta[3][4];
    for (int dim = 0; dim < 3; dim++) {
        float u = u_arr[dim];
        float data[4];
        data[0] = 1.0f - u;
        data[1] = u;
        data[2] = 0.0f;
        data[3] = 0.0f;
        for (int j = 3; j < order; j++) {
            float div = 1.0f / (float)(j - 1);
            data[j - 1] = div * u * data[j - 2];
            for (int k = 1; k < j - 1; k++) {
                data[j - k - 1] = div * ((u + (float)k) * data[j - k - 2] + ((float)(j - k) - u) * data[j - k - 1]);
            }
            data[0] = div * (1.0f - u) * data[0];
        }
        float scale = 1.0f / (float)(order - 1);
        data[order - 1] = scale * u * data[order - 2];
        for (int j = 1; j < order - 1; j++) {
            data[order - j - 1] = scale * ((u + (float)j) * data[order - j - 2] + ((float)(order - j) - u) * data[order - j - 1]);
        }
        data[0] = scale * (1.0f - u) * data[0];
        for (int k = 0; k < order; k++) theta[dim][k] = data[k];
    }

    float q = charges[i];

    for (int kx = 0; kx < order; kx++) {
        int gx = (grid_start[0] + kx) % grid_x;
        if (gx < 0) gx += grid_x;
        float tx = theta[0][kx];

        for (int ky = 0; ky < order; ky++) {
            int gy = (grid_start[1] + ky) % grid_y;
            if (gy < 0) gy += grid_y;
            float ty = theta[1][ky];

            for (int kz = 0; kz < order; kz++) {
                int gz = (grid_start[2] + kz) % grid_z;
                if (gz < 0) gz += grid_z;
                float tz = theta[2][kz];

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


_GATHER_KERNEL_SOURCE = r"""
extern "C" __global__
void gather_kernel(
    const float* __restrict__ positions_x,
    const float* __restrict__ positions_y,
    const float* __restrict__ positions_z,
    const float* __restrict__ charges,
    int num_particles,
    float recip_box_x, float recip_box_y, float recip_box_z,
    int grid_x, int grid_y, int grid_z,
    int order,
    const float* __restrict__ phi_grid,
    float* __restrict__ forces_x,
    float* __restrict__ forces_y,
    float* __restrict__ forces_z,
    float* __restrict__ energy_buffer
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

    int grid_start[3];
    grid_start[0] = ((int)floorf(fx)) % grid_x;
    grid_start[1] = ((int)floorf(fy)) % grid_y;
    grid_start[2] = ((int)floorf(fz)) % grid_z;
    if (grid_start[0] < 0) grid_start[0] += grid_x;
    if (grid_start[1] < 0) grid_start[1] += grid_y;
    if (grid_start[2] < 0) grid_start[2] += grid_z;

    float theta[3][4];
    float dtheta[3][4];
    for (int dim = 0; dim < 3; dim++) {
        float u = u_arr[dim];
        float data[4];
        data[0] = 1.0f - u;
        data[1] = u;
        data[2] = 0.0f;
        data[3] = 0.0f;
        for (int j = 3; j < order; j++) {
            float div = 1.0f / (float)(j - 1);
            data[j - 1] = div * u * data[j - 2];
            for (int k = 1; k < j - 1; k++) {
                data[j - k - 1] = div * ((u + (float)k) * data[j - k - 2] + ((float)(j - k) - u) * data[j - k - 1]);
            }
            data[0] = div * (1.0f - u) * data[0];
        }
        dtheta[dim][0] = -data[0];
        for (int k = 1; k < order; k++) dtheta[dim][k] = data[k - 1] - data[k];
        float scale = 1.0f / (float)(order - 1);
        data[order - 1] = scale * u * data[order - 2];
        for (int j = 1; j < order - 1; j++) {
            data[order - j - 1] = scale * ((u + (float)j) * data[order - j - 2] + ((float)(order - j) - u) * data[order - j - 1]);
        }
        data[0] = scale * (1.0f - u) * data[0];
        for (int k = 0; k < order; k++) theta[dim][k] = data[k];
    }

    float q = charges[i];
    float energy = 0.0f;
    float ffx = 0.0f, ffy = 0.0f, ffz = 0.0f;

    for (int kx = 0; kx < order; kx++) {
        int gx = (grid_start[0] + kx) % grid_x;
        if (gx < 0) gx += grid_x;
        float tx = theta[0][kx];
        float dtx = dtheta[0][kx];

        for (int ky = 0; ky < order; ky++) {
            int gy = (grid_start[1] + ky) % grid_y;
            if (gy < 0) gy += grid_y;
            float ty = theta[1][ky];
            float dty = dtheta[1][ky];

            for (int kz = 0; kz < order; kz++) {
                int gz = (grid_start[2] + kz) % grid_z;
                if (gz < 0) gz += grid_z;
                float tz = theta[2][kz];
                float dtz = dtheta[2][kz];

                int idx = (gx * grid_y + gy) * grid_z + gz;
                float phi = phi_grid[idx];
                float txyz = tx * ty * tz;

                energy += txyz * phi;
                ffx += dtx * ty * tz * phi;
                ffy += tx * dty * tz * phi;
                ffz += tx * ty * dtz * phi;
            }
        }
    }

    energy *= 0.5f * q;
    ffx *= -q * grid_x * recip_box_x;
    ffy *= -q * grid_y * recip_box_y;
    ffz *= -q * grid_z * recip_box_z;

    atomicAdd(&forces_x[i], ffx);
    atomicAdd(&forces_y[i], ffy);
    atomicAdd(&forces_z[i], ffz);

    for (int offset = 16; offset > 0; offset >>= 1) {
        energy += __shfl_down_sync(0xffffffff, energy, offset);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(energy_buffer, energy);
    }
}
"""

_SELF_ENERGY_KERNEL_SOURCE = r"""
extern "C" __global__
void self_energy_kernel(
    float self_energy,
    float* __restrict__ energy_buffer
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        atomicAdd(energy_buffer, self_energy);
    }
}
"""

_EXCLUSION_KERNEL_SOURCE = r"""
extern "C" __global__
void exclusion_kernel(
    const float* __restrict__ positions_x,
    const float* __restrict__ positions_y,
    const float* __restrict__ positions_z,
    const float* __restrict__ charges,
    const int* __restrict__ pair_i,
    const int* __restrict__ pair_j,
    const float* __restrict__ pair_scale,
    int num_pairs,
    float alpha,
    float box_x, float box_y, float box_z,
    float* __restrict__ forces_x,
    float* __restrict__ forces_y,
    float* __restrict__ forces_z,
    float* __restrict__ energy_buffer
) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= num_pairs) return;

    int i = pair_i[p];
    int j = pair_j[p];
    float scale = pair_scale[p];
    float one_minus_scale = 1.0f - scale;

    float dx = positions_x[i] - positions_x[j];
    float dy = positions_y[i] - positions_y[j];
    float dz = positions_z[i] - positions_z[j];

    dx -= roundf(dx / box_x) * box_x;
    dy -= roundf(dy / box_y) * box_y;
    dz -= roundf(dz / box_z) * box_z;

    float r_sq = dx*dx + dy*dy + dz*dz;
    float r = sqrtf(r_sq);
    float inv_r = 1.0f / r;

    float qi = charges[i];
    float qj = charges[j];
    float qq = qi * qj;

    float alpha_r = alpha * r;
    float COULOMB_CONST = 0.13893556595455f;

    float erf_val = erff(alpha_r);

    float z2 = alpha_r * alpha_r;
    float z4 = z2 * z2;

    float fd_a = 0.0011193462567257629232f * z4 + 0.11583842382862377919f;
    float fd_b = 0.014866955030185295499f * z4 + 0.50736591960530292870f;
    float fd_c = fd_a * z4 + 1.0f;
    float fd_d = fd_b * z2 + fd_c;
    float inv_fd = 1.0f / fd_d;

    float fn_a = -1.7357322914161492954e-8f * z4 - 5.3401640219807709149e-5f;
    float fn_b = 1.4703624142580877519e-6f * z4 + 1.0054721316683106153e-3f;
    float fn_c = fn_a * z4 - 1.927831726488838059e-2f;
    float fn_d = fn_b * z4 + 6.9670166153766424023e-2f;
    float fn_e = fn_c * z4 - 0.75225204789749321333f;
    float corr = (fn_d * z2 + fn_e) * inv_fd;

    float alpha3 = alpha * alpha * alpha;

    float corr_energy = -COULOMB_CONST * one_minus_scale * qq * erf_val * inv_r;

    float corr_fmag = -COULOMB_CONST * one_minus_scale * qq * alpha3 * r * corr;

    float fx = corr_fmag * dx * inv_r;
    float fy = corr_fmag * dy * inv_r;
    float fz = corr_fmag * dz * inv_r;

    atomicAdd(&forces_x[i], fx);
    atomicAdd(&forces_y[i], fy);
    atomicAdd(&forces_z[i], fz);
    atomicAdd(&forces_x[j], -fx);
    atomicAdd(&forces_y[j], -fy);
    atomicAdd(&forces_z[j], -fz);

    for (int offset = 16; offset > 0; offset >>= 1) {
        corr_energy += __shfl_down_sync(0xffffffff, corr_energy, offset);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(energy_buffer, corr_energy);
    }
}
"""

_spread_kernel = None
_gather_kernel = None
_self_energy_kernel = None
_exclusion_kernel = None


def get_spread_kernel():
    global _spread_kernel
    if _spread_kernel is None:
        _spread_kernel = cp.RawKernel(_SPREAD_KERNEL_SOURCE, "spread_kernel")
    return _spread_kernel


def get_gather_kernel():
    global _gather_kernel
    if _gather_kernel is None:
        _gather_kernel = cp.RawKernel(_GATHER_KERNEL_SOURCE, "gather_kernel")
    return _gather_kernel


def get_self_energy_kernel():
    global _self_energy_kernel
    if _self_energy_kernel is None:
        _self_energy_kernel = cp.RawKernel(_SELF_ENERGY_KERNEL_SOURCE, "self_energy_kernel")
    return _self_energy_kernel


def get_exclusion_kernel():
    global _exclusion_kernel
    if _exclusion_kernel is None:
        _exclusion_kernel = cp.RawKernel(_EXCLUSION_KERNEL_SOURCE, "exclusion_kernel")
    return _exclusion_kernel
