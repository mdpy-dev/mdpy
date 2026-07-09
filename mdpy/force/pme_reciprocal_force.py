from __future__ import annotations

import math

import cupy as cp
import numpy as np
from cupy.cuda import cufft
from cupy.fft._fft import _default_fft_func
from scipy.special import erfc

from mdpy.force.force_term import ForceTerm
from mdpy.unit import EPSILON0

SQRT_PI = 1.772453850905516

# Coulomb constant 1/(4*pi*epsilon0) in mdpy internal units (file-local).
COULOMB_CONST = 1.0 / (4.0 * math.pi * float(EPSILON0.value))

# CUDA float-literal form of the Coulomb constant, used to inject the value into
# the raw kernel strings below (which cannot be f-strings because CUDA braces
# would need escaping).
_COULOMB_CUDA = f"{COULOMB_CONST}f"


def _calc_ewald_coefficient(cutoff: float, rtol: float = 1e-5) -> float:
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if erfc(mid * cutoff) > rtol:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _next_fft_friendly_size(n: int) -> int:
    while True:
        m = n
        for p in (7, 5, 3, 2):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def compute_bspline_weights(
    fractional: float, order: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    w = fractional - math.floor(fractional)

    data = [0.0] * order
    data[order - 1] = 0.0
    data[1] = w
    data[0] = 1.0 - w

    for j in range(3, order):
        div = 1.0 / (j - 1)
        data[j - 1] = div * w * data[j - 2]
        for k in range(1, j - 1):
            data[j - k - 1] = div * (
                (w + k) * data[j - k - 2] + (j - k - w) * data[j - k - 1]
            )
        data[0] = div * (1.0 - w) * data[0]

    ddata = [0.0] * order
    ddata[0] = -data[0]
    for k in range(1, order):
        ddata[k] = data[k - 1] - data[k]

    scale = 1.0 / (order - 1)
    data[order - 1] = scale * w * data[order - 2]
    for j in range(1, order - 1):
        data[order - j - 1] = scale * (
            (w + j) * data[order - j - 2] + (order - j - w) * data[order - j - 1]
        )
    data[0] = scale * (1.0 - w) * data[0]

    theta = np.array(data, dtype=np.float64)
    dtheta = np.array(ddata, dtype=np.float64)
    return theta, dtheta


_CELL_SPREAD_KERNEL_SOURCE = r"""
extern "C" __global__
void cell_spread_kernel(
    const float* __restrict__ positions_x,
    const float* __restrict__ positions_y,
    const float* __restrict__ positions_z,
    const float* __restrict__ charges,
    const int* __restrict__ cell_block_offset,
    const int* __restrict__ cell_block_count,
    const int* __restrict__ block_atoms,
    int num_particles,
    float recip_box_x, float recip_box_y, float recip_box_z,
    int grid_x, int grid_y, int grid_z,
    int nc_x, int nc_y, int nc_z,
    int subgrid_dx, int subgrid_dy, int subgrid_dz,
    int order,
    float* __restrict__ charge_grid
) {
    extern __shared__ float subgrid[];

    int cell_idx = blockIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    int cx = cell_idx % nc_x;
    int cy = (cell_idx / nc_x) % nc_y;
    int cz = cell_idx / (nc_x * nc_y);

    int gx_origin = (int)floorf((float)cx / nc_x * grid_x) - order;
    int gy_origin = (int)floorf((float)cy / nc_y * grid_y) - order;
    int gz_origin = (int)floorf((float)cz / nc_z * grid_z) - order;

    int subgrid_total = subgrid_dx * subgrid_dy * subgrid_dz;

    for (int i = threadIdx.x; i < subgrid_total; i += 256)
        subgrid[i] = 0.0f;
    __syncthreads();

    int n_blocks = cell_block_count[cell_idx];
    int block_start = cell_block_offset[cell_idx];

    for (int b = warp_id; b < n_blocks; b += 8) {
        int global_block = block_start + b;
        int atom_id = block_atoms[global_block * 32 + lane];
        if (atom_id < 0 || atom_id >= num_particles) continue;

        float px = positions_x[atom_id];
        float py = positions_y[atom_id];
        float pz = positions_z[atom_id];

        float fx = px * recip_box_x * grid_x;
        float fy = py * recip_box_y * grid_y;
        float fz = pz * recip_box_z * grid_z;

        float u_arr[3];
        u_arr[0] = fx - floorf(fx);
        u_arr[1] = fy - floorf(fy);
        u_arr[2] = fz - floorf(fz);

        int g0 = (int)floorf(fx);
        int g1 = (int)floorf(fy);
        int g2 = (int)floorf(fz);

        int lx0 = g0 - gx_origin;
        int ly0 = g1 - gy_origin;
        int lz0 = g2 - gz_origin;

        float theta[3][4];
        for (int dim = 0; dim < 3; dim++) {
            float u = u_arr[dim];
            float u2 = u * u;
            float u3 = u2 * u;
            float om = 1.0f - u;
            float om3 = om * om * om;
            float inv6 = 0.166666667f;
            theta[dim][0] = om3 * inv6;
            theta[dim][1] = (3.0f*u3 - 6.0f*u2 + 4.0f) * inv6;
            theta[dim][2] = (-3.0f*u3 + 3.0f*u2 + 3.0f*u + 1.0f) * inv6;
            theta[dim][3] = u3 * inv6;
        }

        float q = charges[atom_id];

        for (int kx = 0; kx < order; kx++) {
            int lx = lx0 + kx;
            if (lx < 0 || lx >= subgrid_dx) continue;
            for (int ky = 0; ky < order; ky++) {
                int ly = ly0 + ky;
                if (ly < 0 || ly >= subgrid_dy) continue;
                for (int kz = 0; kz < order; kz++) {
                    int lz = lz0 + kz;
                    if (lz < 0 || lz >= subgrid_dz) continue;
                    float contrib = q * theta[0][kx] * theta[1][ky] * theta[2][kz];
                    int local_idx = lx * subgrid_dy * subgrid_dz + ly * subgrid_dz + lz;
                    atomicAdd(&subgrid[local_idx], contrib);
                }
            }
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < subgrid_total; i += 256) {
        if (subgrid[i] == 0.0f) continue;
        int lx = i / (subgrid_dy * subgrid_dz);
        int ly = (i / subgrid_dz) % subgrid_dy;
        int lz = i % subgrid_dz;
        int gx = (gx_origin + lx) % grid_x;
        int gy = (gy_origin + ly) % grid_y;
        int gz = (gz_origin + lz) % grid_z;
        if (gx < 0) gx += grid_x;
        if (gy < 0) gy += grid_y;
        if (gz < 0) gz += grid_z;
        int global_idx = (gx * grid_y + gy) * grid_z + gz;
        atomicAdd(&charge_grid[global_idx], subgrid[i]);
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
        data[order - l - 1] = div * (
            l * data[order - l - 2] + (order - l) * data[order - l - 1]
        )
    data[0] = div * data[0]

    bsplines_data = np.zeros(grid_dim, dtype=np.float64)
    for i in range(1, min(order + 1, grid_dim)):
        bsplines_data[i] = data[i - 1]

    dft = np.fft.fft(bsplines_data)
    moduli = np.abs(dft) ** 2

    for i in range(grid_dim):
        if moduli[i] < 1e-7:
            moduli[i] = (
                moduli[(i - 1 + grid_dim) % grid_dim] + moduli[(i + 1) % grid_dim]
            ) * 0.5

    return moduli


def precompute_bk_factors(
    alpha: float,
    grid_x: int,
    grid_y: int,
    grid_z: int,
    order: int,
    box_x: float,
    box_y: float,
    box_z: float,
) -> tuple[np.ndarray, np.ndarray]:  # (bk, bk_virial)
    moduli_x = _compute_bspline_moduli(grid_x, order)
    moduli_y = _compute_bspline_moduli(grid_y, order)
    moduli_z = _compute_bspline_moduli(grid_z, order)

    volume = box_x * box_y * box_z
    grid_total = grid_x * grid_y * grid_z
    scale_factor = 2.0 * volume / grid_total
    recip_exp_factor = math.pi**2 / (alpha**2)

    recip_x = 1.0 / box_x
    recip_y = 1.0 / box_y
    recip_z = 1.0 / box_z

    nz_half = grid_z // 2 + 1
    bk = np.zeros((grid_x, grid_y, nz_half), dtype=np.float32)
    bk_virial = np.zeros((grid_x, grid_y, nz_half), dtype=np.float32)

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
                bk_val = math.exp(-recip_exp_factor * m2) / denom
                bk[kx, ky, kz] = bk_val
                virial_factor = 1.0 - 2.0 * recip_exp_factor * m2
                bk_virial[kx, ky, kz] = bk_val * virial_factor

            firstz = 0

    return bk, bk_virial


_GATHER_KERNEL_SOURCE = r"""
extern "C" __global__
void gather_kernel(
    const float* __restrict__ positions_x,
    const float* __restrict__ positions_y,
    const float* __restrict__ positions_z,
    const float* __restrict__ charges,
    int num_particles,
    const int* __restrict__ block_atoms,
    int total_slots,
    float recip_box_x, float recip_box_y, float recip_box_z,
    int grid_x, int grid_y, int grid_z,
    int order,
    const float* __restrict__ phi_grid,
    float* __restrict__ forces_x,
    float* __restrict__ forces_y,
    float* __restrict__ forces_z,
    float* __restrict__ energy_buffer
) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= total_slots) return;
    int i = block_atoms[slot];
    if (i < 0 || i >= num_particles) return;

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
        float d0, d1, d2, d3;

        d0 = 1.0f - u;
        d1 = u;
        d2 = 0.0f;
        d3 = 0.0f;

        d2 = 0.5f * u * d1;
        d1 = 0.5f * ((u + 1.0f) * d0 + (2.0f - u) * d1);
        d0 = 0.5f * (1.0f - u) * d0;

        dtheta[dim][0] = -d0;
        dtheta[dim][1] = d0 - d1;
        dtheta[dim][2] = d1 - d2;
        dtheta[dim][3] = d2 - d3;

        d3 = (1.0f/3.0f) * u * d2;
        d2 = (1.0f/3.0f) * ((u + 1.0f) * d1 + (3.0f - u) * d2);
        d1 = (1.0f/3.0f) * ((u + 2.0f) * d0 + (2.0f - u) * d1);
        d0 = (1.0f/3.0f) * (1.0f - u) * d0;

        theta[dim][0] = d0;
        theta[dim][1] = d1;
        theta[dim][2] = d2;
        theta[dim][3] = d3;
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
                float phi = __ldg(&phi_grid[idx]);
                float txyz = tx * ty * tz;

                energy += txyz * phi;
                ffx += dtx * ty * tz * phi;
                ffy += tx * dty * tz * phi;
                ffz += tx * ty * dtz * phi;
            }
        }
    }

    energy *= 0.5f * __MDPY_COULOMB__ * q;
    ffx *= -__MDPY_COULOMB__ * q * grid_x * recip_box_x;
    ffy *= -__MDPY_COULOMB__ * q * grid_y * recip_box_y;
    ffz *= -__MDPY_COULOMB__ * q * grid_z * recip_box_z;

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
""".replace("__MDPY_COULOMB__", _COULOMB_CUDA)

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

_gather_kernel = None
_self_energy_kernel = None
_cell_spread_kernel = None


def get_gather_kernel():
    global _gather_kernel
    if _gather_kernel is None:
        _gather_kernel = cp.RawKernel(_GATHER_KERNEL_SOURCE, "gather_kernel")
    return _gather_kernel


def get_self_energy_kernel():
    global _self_energy_kernel
    if _self_energy_kernel is None:
        _self_energy_kernel = cp.RawKernel(
            _SELF_ENERGY_KERNEL_SOURCE, "self_energy_kernel"
        )
    return _self_energy_kernel


def get_cell_spread_kernel():
    global _cell_spread_kernel
    if _cell_spread_kernel is None:
        _cell_spread_kernel = cp.RawKernel(
            _CELL_SPREAD_KERNEL_SOURCE, "cell_spread_kernel"
        )
    return _cell_spread_kernel


class PMEReciprocalForce(ForceTerm):
    name = "pme_reciprocal"

    def __init__(
        self,
        cutoff: float,
        order: int = 4,
        fourier_spacing: float = 1.2,
        ewald_rtol: float = 1e-5,
    ):
        self.cutoff = cutoff
        self._order = order
        self._fourier_spacing = fourier_spacing
        self._ewald_rtol = ewald_rtol

        self.alpha = 0.0
        self.grid_x = 0
        self.grid_y = 0
        self.grid_z = 0

        self._d_bk_factors = None
        self._d_bk_virial_factors = None
        self._d_complex_buffer_virial = None
        self._d_charge_grid = None

        self._N = 0
        self._fft_warmed = False
        self._subgrid_initialized = False
        self._subgrid_dx = 0
        self._subgrid_dy = 0
        self._subgrid_dz = 0
        self._subgrid_total = 0

    @property
    def order(self):
        return self._order

    def initialize_grid(self, topology, parameter_table, pbc_matrix=None):
        N = topology.num_particles
        self._N = N

        if pbc_matrix is None:
            raise ValueError(
                "PMEReciprocalForce.bind requires pbc_matrix for FFT grid sizing."
            )
        pbc_2d = np.asarray(pbc_matrix, dtype=np.float64).reshape(3, 3)
        box_x = abs(float(pbc_2d[0, 0]))
        box_y = abs(float(pbc_2d[1, 1]))
        box_z = abs(float(pbc_2d[2, 2]))

        self.alpha = _calc_ewald_coefficient(self.cutoff, self._ewald_rtol)

        def grid_dim(box_dim: float) -> int:
            nmin = max(self._order, math.ceil(box_dim / self._fourier_spacing))
            return _next_fft_friendly_size(nmin)

        self.grid_x = grid_dim(box_x)
        self.grid_y = grid_dim(box_y)
        self.grid_z = grid_dim(box_z)

        grid_size = self.grid_x * self.grid_y * self.grid_z
        self._d_charge_grid = cp.zeros(grid_size, dtype=np.float32)

        nz_half = self.grid_z // 2 + 1
        self._d_complex_buffer = cp.zeros(
            (self.grid_x, self.grid_y, nz_half), dtype=cp.complex64
        )

        bk, bk_virial = precompute_bk_factors(
            self.alpha,
            self.grid_x,
            self.grid_y,
            self.grid_z,
            self._order,
            box_x,
            box_y,
            box_z,
        )
        self._d_bk_factors = cp.asarray(bk)
        self._d_bk_virial_factors = cp.asarray(bk_virial)

        self._warm_fft()

    def _warm_fft(self):
        if self._fft_warmed:
            return
        dummy = cp.zeros((self.grid_x, self.grid_y, self.grid_z), dtype=np.float32)
        fft = cp.fft.rfftn(dummy)
        cp.fft.irfftn(fft, s=(self.grid_x, self.grid_y, self.grid_z))
        self._fft_warmed = True

    def compute(
        self, state, block_list=None, compute_energy=True, compute_virial=False
    ):
        N = self._N
        order = self.order
        gx, gy, gz = self.grid_x, self.grid_y, self.grid_z

        threads_per_block = 256
        total_slots = block_list.max_blocks * 32
        grid_1d = ((total_slots + threads_per_block - 1) // threads_per_block,)

        self._d_charge_grid[:] = 0

        if not self._subgrid_initialized:
            self._subgrid_dx = -(-gx // block_list.num_cells_x) + 2 * order
            self._subgrid_dy = -(-gy // block_list.num_cells_y) + 2 * order
            self._subgrid_dz = -(-gz // block_list.num_cells_z) + 2 * order
            self._subgrid_total = self._subgrid_dx * self._subgrid_dy * self._subgrid_dz
            self._subgrid_initialized = True

        sorted_pos_x = state.d_positions_x
        sorted_pos_y = state.d_positions_y
        sorted_pos_z = state.d_positions_z
        sorted_charges = state.d_particle_charges

        cell_spread_k = get_cell_spread_kernel()
        shmem = self._subgrid_total * 4
        cell_spread_k(
            (block_list.num_cells_total,),
            (threads_per_block,),
            (
                sorted_pos_x,
                sorted_pos_y,
                sorted_pos_z,
                sorted_charges,
                block_list.d_cell_block_offset,
                block_list.d_cell_block_count,
                block_list.d_block_atoms,
                np.int32(N),
                np.float32(state.inv_box_x),
                np.float32(state.inv_box_y),
                np.float32(state.inv_box_z),
                np.int32(gx),
                np.int32(gy),
                np.int32(gz),
                np.int32(block_list.num_cells_x),
                np.int32(block_list.num_cells_y),
                np.int32(block_list.num_cells_z),
                np.int32(self._subgrid_dx),
                np.int32(self._subgrid_dy),
                np.int32(self._subgrid_dz),
                np.int32(order),
                self._d_charge_grid,
            ),
            shared_mem=shmem,
        )

        grid_3d = self._d_charge_grid.reshape(gx, gy, gz)
        _rfft_func = _default_fft_func(grid_3d, None, None, value_type="R2C")
        _rfft_func(
            grid_3d,
            None,
            None,
            None,
            cufft.CUFFT_FORWARD,
            "R2C",
            out=self._d_complex_buffer,
        )

        if compute_virial and self._d_bk_virial_factors is not None:
            self._d_complex_buffer_virial = self._d_complex_buffer.copy()

        cp.multiply(
            self._d_complex_buffer, self._d_bk_factors, out=self._d_complex_buffer
        )

        if compute_virial and self._d_bk_virial_factors is not None:
            rho_sq = cp.abs(self._d_complex_buffer_virial) ** 2
            grid_total = self.grid_x * self.grid_y * self.grid_z
            vol = state.box_x * state.box_y * state.box_z
            norm = vol / (grid_total * grid_total)
            virial_recip = float(cp.sum(rho_sq * self._d_bk_virial_factors)) * norm
            cp.cuda.Stream.null.synchronize()
            state.d_virial[0] += np.float32(virial_recip * COULOMB_CONST)

        _irfft_func = _default_fft_func(
            self._d_complex_buffer, None, None, value_type="C2R"
        )
        _irfft_func(
            self._d_complex_buffer,
            (gx, gy, gz),
            None,
            None,
            cufft.CUFFT_INVERSE,
            "C2R",
            out=grid_3d,
        )

        gather_k = get_gather_kernel()
        gather_k(
            grid_1d,
            (threads_per_block,),
            (
                state.d_positions_x,
                state.d_positions_y,
                state.d_positions_z,
                state.d_particle_charges,
                np.int32(N),
                block_list.d_block_atoms,
                np.int32(total_slots),
                np.float32(state.inv_box_x),
                np.float32(state.inv_box_y),
                np.float32(state.inv_box_z),
                np.int32(gx),
                np.int32(gy),
                np.int32(gz),
                np.int32(order),
                self._d_charge_grid,
                state.d_forces_x,
                state.d_forces_y,
                state.d_forces_z,
                state.d_energy,
            ),
        )

        if compute_energy:
            sum_q2 = float(cp.sum(state.d_particle_charges.astype(cp.float64) ** 2))
            self_energy_factor = -COULOMB_CONST * self.alpha / SQRT_PI * sum_q2
            self_k = get_self_energy_kernel()
            self_k((1,), (1,), (np.float32(self_energy_factor), state.d_energy))
