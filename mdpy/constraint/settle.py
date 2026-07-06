import cupy as cp
import numpy as np
from .constraint_base import ConstraintBase

_SETTLE_KERNEL = r"""
extern "C" __global__
void settle_kernel(
    const float* __restrict__ old_x,
    const float* __restrict__ old_y,
    const float* __restrict__ old_z,
    float* __restrict__ new_x,
    float* __restrict__ new_y,
    float* __restrict__ new_z,
    const float* __restrict__ pbc_matrix,
    const float* __restrict__ pbc_inv,
    const int* __restrict__ water_idx,
    int num_waters,
    float wh,
    float ra,
    float rb,
    float rc,
    float inv_dHH,
    float time_step
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_waters) return;

    int ow1 = water_idx[tid * 3 + 0];
    int hw2 = water_idx[tid * 3 + 1];
    int hw3 = water_idx[tid * 3 + 2];

    // Load OLD (reference) positions
    float ox = old_x[ow1], oy = old_y[ow1], oz = old_z[ow1];
    float h1x = old_x[hw2], h1y = old_y[hw2], h1z = old_z[hw2];
    float h2x = old_x[hw3], h2y = old_y[hw3], h2z = old_z[hw3];

    // Old O->H vectors with PBC minimum image
    float d21x = h1x - ox, d21y = h1y - oy, d21z = h1z - oz;
    {
        float fx = d21x*pbc_inv[0] + d21y*pbc_inv[3] + d21z*pbc_inv[6];
        float fy = d21x*pbc_inv[1] + d21y*pbc_inv[4] + d21z*pbc_inv[7];
        float fz = d21x*pbc_inv[2] + d21y*pbc_inv[5] + d21z*pbc_inv[8];
        fx -= roundf(fx); fy -= roundf(fy); fz -= roundf(fz);
        d21x = fx*pbc_matrix[0] + fy*pbc_matrix[3] + fz*pbc_matrix[6];
        d21y = fx*pbc_matrix[1] + fy*pbc_matrix[4] + fz*pbc_matrix[7];
        d21z = fx*pbc_matrix[2] + fy*pbc_matrix[5] + fz*pbc_matrix[8];
    }
    float d31x = h2x - ox, d31y = h2y - oy, d31z = h2z - oz;
    {
        float fx = d31x*pbc_inv[0] + d31y*pbc_inv[3] + d31z*pbc_inv[6];
        float fy = d31x*pbc_inv[1] + d31y*pbc_inv[4] + d31z*pbc_inv[7];
        float fz = d31x*pbc_inv[2] + d31y*pbc_inv[5] + d31z*pbc_inv[8];
        fx -= roundf(fx); fy -= roundf(fy); fz -= roundf(fz);
        d31x = fx*pbc_matrix[0] + fy*pbc_matrix[3] + fz*pbc_matrix[6];
        d31y = fx*pbc_matrix[1] + fy*pbc_matrix[4] + fz*pbc_matrix[7];
        d31z = fx*pbc_matrix[2] + fy*pbc_matrix[5] + fz*pbc_matrix[8];
    }

    // Load NEW (unconstrained) positions
    float nox = new_x[ow1], noy = new_y[ow1], noz = new_z[ow1];
    float nh1x = new_x[hw2], nh1y = new_y[hw2], nh1z = new_z[hw2];
    float nh2x = new_x[hw3], nh2y = new_y[hw3], nh2z = new_z[hw3];

    // New O->H vectors with PBC
    float nd21x = nh1x - nox, nd21y = nh1y - noy, nd21z = nh1z - noz;
    {
        float fx = nd21x*pbc_inv[0] + nd21y*pbc_inv[3] + nd21z*pbc_inv[6];
        float fy = nd21x*pbc_inv[1] + nd21y*pbc_inv[4] + nd21z*pbc_inv[7];
        float fz = nd21x*pbc_inv[2] + nd21y*pbc_inv[5] + nd21z*pbc_inv[8];
        fx -= roundf(fx); fy -= roundf(fy); fz -= roundf(fz);
        nd21x = fx*pbc_matrix[0] + fy*pbc_matrix[3] + fz*pbc_matrix[6];
        nd21y = fx*pbc_matrix[1] + fy*pbc_matrix[4] + fz*pbc_matrix[7];
        nd21z = fx*pbc_matrix[2] + fy*pbc_matrix[5] + fz*pbc_matrix[8];
    }
    float nd31x = nh2x - nox, nd31y = nh2y - noy, nd31z = nh2z - noz;
    {
        float fx = nd31x*pbc_inv[0] + nd31y*pbc_inv[3] + nd31z*pbc_inv[6];
        float fy = nd31x*pbc_inv[1] + nd31y*pbc_inv[4] + nd31z*pbc_inv[7];
        float fz = nd31x*pbc_inv[2] + nd31y*pbc_inv[5] + nd31z*pbc_inv[8];
        fx -= roundf(fx); fy -= roundf(fy); fz -= roundf(fz);
        nd31x = fx*pbc_matrix[0] + fy*pbc_matrix[3] + fz*pbc_matrix[6];
        nd31y = fx*pbc_matrix[1] + fy*pbc_matrix[4] + fz*pbc_matrix[7];
        nd31z = fx*pbc_matrix[2] + fy*pbc_matrix[5] + fz*pbc_matrix[8];
    }

    // Hess optimization: reference from O, not COM
    float a1x = -(nd21x + nd31x) * wh;
    float a1y = -(nd21y + nd31y) * wh;
    float a1z = -(nd21z + nd31z) * wh;

    float b1x = nd21x + a1x, b1y = nd21y + a1y, b1z = nd21z + a1z;
    float c1x = nd31x + a1x, c1y = nd31y + a1y, c1z = nd31z + a1z;

    // Molecular frame from old geometry
    float zx = d21y*d31z - d21z*d31y;
    float zy = d21z*d31x - d21x*d31z;
    float zz = d21x*d31y - d21y*d31x;
    float zinv = rsqrtf(zx*zx + zy*zy + zz*zz + 1e-30f);
    zx *= zinv; zy *= zinv; zz *= zinv;

    float xx = a1y*zz - a1z*zy;
    float xy = a1z*zx - a1x*zz;
    float xz = a1x*zy - a1y*zx;
    float xinv = rsqrtf(xx*xx + xy*xy + xz*xz + 1e-30f);
    xx *= xinv; xy *= xinv; xz *= xinv;

    float yx = zy*xz - zz*xy;
    float yy = zz*xx - zx*xz;
    float yz = zx*xy - zy*xx;

    // Project old geometry into molecular frame (2D: x,y only)
    float b0x = d21x*xx + d21y*xy + d21z*xz;
    float b0y = d21x*yx + d21y*yy + d21z*yz;
    float c0x = d31x*xx + d31y*xy + d31z*xz;
    float c0y = d31x*yx + d31y*yy + d31z*yz;

    // Project new geometry into molecular frame
    float a1z_m = a1x*zx + a1y*zy + a1z*zz;
    float b1x_m = b1x*xx + b1y*xy + b1z*xz;
    float b1y_m = b1x*yx + b1y*yy + b1z*yz;
    float b1z_m = b1x*zx + b1y*zy + b1z*zz;
    float c1x_m = c1x*xx + c1y*xy + c1z*xz;
    float c1y_m = c1x*yx + c1y*yy + c1z*yz;
    float c1z_m = c1x*zx + c1y*zy + c1z*zz;

    // Solve Euler angles phi and psi
    float sin_phi = a1z_m / ra;
    float cos_phi_sq = 1.0f - sin_phi * sin_phi;
    if (cos_phi_sq < 1e-12f) return;
    float cos_phi = sqrtf(cos_phi_sq);

    float sin_psi = (b1z_m - c1z_m) * inv_dHH * 0.5f / cos_phi;
    if (sin_psi > 1.0f) sin_psi = 1.0f;
    if (sin_psi < -1.0f) sin_psi = -1.0f;
    float cos_psi = sqrtf(1.0f - sin_psi * sin_psi);

    // Constrained positions in molecular frame
    float a2y = ra * cos_phi;
    float b2x = -rc * cos_psi;
    float t1 = -rb * cos_phi;
    float t2 = rc * sin_psi * sin_phi;
    float b2y = t1 - t2;
    float c2y = t1 + t2;

    // Solve theta (third Euler angle)
    float alpha = b2x*(b0x - c0x) + b0y*b2y + c0y*c2y;
    float beta  = b2x*(c0y - b0y) + b0x*b2y + c0x*c2y;
    float gamma = b0x*b1y_m - b1x_m*b0y + c0x*c1y_m - c1x_m*c0y;

    float a2 = alpha*alpha + beta*beta;
    float sin_the, cos_the;
    if (a2 < 1e-30f) {
        sin_the = 0.0f;
        cos_the = 1.0f;
    } else {
        sin_the = (alpha*gamma - beta*sqrtf(fmaxf(a2 - gamma*gamma, 0.0f))) / a2;
        if (sin_the > 1.0f) sin_the = 1.0f;
        if (sin_the < -1.0f) sin_the = -1.0f;
        cos_the = sqrtf(1.0f - sin_the*sin_the);
    }

    // Construct constrained positions in molecular frame
    float a3x_m = -a2y * sin_the;
    float a3y_m =  a2y * cos_the;
    float a3z_m =  a1z_m;

    float b3x_m = b2x*cos_the - b2y*sin_the;
    float b3y_m = b2x*sin_the + b2y*cos_the;
    float b3z_m = b1z_m;

    float c3x_m = -b2x*cos_the - c2y*sin_the;
    float c3y_m = -b2x*sin_the + c2y*cos_the;
    float c3z_m = c1z_m;

    // Rotate back to lab frame
    float a3x = xx*a3x_m + yx*a3y_m + zx*a3z_m;
    float a3y = xy*a3x_m + yy*a3y_m + zy*a3z_m;
    float a3z = xz*a3x_m + yz*a3y_m + zz*a3z_m;

    float b3x = xx*b3x_m + yx*b3y_m + zx*b3z_m;
    float b3y = xy*b3x_m + yy*b3y_m + zy*b3z_m;
    float b3z = xz*b3x_m + yz*b3y_m + zz*b3z_m;

    float c3x = xx*c3x_m + yx*c3y_m + zx*c3z_m;
    float c3y = xy*c3x_m + yy*c3y_m + zy*c3z_m;
    float c3z = xz*c3x_m + yz*c3y_m + zz*c3z_m;

    // Position corrections
    float dxO = a3x - a1x, dyO = a3y - a1y, dzO = a3z - a1z;
    float dxH1 = b3x - b1x, dyH1 = b3y - b1y, dzH1 = b3z - b1z;
    float dxH2 = c3x - c1x, dyH2 = c3y - c1y, dzH2 = c3z - c1z;

    // Apply corrections (no atomicAdd needed: atoms non-overlapping across waters)
    new_x[ow1] = nox + dxO;
    new_y[ow1] = noy + dyO;
    new_z[ow1] = noz + dzO;
    new_x[hw2] = nh1x + dxH1;
    new_y[hw2] = nh1y + dyH1;
    new_z[hw2] = nh1z + dzH1;
    new_x[hw3] = nh2x + dxH2;
    new_y[hw3] = nh2y + dyH2;
    new_z[hw3] = nh2z + dzH2;
}
"""

_REMAP_INDICES_KERNEL = r"""
extern "C" __global__
void remap_indices_kernel(
    const int* __restrict__ d_remap,
    int* __restrict__ d_indices,
    int num_indices
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_indices) return;
    d_indices[i] = d_remap[d_indices[i]];
}
"""


class SettleConstraint(ConstraintBase):
    name = 'settle'

    _remap_kernel = None

    @classmethod
    def _get_remap_kernel(cls):
        if cls._remap_kernel is None:
            cls._remap_kernel = cp.RawKernel(_REMAP_INDICES_KERNEL, "remap_indices_kernel")
        return cls._remap_kernel

    def __init__(self, water_triplets, masses, dOH=1.0, dHH=1.63298):
        self.num_waters = len(water_triplets)
        self.dOH = dOH
        self.dHH = dHH

        idx_array = np.array(water_triplets, dtype=np.int32).ravel()
        self.d_water_idx = cp.asarray(idx_array)
        self._n_idx = idx_array.size

        mO = float(masses[water_triplets[0][0]]) if water_triplets else 16.0
        mH = float(masses[water_triplets[0][1]]) if water_triplets else 1.008

        wohh = mO + 2.0 * mH
        self.wh = mH / wohh
        rc = dHH / 2.0
        height = np.sqrt(dOH * dOH - rc * rc)
        self.ra = 2.0 * mH * height / wohh
        self.rb = height - self.ra
        self.rc = rc
        self.inv_dOH = 1.0 / dOH
        self.inv_dHH = 1.0 / dHH

        self._kernel = cp.RawKernel(_SETTLE_KERNEL, "settle_kernel")

    def apply(self, gpu_context, time_step, **kwargs):
        if self.num_waters == 0:
            return
        block = 256
        grid = (self.num_waters + block - 1) // block
        self._kernel((grid,), (block,), (
            gpu_context.d_prev_positions_x,
            gpu_context.d_prev_positions_y,
            gpu_context.d_prev_positions_z,
            gpu_context.d_positions_x,
            gpu_context.d_positions_y,
            gpu_context.d_positions_z,
            gpu_context.d_pbc_matrix,
            gpu_context.d_pbc_inv,
            self.d_water_idx,
            np.int32(self.num_waters),
            np.float32(self.wh),
            np.float32(self.ra),
            np.float32(self.rb),
            np.float32(self.rc),
            np.float32(self.inv_dHH),
            np.float32(time_step),
        ))

    def remap_indices_gpu(self, d_remap):
        if self.num_waters == 0:
            return
        kernel = self._get_remap_kernel()
        tpb = 256
        grid = ((self._n_idx + tpb - 1) // tpb,)
        kernel(grid, (tpb,), (d_remap, self.d_water_idx, np.int32(self._n_idx)))
