import cupy as cp
import numpy as np
from .constraint_base import ConstraintBase

_LINCS_KERNEL = r"""
extern "C" __global__
void lincs_kernel(
    const float* __restrict__ old_x,
    const float* __restrict__ old_y,
    const float* __restrict__ old_z,
    float* __restrict__ pos_x,
    float* __restrict__ pos_y,
    float* __restrict__ pos_z,
    const float* __restrict__ pbc_matrix,
    const float* __restrict__ pbc_inv,
    const int* __restrict__ con_idx,
    const float* __restrict__ target_len,
    const float* __restrict__ inv_mass_i,
    const float* __restrict__ inv_mass_j,
    const float* __restrict__ blc_arr,
    const int* __restrict__ coupled_counts,
    const int* __restrict__ coupled_indices,
    const float* __restrict__ mass_factors,
    float* __restrict__ matrix_a,
    int num_constraints,
    int max_coupled,
    int num_constraint_threads,
    int expansion_order,
    int num_iterations
) {
    extern __shared__ float sm[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;
    int block_offset = blockIdx.x * blockDim.x;
    bool is_dummy = (tid >= num_constraints);

    int ai_s = -1, aj_s = -1;
    float d0 = 0.0f, blc = 0.0f, imi = 0.0f, imj = 0.0f;
    float rcx = 0.0f, rcy = 0.0f, rcz = 0.0f;

    if (!is_dummy) {
        ai_s = con_idx[tid * 2 + 0];
        aj_s = con_idx[tid * 2 + 1];
        d0 = target_len[tid];
        blc = blc_arr[tid];
        imi = inv_mass_i[tid];
        imj = inv_mass_j[tid];
    }

    // Phase 1: reference direction from old positions
    if (!is_dummy) {
        float ox = old_x[ai_s], oy = old_y[ai_s], oz = old_z[ai_s];
        float jx = old_x[aj_s], jy = old_y[aj_s], jz = old_z[aj_s];
        float dx = jx - ox, dy = jy - oy, dz = jz - oz;
        float fx = dx*pbc_inv[0] + dy*pbc_inv[3] + dz*pbc_inv[6];
        float fy = dx*pbc_inv[1] + dy*pbc_inv[4] + dz*pbc_inv[7];
        float fz = dx*pbc_inv[2] + dy*pbc_inv[5] + dz*pbc_inv[8];
        fx -= roundf(fx); fy -= roundf(fy); fz -= roundf(fz);
        dx = fx*pbc_matrix[0] + fy*pbc_matrix[3] + fz*pbc_matrix[6];
        dy = fx*pbc_matrix[1] + fy*pbc_matrix[4] + fz*pbc_matrix[7];
        dz = fx*pbc_matrix[2] + fy*pbc_matrix[5] + fz*pbc_matrix[8];
        float inv_d = rsqrtf(dx*dx + dy*dy + dz*dz + 1e-30f);
        rcx = dx * inv_d; rcy = dy * inv_d; rcz = dz * inv_d;
        sm[lid*3+0] = rcx; sm[lid*3+1] = rcy; sm[lid*3+2] = rcz;
    }
    __syncthreads();

    // Phase 2: build coupling matrix A_ij = mf * (rc_i . rc_j)
    if (!is_dummy) {
        int nc = coupled_counts[tid];
        for (int n = 0; n < nc; n++) {
            int c_idx = coupled_indices[n * num_constraint_threads + tid];
            int c_lid = c_idx - block_offset;
            float mf = mass_factors[n * num_constraint_threads + tid];
            float r1x = sm[c_lid*3+0], r1y = sm[c_lid*3+1], r1z = sm[c_lid*3+2];
            matrix_a[n * num_constraint_threads + tid] = mf * (rcx*r1x + rcy*r1y + rcz*r1z);
        }
    }
    __syncthreads();

    // Phase 3: initial RHS = blc * (rc . delta_new - d0)
    float sol = 0.0f;
    if (!is_dummy) {
        float nix = pos_x[ai_s], niy = pos_y[ai_s], niz = pos_z[ai_s];
        float njx = pos_x[aj_s], njy = pos_y[aj_s], njz = pos_z[aj_s];
        float dx = njx - nix, dy = njy - niy, dz = njz - niz;
        float fx = dx*pbc_inv[0] + dy*pbc_inv[3] + dz*pbc_inv[6];
        float fy = dx*pbc_inv[1] + dy*pbc_inv[4] + dz*pbc_inv[7];
        float fz = dx*pbc_inv[2] + dy*pbc_inv[5] + dz*pbc_inv[8];
        fx -= roundf(fx); fy -= roundf(fy); fz -= roundf(fz);
        dx = fx*pbc_matrix[0] + fy*pbc_matrix[3] + fz*pbc_matrix[6];
        dy = fx*pbc_matrix[1] + fy*pbc_matrix[4] + fz*pbc_matrix[7];
        dz = fx*pbc_matrix[2] + fy*pbc_matrix[5] + fz*pbc_matrix[8];
        sol = blc * (rcx*dx + rcy*dy + rcz*dz - d0);
    }

    // Phase 4: Neumann series  sol = (I + A + A^2 + ... + A^L) * rhs
    float* sm_rhs = sm;
    sm_rhs[lid + blockDim.x * 0] = sol;
    __syncthreads();
    for (int rec = 0; rec < expansion_order; rec++) {
        float mvb = 0.0f;
        if (!is_dummy) {
            int nc = coupled_counts[tid];
            for (int n = 0; n < nc; n++) {
                int c_idx = coupled_indices[n * num_constraint_threads + tid];
                int c_lid = c_idx - block_offset;
                float a_val = matrix_a[n * num_constraint_threads + tid];
                mvb += a_val * sm_rhs[c_lid + blockDim.x * (rec % 2)];
            }
        }
        sm_rhs[lid + blockDim.x * ((rec+1) % 2)] = mvb;
        __syncthreads();
        sol += mvb;
    }

    // Phase 5: first coordinate update
    if (!is_dummy) {
        float lagrange = sol;
        float ci = lagrange * imi;
        float cj = -lagrange * imj;
        atomicAdd(&pos_x[ai_s], rcx*ci);
        atomicAdd(&pos_y[ai_s], rcy*ci);
        atomicAdd(&pos_z[ai_s], rcz*ci);
        atomicAdd(&pos_x[aj_s], rcx*cj);
        atomicAdd(&pos_y[aj_s], rcy*cj);
        atomicAdd(&pos_z[aj_s], rcz*cj);
    }
    __syncthreads();

    // Phase 6: centripetal correction iterations
    for (int iter = 0; iter < num_iterations; iter++) {
        float proj = 0.0f;
        if (!is_dummy) {
            float nix = pos_x[ai_s], niy = pos_y[ai_s], niz = pos_z[ai_s];
            float njx = pos_x[aj_s], njy = pos_y[aj_s], njz = pos_z[aj_s];
            float dx = njx - nix, dy = njy - niy, dz = njz - niz;
            float fx = dx*pbc_inv[0] + dy*pbc_inv[3] + dz*pbc_inv[6];
            float fy = dx*pbc_inv[1] + dy*pbc_inv[4] + dz*pbc_inv[7];
            float fz = dx*pbc_inv[2] + dy*pbc_inv[5] + dz*pbc_inv[8];
            fx -= roundf(fx); fy -= roundf(fy); fz -= roundf(fz);
            dx = fx*pbc_matrix[0] + fy*pbc_matrix[3] + fz*pbc_matrix[6];
            dy = fx*pbc_matrix[1] + fy*pbc_matrix[4] + fz*pbc_matrix[7];
            dz = fx*pbc_matrix[2] + fy*pbc_matrix[5] + fz*pbc_matrix[8];
            float d_sq = dx*dx + dy*dy + dz*dz;
            float dlen2 = 2.0f*d0*d0 - d_sq;
            if (dlen2 > 0.0f) {
                proj = blc * (d0 - sqrtf(dlen2));
            } else {
                proj = blc * d0;
            }
        }
        float sol_iter = proj;
        sm_rhs[lid + blockDim.x * 0] = proj;
        __syncthreads();
        for (int rec = 0; rec < expansion_order; rec++) {
            float mvb = 0.0f;
            if (!is_dummy) {
                int nc = coupled_counts[tid];
                for (int n = 0; n < nc; n++) {
                    int c_idx = coupled_indices[n * num_constraint_threads + tid];
                    int c_lid = c_idx - block_offset;
                    float a_val = matrix_a[n * num_constraint_threads + tid];
                    mvb += a_val * sm_rhs[c_lid + blockDim.x * (rec % 2)];
                }
            }
            sm_rhs[lid + blockDim.x * ((rec+1) % 2)] = mvb;
            __syncthreads();
            sol_iter += mvb;
        }
        if (!is_dummy) {
            float dl = sol_iter;
            float ci = dl * imi;
            float cj = -dl * imj;
            atomicAdd(&pos_x[ai_s], rcx*ci);
            atomicAdd(&pos_y[ai_s], rcy*ci);
            atomicAdd(&pos_z[ai_s], rcz*ci);
            atomicAdd(&pos_x[aj_s], rcx*cj);
            atomicAdd(&pos_y[aj_s], rcy*cj);
            atomicAdd(&pos_z[aj_s], rcz*cj);
        }
        __syncthreads();
    }
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
    int val = d_indices[i];
    if (val >= 0) {
        d_indices[i] = d_remap[val];
    }
}
"""


def _build_coupling_data(constraint_pairs, masses, target_lengths, block_size=256):
    num_constraints = len(constraint_pairs)
    if num_constraints == 0:
        e = lambda dt: np.array([], dtype=dt)
        return (e(np.int32), e(np.float32), e(np.float32), e(np.float32),
                e(np.float32), e(np.int32), e(np.int32), e(np.float32),
                e(np.float32), 0, 0)

    atom_to_con = {}
    for c, (i, j) in enumerate(constraint_pairs):
        atom_to_con.setdefault(i, []).append(c)
        atom_to_con.setdefault(j, []).append(c)

    con_coupled = [set() for _ in range(num_constraints)]
    for atom, clist in atom_to_con.items():
        for a in range(len(clist)):
            for b in range(a + 1, len(clist)):
                con_coupled[clist[a]].add(clist[b])
                con_coupled[clist[b]].add(clist[a])

    visited = [False] * num_constraints
    groups = []
    for s in range(num_constraints):
        if visited[s]:
            continue
        group = []
        stack = [s]
        while stack:
            c = stack.pop()
            if visited[c]:
                continue
            visited[c] = True
            group.append(c)
            for nb in con_coupled[c]:
                if not visited[nb]:
                    stack.append(nb)
        groups.append(group)

    split_map = [0] * num_constraints
    next_slot = 0
    for group in groups:
        block_start = ((next_slot + block_size - 1) // block_size) * block_size
        pos = block_start
        for orig in group:
            split_map[orig] = pos
            pos += 1
        next_slot = pos

    num_ct = ((next_slot + block_size - 1) // block_size) * block_size

    con_idx = np.full((num_ct, 2), -1, dtype=np.int32)
    inv_mi = np.zeros(num_ct, dtype=np.float32)
    inv_mj = np.zeros(num_ct, dtype=np.float32)
    blc_arr = np.zeros(num_ct, dtype=np.float32)
    tl_arr = np.zeros(num_ct, dtype=np.float32)

    for orig in range(num_constraints):
        np_ = split_map[orig]
        i, j = constraint_pairs[orig]
        con_idx[np_, 0] = i
        con_idx[np_, 1] = j
        inv_mi[np_] = 1.0 / float(masses[i])
        inv_mj[np_] = 1.0 / float(masses[j])
        blc_arr[np_] = 1.0 / (inv_mi[np_] + inv_mj[np_])
        tl_arr[np_] = target_lengths[orig]

    max_c = 1
    for orig in range(num_constraints):
        max_c = max(max_c, len(con_coupled[orig]))

    cc_counts = np.zeros(num_ct, dtype=np.int32)
    cc_indices = np.zeros(max_c * num_ct, dtype=np.int32)
    mf_arr = np.zeros(max_c * num_ct, dtype=np.float32)
    mat_a = np.zeros(max_c * num_ct, dtype=np.float32)

    for orig in range(num_constraints):
        np_ = split_map[orig]
        i, j = constraint_pairs[orig]
        blc_i = blc_arr[np_]
        coupled = sorted(con_coupled[orig])
        cc_counts[np_] = len(coupled)
        for n, c_orig in enumerate(coupled):
            c_np = split_map[c_orig]
            ci, cj = constraint_pairs[c_orig]
            blc_c = blc_arr[c_np]
            shared = None
            sign = 1.0
            if i == ci:
                shared = i
                sign = -1.0
            elif j == cj:
                shared = j
                sign = -1.0
            elif i == cj:
                shared = i
                sign = 1.0
            elif j == ci:
                shared = j
                sign = 1.0
            if shared is not None:
                inv_ms = 1.0 / float(masses[shared])
                mf = sign * inv_ms * blc_i * blc_c
                cc_indices[n * num_ct + np_] = c_np
                mf_arr[n * num_ct + np_] = mf

    return (con_idx.ravel(), tl_arr, inv_mi, inv_mj, blc_arr,
            cc_counts, cc_indices, mf_arr, mat_a,
            num_ct, max_c)


class LincsConstraint(ConstraintBase):
    name = 'lincs'

    _remap_kernel = None

    @classmethod
    def _get_remap_kernel(cls):
        if cls._remap_kernel is None:
            cls._remap_kernel = cp.RawKernel(_REMAP_INDICES_KERNEL, "remap_indices_kernel")
        return cls._remap_kernel

    def __init__(self, constraint_pairs, target_lengths, masses,
                 expansion_order=4, num_iterations=1):
        self.num_constraints = len(constraint_pairs)
        self.expansion_order = expansion_order
        self.num_iterations = num_iterations

        (con_idx, tl_arr, inv_mi, inv_mj, blc_arr,
         cc_counts, cc_indices, mf_arr, mat_a,
         num_ct, max_c) = _build_coupling_data(
            list(constraint_pairs), masses, list(target_lengths))

        self.num_constraint_threads = num_ct
        self.max_coupled = max_c

        self.d_con_idx = cp.asarray(con_idx)
        self.d_target_len = cp.asarray(tl_arr)
        self.d_inv_mass_i = cp.asarray(inv_mi)
        self.d_inv_mass_j = cp.asarray(inv_mj)
        self.d_blc = cp.asarray(blc_arr)
        self.d_coupled_counts = cp.asarray(cc_counts)
        self.d_coupled_indices = cp.asarray(cc_indices)
        self.d_mass_factors = cp.asarray(mf_arr)
        self.d_matrix_a = cp.asarray(mat_a)

        self._n_idx = con_idx.size
        self._kernel = cp.RawKernel(_LINCS_KERNEL, "lincs_kernel")

    def apply(self, gpu_context, dt, **kwargs):
        if self.num_constraints == 0:
            return
        block = 256
        grid = (self.num_constraint_threads + block - 1) // block
        shared_mem = 3 * block * 4
        self._kernel((grid,), (block,), (
            gpu_context.d_prev_positions_x,
            gpu_context.d_prev_positions_y,
            gpu_context.d_prev_positions_z,
            gpu_context.d_positions_x,
            gpu_context.d_positions_y,
            gpu_context.d_positions_z,
            gpu_context.d_pbc_matrix,
            gpu_context.d_pbc_inv,
            self.d_con_idx,
            self.d_target_len,
            self.d_inv_mass_i,
            self.d_inv_mass_j,
            self.d_blc,
            self.d_coupled_counts,
            self.d_coupled_indices,
            self.d_mass_factors,
            self.d_matrix_a,
            np.int32(self.num_constraints),
            np.int32(self.max_coupled),
            np.int32(self.num_constraint_threads),
            np.int32(self.expansion_order),
            np.int32(self.num_iterations),
        ), shared_mem=shared_mem)

    def remap_indices_gpu(self, d_remap):
        if self.num_constraints == 0:
            return
        kernel = self._get_remap_kernel()
        tpb = 256
        n = self.d_con_idx.size
        grid = ((n + tpb - 1) // tpb,)
        kernel(grid, (tpb,), (d_remap, self.d_con_idx, np.int32(n)))
