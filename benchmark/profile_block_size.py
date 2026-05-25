"""Measure tile pair hit rate for different block sizes (W=32, 16, 8).

Uses the correct Skilling Hilbert kernel for all cases to isolate
the effect of block size from curve type.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n md_analysis python benchmark/profile_block_size.py
"""
import os
import numpy as np
import cupy as cp

from benchmark._data_path import DATA_DIR
PSF_PATH = os.path.join(DATA_DIR, '1M9Z.psf')
PDB_PATH = os.path.join(DATA_DIR, '1M9Z_minimized.pdb')
PRM_PATH = os.path.join(DATA_DIR, 'par_all36_prot.prm')
STR_PATH = os.path.join(DATA_DIR, 'toppar_water_ions.str')

BOX_SIZE = 108.0
CUTOFF = 12.0
SKIN = 1.0

_MORTON_ENCODE_KERNEL = r"""
__device__ unsigned long long morton_split(unsigned int v) {
    v = v & 0x000003FFu;
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v << 8))  & 0x0300F00Fu;
    v = (v | (v << 4))  & 0x030C30C3u;
    v = (v | (v << 2))  & 0x09249249u;
    return (unsigned long long)v;
}

extern "C" __global__
void morton_encode_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    const float* __restrict__ pbc_matrix,
    const float* __restrict__ pbc_inv,
    int number_particles,
    float box_x, float box_y, float box_z,
    unsigned long long* __restrict__ morton_codes
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= number_particles) return;
    float px = pos_x[index], py = pos_y[index], pz = pos_z[index];
    float fx = px*pbc_inv[0]+py*pbc_inv[3]+pz*pbc_inv[6];
    float fy = px*pbc_inv[1]+py*pbc_inv[4]+pz*pbc_inv[7];
    float fz = px*pbc_inv[2]+py*pbc_inv[5]+pz*pbc_inv[8];
    fx-=floorf(fx); fy-=floorf(fy); fz-=floorf(fz);
    float wx=fx*pbc_matrix[0]+fy*pbc_matrix[3]+fz*pbc_matrix[6];
    float wy=fx*pbc_matrix[1]+fy*pbc_matrix[4]+fz*pbc_matrix[7];
    float wz=fx*pbc_matrix[2]+fy*pbc_matrix[5]+fz*pbc_matrix[8];
    unsigned int ix=min((unsigned int)(wx/box_x*1023.0f),1023u);
    unsigned int iy=min((unsigned int)(wy/box_y*1023.0f),1023u);
    unsigned int iz=min((unsigned int)(wz/box_z*1023.0f),1023u);
    morton_codes[index] = morton_split(ix)|(morton_split(iy)<<1)|(morton_split(iz)<<2);
}
"""

_COUNT_PAIRS_TEMPLATE = r"""
extern "C" __global__
void count_pairs_W{W}_kernel(
    const float* __restrict__ sorted_pos_x,
    const float* __restrict__ sorted_pos_y,
    const float* __restrict__ sorted_pos_z,
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    const int* __restrict__ block_atoms,
    const int* __restrict__ tiles,
    const int* __restrict__ interacting_atoms,
    const unsigned int* __restrict__ exclusion_masks,
    float cutoff_sq,
    int num_tiles,
    int num_particles,
    float box_x, float box_y, float box_z,
    float inv_box_x, float inv_box_y, float inv_box_z,
    int* __restrict__ pair_counts
) {{
    int total_warps = (blockDim.x * gridDim.x) / 32;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int tgx = threadIdx.x & 31;
    int tbx = threadIdx.x - tgx;

    int pos = (int)((long long)warp_id * num_tiles / total_warps);
    int end = (int)((long long)(warp_id + 1) * num_tiles / total_warps);

    for (; pos < end; pos++) {{
        int block_x = tiles[pos];

        int gj = interacting_atoms[pos * 32 + tgx];
        float pj_x = 0.0f, pj_y = 0.0f, pj_z = 0.0f;
        if (gj >= 0 && gj < num_particles) {{
            pj_x = pos_x[gj]; pj_y = pos_y[gj]; pj_z = pos_z[gj];
        }}

        int local_count = 0;

        for (int s = 0; s < {W}; s++) {{
            int gi = block_atoms[block_x * {W} + s];
            if (gi < 0 || gi >= num_particles) continue;

            float px_i = sorted_pos_x[block_x * {W} + s];
            float py_i = sorted_pos_y[block_x * {W} + s];
            float pz_i = sorted_pos_z[block_x * {W} + s];

            unsigned int excl_j = exclusion_masks[pos * 32 + tgx];
            bool excluded = (gj < 0 || gj >= num_particles)
                         || ((excl_j >> s) & 1);
            if (!excluded) {{
                float dx = pj_x - px_i;
                float dy = pj_y - py_i;
                float dz = pj_z - pz_i;
                dx -= box_x * roundf(dx * inv_box_x);
                dy -= box_y * roundf(dy * inv_box_y);
                dz -= box_z * roundf(dz * inv_box_z);
                float dist_sq = dx*dx + dy*dy + dz*dz;
                if (dist_sq > 1.0e-12f && dist_sq <= cutoff_sq)
                    local_count++;
            }}
        }}

        for (int offset = 16; offset > 0; offset /= 2)
            local_count += __shfl_down_sync(0xffffffff, local_count, offset);
        if (tgx == 0) pair_counts[pos] = local_count;
    }}
}}
"""

_FIND_TILES_TEMPLATE = r"""
extern "C" __global__ __launch_bounds__(256, 3)
void find_tiles_W{W}_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    const int* __restrict__ block_atoms,
    const float* __restrict__ block_center,
    const float* __restrict__ block_size,
    int num_blocks,
    int num_particles,
    float build_radius_sq,
    float box_x, float box_y, float box_z,
    float inv_box_x, float inv_box_y, float inv_box_z,
    const float* __restrict__ large_block_center,
    const float* __restrict__ large_block_size,
    int* __restrict__ tiles_out,
    int* __restrict__ interacting_atoms_out,
    int* __restrict__ interaction_count,
    int max_tiles
) {{
    int tgx = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int global_warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (global_warp >= num_blocks) return;

    __shared__ int buffer[8 * 256];
    int* my_buf = buffer + warp_in_block * 256;
    int nBuf = 0;

    int bx = global_warp;
    float cx = block_center[bx*3], cy = block_center[bx*3+1], cz = block_center[bx*3+2];
    float sx = block_size[bx*3],   sy = block_size[bx*3+1],   sz = block_size[bx*3+2];
    int my_large_block = bx >> 5;
    int num_large_blocks = (num_blocks + 31) >> 5;

    for (int lb = my_large_block; lb < num_large_blocks; lb++) {{
        bool lb_pass;
        if (lb == my_large_block) {{
            lb_pass = true;
        }} else {{
            float lcx = large_block_center[lb*3];
            float lcy = large_block_center[lb*3+1];
            float lcz = large_block_center[lb*3+2];
            float lsx = large_block_size[lb*3];
            float lsy = large_block_size[lb*3+1];
            float lsz = large_block_size[lb*3+2];
            float dx = lcx - cx, dy = lcy - cy, dz = lcz - cz;
            dx -= box_x * roundf(dx * inv_box_x);
            dy -= box_y * roundf(dy * inv_box_y);
            dz -= box_z * roundf(dz * inv_box_z);
            dx = fmaxf(0.0f, fabsf(dx) - sx - lsx);
            dy = fmaxf(0.0f, fabsf(dy) - sy - lsy);
            dz = fmaxf(0.0f, fabsf(dz) - sz - lsz);
            lb_pass = (dx*dx + dy*dy + dz*dz < build_radius_sq);
        }}
        if (!lb_pass) continue;

        int block2Base = lb << 5;
        {{
            int block2 = block2Base + tgx;
            bool include_block = false;

            if (block2 < num_blocks && block2 > bx) {{
                float bcx2 = block_center[block2*3];
                float bcy2 = block_center[block2*3+1];
                float bcz2 = block_center[block2*3+2];
                float bsx2 = block_size[block2*3];
                float bsy2 = block_size[block2*3+1];
                float bsz2 = block_size[block2*3+2];
                float dx = bcx2 - cx, dy = bcy2 - cy, dz = bcz2 - cz;
                dx -= box_x * roundf(dx * inv_box_x);
                dy -= box_y * roundf(dy * inv_box_y);
                dz -= box_z * roundf(dz * inv_box_z);
                dx = fmaxf(0.0f, fabsf(dx) - sx - bsx2);
                dy = fmaxf(0.0f, fabsf(dy) - sy - bsy2);
                dz = fmaxf(0.0f, fabsf(dz) - sz - bsz2);
                include_block = (dx*dx + dy*dy + dz*dz < build_radius_sq);
            }}

            unsigned int include_flags = __ballot_sync(0xffffffff, include_block);

            while (include_flags != 0) {{
                int i = __ffs(include_flags) - 1;
                include_flags &= include_flags - 1;
                int by = block2Base + i;

                int gj = block_atoms[by * {W} + tgx];
                int interacts = 0;

                if (tgx < {W} && gj >= 0 && gj < num_particles) {{
                    float px_j = pos_x[gj];
                    float py_j = pos_y[gj];
                    float pz_j = pos_z[gj];
                    for (int k = 0; k < {W}; k++) {{
                        int gk = block_atoms[bx * {W} + k];
                        if (gk < 0) continue;
                        float ddx = px_j - pos_x[gk];
                        float ddy = py_j - pos_y[gk];
                        float ddz = pz_j - pos_z[gk];
                        ddx -= box_x * roundf(ddx * inv_box_x);
                        ddy -= box_y * roundf(ddy * inv_box_y);
                        ddz -= box_z * roundf(ddz * inv_box_z);
                        if (ddx*ddx + ddy*ddy + ddz*ddz <= build_radius_sq) {{
                            interacts = 1; break;
                        }}
                    }}
                }}

                unsigned int ballot = __ballot_sync(0xffffffff, interacts);
                int rank = __popc(ballot & ((1u << tgx) - 1));
                if (interacts) my_buf[nBuf + rank] = gj;
                nBuf += __popc(ballot);

                while (nBuf >= 32) {{
                    int ti = 0;
                    if (tgx == 0) ti = atomicAdd(interaction_count, 1);
                    ti = __shfl_sync(0xffffffff, ti, 0);
                    if (ti < max_tiles) {{
                        if (tgx < 1) tiles_out[ti] = bx;
                        interacting_atoms_out[ti * 32 + tgx] = my_buf[tgx];
                    }}
                    for (int s = tgx; s < nBuf - 32; s += 32)
                        my_buf[s] = my_buf[s + 32];
                    nBuf -= 32;
                }}
            }}
        }}
    }}

    // Self-tile
    {{
        int gj = (tgx < {W}) ? block_atoms[bx * {W} + tgx] : -1;
        int interacts = (gj >= 0 && gj < num_particles) ? 1 : 0;
        unsigned int ballot = __ballot_sync(0xffffffff, interacts);
        int rank = __popc(ballot & ((1u << tgx) - 1));
        if (interacts) my_buf[nBuf + rank] = gj;
        nBuf += __popc(ballot);

        while (nBuf >= 32) {{
            int ti = 0;
            if (tgx == 0) ti = atomicAdd(interaction_count, 1);
            ti = __shfl_sync(0xffffffff, ti, 0);
            if (ti < max_tiles) {{
                if (tgx < 1) tiles_out[ti] = bx;
                interacting_atoms_out[ti * 32 + tgx] = my_buf[tgx];
            }}
            for (int s = tgx; s < nBuf - 32; s += 32)
                my_buf[s] = my_buf[s + 32];
            nBuf -= 32;
        }}

        if (nBuf > 0) {{
            int ti = 0;
            if (tgx == 0) ti = atomicAdd(interaction_count, 1);
            ti = __shfl_sync(0xffffffff, ti, 0);
            if (ti < max_tiles) {{
                if (tgx < 1) tiles_out[ti] = bx;
                interacting_atoms_out[ti * 32 + tgx] =
                    (tgx < nBuf) ? my_buf[tgx] : 0x7FFFFFFF;
            }}
        }}
    }}
}}
"""


def build_system(W):
    from mdpy.forcefield.charmm_forcefield import CharmmForcefield

    ff = CharmmForcefield(PSF_PATH, PDB_PATH, [PRM_PATH, STR_PATH])
    topology = ff.create_topology()
    parameter_table = ff.create_parameter_table()

    pbc_matrix = np.eye(3, dtype=np.float64) * BOX_SIZE
    pbc_inv = np.linalg.inv(pbc_matrix)

    raw = ff._pdb.positions.astype(np.float64)
    frac = raw @ pbc_inv
    frac -= np.floor(frac)
    wrapped = frac @ pbc_matrix
    pos_x = np.ascontiguousarray(wrapped[:, 0], dtype=np.float32)
    pos_y = np.ascontiguousarray(wrapped[:, 1], dtype=np.float32)
    pos_z = np.ascontiguousarray(wrapped[:, 2], dtype=np.float32)

    N = topology.num_particles
    tpb = 256

    pbc_flat = np.ascontiguousarray(pbc_matrix.ravel(), dtype=np.float32)
    pbc_inv_flat = np.ascontiguousarray(pbc_inv.ravel(), dtype=np.float32)
    d_pbc = cp.asarray(pbc_flat)
    d_pbc_inv = cp.asarray(pbc_inv_flat)

    d_px = cp.asarray(pos_x, dtype=np.float32).copy()
    d_py = cp.asarray(pos_y, dtype=np.float32).copy()
    d_pz = cp.asarray(pos_z, dtype=np.float32).copy()

    n3 = (N + tpb - 1) // tpb
    wrap_kern = cp.RawKernel(r"""
    extern "C" __global__
    void wrap(float* px, float* py, float* pz, const float* pbc, const float* pbc_inv, int N) {
        int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N)return;
        float x=px[i],y=py[i],z=pz[i];
        float fx=x*pbc_inv[0]+y*pbc_inv[3]+z*pbc_inv[6];
        float fy=x*pbc_inv[1]+y*pbc_inv[4]+z*pbc_inv[7];
        float fz=x*pbc_inv[2]+y*pbc_inv[5]+z*pbc_inv[8];
        fx-=floorf(fx);fy-=floorf(fy);fz-=floorf(fz);
        px[i]=fx*pbc[0]+fy*pbc[3]+fz*pbc[6];
        py[i]=fx*pbc[1]+fy*pbc[4]+fz*pbc[7];
        pz[i]=fx*pbc[2]+fy*pbc[5]+fz*pbc[8];
    }
    """, 'wrap')
    wrap_kern((n3,), (tpb,), (d_px, d_py, d_pz, d_pbc, d_pbc_inv, np.int32(N)))

    pbc_2d = np.asarray(pbc_matrix).reshape(3, 3)
    box_x = abs(float(pbc_2d[0, 0]))
    box_y = abs(float(pbc_2d[1, 1]))
    box_z = abs(float(pbc_2d[2, 2]))

    morton_kern = cp.RawKernel(_MORTON_ENCODE_KERNEL, 'morton_encode_kernel')
    codes = cp.empty(N, dtype=np.uint64)
    nm = (N + tpb - 1) // tpb
    morton_kern((nm,), (tpb,),
        (d_px, d_py, d_pz, d_pbc, d_pbc_inv, np.int32(N),
         np.float32(box_x), np.float32(box_y), np.float32(box_z), codes))

    sorted_indices = cp.argsort(codes).astype(np.int32)

    num_blocks = (N + W - 1) // W
    total_slots = num_blocks * W
    block_atoms = cp.full(total_slots, -1, dtype=np.int32)

    form_kern = cp.RawKernel(r"""
    extern "C" __global__
    void form(const int* si, int N, int W, int nb, int* ba) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= nb*W) return;
        ba[i] = (i < N) ? si[i] : -1;
    }
    """, 'form')
    nf = (total_slots + tpb - 1) // tpb
    form_kern((nf,), (tpb,), (sorted_indices, np.int32(N), np.int32(W),
                              np.int32(num_blocks), block_atoms))

    block_center = cp.empty(num_blocks * 3, dtype=np.float32)
    block_size_d = cp.empty(num_blocks * 3, dtype=np.float32)
    bounds_kern = cp.RawKernel(r"""
    extern "C" __global__
    void bounds(const float* px, const float* py, const float* pz,
                const int* ba, int W, int nb, float* bc, float* bs) {
        int bi=blockIdx.x*blockDim.x+threadIdx.x;
        if(bi>=nb)return;
        float mx=1e30f,my=1e30f,mz=1e30f,Mx=-1e30f,My=-1e30f,Mz=-1e30f;
        for(int s=0;s<W;s++){
            int a=ba[bi*W+s]; if(a<0)continue;
            float x=px[a],y=py[a],z=pz[a];
            mx=fminf(mx,x);Mx=fmaxf(Mx,x);
            my=fminf(my,y);My=fmaxf(My,y);
            mz=fminf(mz,z);Mz=fmaxf(Mz,z);
        }
        bc[bi*3]=.5f*(mx+Mx);bc[bi*3+1]=.5f*(my+My);bc[bi*3+2]=.5f*(mz+Mz);
        bs[bi*3]=.5f*(Mx-mx);bs[bi*3+1]=.5f*(My-my);bs[bi*3+2]=.5f*(Mz-mz);
    }
    """, 'bounds')
    nb_grid = (num_blocks + tpb - 1) // tpb
    bounds_kern((nb_grid,), (tpb,),
        (d_px, d_py, d_pz, block_atoms, np.int32(W), np.int32(num_blocks),
         block_center, block_size_d))

    num_large_blocks = (num_blocks + 31) // 32
    d_lb_center = cp.empty(num_large_blocks * 3, dtype=np.float32)
    d_lb_size = cp.empty(num_large_blocks * 3, dtype=np.float32)
    lb_kern = cp.RawKernel(r"""
    extern "C" __global__
    void lb_bounds(const float* bc, const float* bs, int nb, int nlb,
                   float bx, float by, float bz, float ibx, float iby, float ibz,
                   float* lbc, float* lbs) {
        int lb=blockIdx.x*blockDim.x+threadIdx.x;
        if(lb>=nlb)return;
        int start=lb*32,end=min(start+32,nb);
        float mx=1e30f,my=1e30f,mz=1e30f,Mx=-1e30f,My=-1e30f,Mz=-1e30f;
        for(int b=start;b<end;b++){
            float cx=bc[b*3],cy=bc[b*3+1],cz=bc[b*3+2];
            float sx=bs[b*3],sy=bs[b*3+1],sz=bs[b*3+2];
            if(b>start){
                float rx=bc[start*3],ry=bc[start*3+1],rz=bc[start*3+2];
                float dx=cx-rx,dy=cy-ry,dz=cz-rz;
                dx-=bx*roundf(dx*ibx);dy-=by*roundf(dy*iby);dz-=bz*roundf(dz*ibz);
                cx=rx+dx;cy=ry+dy;cz=rz+dz;
            }
            mx=fminf(mx,cx-sx);my=fminf(my,cy-sy);mz=fminf(mz,cz-sz);
            Mx=fmaxf(Mx,cx+sx);My=fmaxf(My,cy+sy);Mz=fmaxf(Mz,cz+sz);
        }
        lbc[lb*3]=.5f*(mx+Mx);lbc[lb*3+1]=.5f*(my+My);lbc[lb*3+2]=.5f*(mz+Mz);
        lbs[lb*3]=.5f*(Mx-mx);lbs[lb*3+1]=.5f*(My-my);lbs[lb*3+2]=.5f*(Mz-mz);
    }
    """, 'lb_bounds')
    nlb = (num_large_blocks + tpb - 1) // tpb
    lb_kern((nlb,), (tpb,),
        (block_center, block_size_d, np.int32(num_blocks), np.int32(num_large_blocks),
         np.float32(box_x), np.float32(box_y), np.float32(box_z),
         np.float32(1.0/box_x), np.float32(1.0/box_y), np.float32(1.0/box_z),
         d_lb_center, d_lb_size))

    d_counters = cp.zeros(1, dtype=np.int32)
    max_tiles = max(num_blocks * 100, 10000)
    d_tile_buf = cp.empty(max_tiles, dtype=np.int32)
    d_ia_buf = cp.empty(max_tiles * 32, dtype=np.int32)

    find_kern = cp.RawKernel(_FIND_TILES_TEMPLATE.replace('{W}', str(W)),
                              f'find_tiles_W{W}_kernel')
    d_counters[0] = 0
    grid_blocks = max((num_blocks + 7) // 8, 1)
    find_kern((grid_blocks,), (tpb,),
        (d_px, d_py, d_pz, block_atoms, block_center, block_size_d,
         np.int32(num_blocks), np.int32(N),
         np.float32((CUTOFF + SKIN) ** 2),
         np.float32(box_x), np.float32(box_y), np.float32(box_z),
         np.float32(1.0/box_x), np.float32(1.0/box_y), np.float32(1.0/box_z),
         d_lb_center, d_lb_size,
         d_tile_buf, d_ia_buf, d_counters, np.int32(max_tiles)))

    num_tiles = int(d_counters[0])
    d_tiles = d_tile_buf[:num_tiles].copy()
    d_interacting = d_ia_buf[:num_tiles * 32].copy()

    sorted_pos_x = cp.empty(total_slots, dtype=np.float32)
    sorted_pos_y = cp.empty(total_slots, dtype=np.float32)
    sorted_pos_z = cp.empty(total_slots, dtype=np.float32)
    gather_kern = cp.RawKernel(r"""
    extern "C" __global__
    void gather(const float* src, const int* ba, int total, int N, float* dst) {
        int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=total)return;
        int a=ba[i]; dst[i]=(a>=0&&a<N)?src[a]:0.0f;
    }
    """, 'gather')
    g_grid = ((total_slots + tpb - 1) // tpb,)
    gather_kern(g_grid, (tpb,), (d_px, block_atoms, np.int32(total_slots), np.int32(N), sorted_pos_x))
    gather_kern(g_grid, (tpb,), (d_py, block_atoms, np.int32(total_slots), np.int32(N), sorted_pos_y))
    gather_kern(g_grid, (tpb,), (d_pz, block_atoms, np.int32(total_slots), np.int32(N), sorted_pos_z))

    # Build exclusion masks
    from mdpy.core.tile_list import _BUILD_REVERSE_COUNT_KERNEL, _FILL_REVERSE_KERNEL, _BUILD_MASKS_KERNEL

    d_excl_offset = cp.asarray(np.ascontiguousarray(topology.exclusion_offset, dtype=np.int32))
    d_excl_neighbors = cp.asarray(np.ascontiguousarray(topology.exclusion_neighbors, dtype=np.int32))
    d_excl_scale = cp.asarray(np.ascontiguousarray(topology.exclusion_scale, dtype=np.float32))

    d_atom_to_block = cp.full(N, -1, dtype=np.int32)
    d_atom_to_slot = cp.full(N, -1, dtype=np.int32)
    atom_map_kern = cp.RawKernel(r"""
    extern "C" __global__
    void amap(const int* ba, int nb, int W, int* a2b, int* a2s) {
        int bi=blockIdx.x*blockDim.x+threadIdx.x; if(bi>=nb)return;
        for(int s=0;s<W;s++){int a=ba[bi*W+s]; if(a>=0){a2b[a]=bi;a2s[a]=s;}}
    }
    """, 'amap')
    atom_map_kern((nb_grid,), (tpb,),
        (block_atoms, np.int32(num_blocks), np.int32(W), d_atom_to_block, d_atom_to_slot))

    d_rev_offset = cp.zeros(N + 1, dtype=np.int32)
    rev_count_kern = cp.RawKernel(_BUILD_REVERSE_COUNT_KERNEL, 'build_reverse_count_kernel')
    n1 = (N + tpb - 1) // tpb
    rev_count_kern((n1,), (tpb,),
        (d_excl_offset, d_excl_neighbors, np.int32(N), d_rev_offset))
    d_rev_offset = cp.cumsum(d_rev_offset, dtype=np.int32)
    max_rev = int(d_rev_offset[-1])
    d_rev_neighbors = cp.empty(max_rev, dtype=np.int32)
    d_rev_scale = cp.empty(max_rev, dtype=np.float32)
    d_temp = d_rev_offset.copy()
    rev_fill_kern = cp.RawKernel(_FILL_REVERSE_KERNEL, 'fill_reverse_kernel')
    rev_fill_kern((n1,), (tpb,),
        (d_excl_offset, d_excl_neighbors, d_excl_scale,
         d_rev_offset, np.int32(N), d_rev_neighbors, d_rev_scale, d_temp))

    d_excl_masks = cp.empty(num_tiles * 32, dtype=np.uint32)
    d_scaling_masks = cp.empty(num_tiles * 32, dtype=np.uint32)
    mask_kern = cp.RawKernel(_BUILD_MASKS_KERNEL, 'build_masks_kernel')
    total_work = num_tiles * 32
    mask_grid = ((total_work + tpb - 1) // tpb,)
    mask_kern(mask_grid, (tpb,),
        (d_tiles, d_interacting, block_atoms,
         d_atom_to_block, d_atom_to_slot,
         d_excl_offset, d_excl_neighbors, d_excl_scale,
         d_rev_offset, d_rev_neighbors, d_rev_scale,
         np.int32(num_tiles), np.int32(N),
         d_excl_masks, d_scaling_masks))

    # Count pairs
    count_kern = cp.RawKernel(_COUNT_PAIRS_TEMPLATE.replace('{W}', str(W)),
                               f'count_pairs_W{W}_kernel')
    d_pair_counts = cp.zeros(num_tiles, dtype=np.int32)
    cutoff_sq = CUTOFF ** 2
    n_warps = max((num_tiles + 7) // 8, 1)
    count_kern((n_warps,), (tpb,),
        (sorted_pos_x, sorted_pos_y, sorted_pos_z,
         d_px, d_py, d_pz,
         block_atoms, d_tiles, d_interacting, d_excl_masks,
         np.float32(cutoff_sq), np.int32(num_tiles), np.int32(N),
         np.float32(box_x), np.float32(box_y), np.float32(box_z),
         np.float32(1.0/box_x), np.float32(1.0/box_y), np.float32(1.0/box_z),
         d_pair_counts))

    pair_counts = cp.asnumpy(d_pair_counts)
    block_size_np = cp.asnumpy(block_size_d).reshape(-1, 3)

    return num_blocks, num_tiles, pair_counts, block_size_np, N


def main():
    for W in [32, 16, 8]:
        print(f"=== W={W} (block={W} atoms, tile=32 j-atoms) ===")
        num_blocks, num_tiles, pair_counts, block_size_np, N = build_system(W)

        max_pairs_per_tile = W * 32
        total_pairs = int(pair_counts.sum())
        total_checked = num_tiles * max_pairs_per_tile
        rate = total_pairs / total_checked if total_checked > 0 else 0

        per_tile_rate = pair_counts.astype(np.float64) / max_pairs_per_tile

        dims = np.stack([np.maximum(2.0 * block_size_np[:, 0], 1e-6),
                         np.maximum(2.0 * block_size_np[:, 1], 1e-6),
                         np.maximum(2.0 * block_size_np[:, 2], 1e-6)], axis=-1)
        sorted_dims = np.sort(dims, axis=-1)
        aspect = sorted_dims[:, 2] / np.maximum(sorted_dims[:, 0], 1e-10)
        vol = np.prod(dims, axis=-1)

        print(f"  Blocks: {num_blocks}")
        print(f"  Tiles: {num_tiles}")
        print(f"  Pairs/tile: {W}×32 = {max_pairs_per_tile}")
        print(f"  Block aspect ratio: mean={aspect.mean():.2f}, median={np.median(aspect):.2f}")
        print(f"  Block volume: mean={vol.mean():.0f}, median={np.median(vol):.0f}")
        print(f"  Total interacting pairs: {total_pairs:,}")
        print(f"  Total pairs checked: {total_checked:,}")
        print(f"  Hit rate: {rate:.4f} ({rate*100:.1f}%)")
        print(f"  Per-tile hit rate: mean={np.mean(per_tile_rate):.4f}, "
              f"median={np.median(per_tile_rate):.4f}, "
              f"p10={np.percentile(per_tile_rate,10):.4f}, "
              f"p90={np.percentile(per_tile_rate,90):.4f}")
        print()


if __name__ == '__main__':
    main()
