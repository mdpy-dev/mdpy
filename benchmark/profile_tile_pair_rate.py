"""Measure true tile pair interaction rate: how many of the 32x32=1024 pairs
per tile actually have distance < cutoff.

This is the real occupancy metric: the fraction of GPU thread work that
produces useful results vs. being wasted on pairs outside cutoff.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n md_analysis python benchmark/profile_tile_pair_rate.py
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

_HILBERT_ENCODE_KERNEL = r"""
extern "C" __global__
void hilbert_encode_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    const float* __restrict__ pbc_matrix,
    const float* __restrict__ pbc_inv,
    int number_particles,
    float box_x, float box_y, float box_z,
    unsigned long long* __restrict__ hilbert_codes
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
    unsigned int X=ix,Y=iy,Z=iz;
    const int p=10;
    unsigned int M=1u<<(p-1),Q=M;
    while(Q>1){
        unsigned int P=Q-1;
        if(X&Q){X^=P;}
        if(Y&Q){X^=P;} else{unsigned int t=(X^Y)&P;X^=t;Y^=t;}
        if(Z&Q){X^=P;} else{unsigned int t=(X^Z)&P;X^=t;Z^=t;}
        Q>>=1;
    }
    Y^=X;Z^=Y;
    unsigned int t=0;Q=M;
    while(Q>1){if(Z&Q)t^=(Q-1);Q>>=1;}
    X^=t;Y^=t;Z^=t;
    unsigned long long h=0;
    for(int i=p-1;i>=0;i--){
        h=(h<<1)|((X>>i)&1);
        h=(h<<1)|((Y>>i)&1);
        h=(h<<1)|((Z>>i)&1);
    }
    hilbert_codes[index]=h;
}
"""

_COUNT_PAIRS_KERNEL = r"""
extern "C" __global__
void count_pairs_kernel(
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
) {
    int total_warps = (blockDim.x * gridDim.x) / 32;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int tgx = threadIdx.x & 31;
    int tbx = threadIdx.x - tgx;

    int pos = (int)((long long)warp_id * num_tiles / total_warps);
    int end = (int)((long long)(warp_id + 1) * num_tiles / total_warps);

    __shared__ int atom_j_shared[256];

    for (; pos < end; pos++) {
        int block_x = tiles[pos];
        int gi = block_atoms[block_x * 32 + tgx];
        float px_i = sorted_pos_x[block_x * 32 + tgx];
        float py_i = sorted_pos_y[block_x * 32 + tgx];
        float pz_i = sorted_pos_z[block_x * 32 + tgx];

        int gj = interacting_atoms[pos * 32 + tgx];
        float pj_x = 0.0f, pj_y = 0.0f, pj_z = 0.0f;
        if (gj >= 0 && gj < num_particles) {
            pj_x = pos_x[gj];
            pj_y = pos_y[gj];
            pj_z = pos_z[gj];
        }
        atom_j_shared[threadIdx.x] = gj;

        int local_count = 0;

        for (int j = 0; j < 32; j++) {
            int atom2 = atom_j_shared[tbx + j];
            unsigned int excl_j = exclusion_masks[pos * 32 + j];
            bool excluded = (atom2 < 0 || atom2 >= num_particles)
                         || ((excl_j >> tgx) & 1);
            if (excluded) continue;
            if (gi < 0 || gi >= num_particles) continue;

            float dx = pj_x - px_i;
            float dy = pj_y - py_i;
            float dz = pj_z - pz_i;
            dx -= box_x * roundf(dx * inv_box_x);
            dy -= box_y * roundf(dy * inv_box_y);
            dz -= box_z * roundf(dz * inv_box_z);
            float dist_sq = dx*dx + dy*dy + dz*dz;

            if (dist_sq > 1.0e-12f && dist_sq <= cutoff_sq) {
                local_count++;
            }
        }

        // Sum across warp using shuffle
        for (int offset = 16; offset > 0; offset /= 2) {
            local_count += __shfl_down_sync(0xffffffff, local_count, offset);
        }
        if (tgx == 0) {
            pair_counts[pos] = local_count;
        }
    }
}
"""


def build_tile_list(curve_type, pos_x, pos_y, pos_z, pbc_matrix, pbc_inv,
                    topology, N):
    from mdpy.core.tile_list import TileList, _MORTON_ENCODE_KERNEL

    tpb = 256
    pbc_2d = np.asarray(pbc_matrix).reshape(3, 3)
    box_x = abs(float(pbc_2d[0, 0]))
    box_y = abs(float(pbc_2d[1, 1]))
    box_z = abs(float(pbc_2d[2, 2]))

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
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;
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

    if curve_type == 'morton':
        kern = cp.RawKernel(_MORTON_ENCODE_KERNEL, 'morton_encode_kernel')
    else:
        kern = cp.RawKernel(_HILBERT_ENCODE_KERNEL, 'hilbert_encode_kernel')

    codes = cp.empty(N, dtype=np.uint64)
    nm = (N + tpb - 1) // tpb
    kern((nm,), (tpb,),
         (d_px, d_py, d_pz, d_pbc, d_pbc_inv, np.int32(N),
          np.float32(box_x), np.float32(box_y), np.float32(box_z), codes))

    sorted_indices = cp.argsort(codes).astype(np.int32)
    W = 32
    num_blocks = (N + W - 1) // W
    total_slots = num_blocks * W
    block_atoms = cp.full(total_slots, -1, dtype=np.int32)

    form_kern = cp.RawKernel(r"""
    extern "C" __global__
    void form(const int* si, int N, int nb, int* ba) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= nb*32) return;
        ba[i] = (i < N) ? si[i] : -1;
    }
    """, 'form')
    nf = (total_slots + tpb - 1) // tpb
    form_kern((nf,), (tpb,), (sorted_indices, np.int32(N), np.int32(num_blocks), block_atoms))

    block_center = cp.empty(num_blocks * 3, dtype=np.float32)
    block_size_d = cp.empty(num_blocks * 3, dtype=np.float32)
    bounds_kern = cp.RawKernel(r"""
    extern "C" __global__
    void bounds(const float* px, const float* py, const float* pz,
                const int* ba, int nb, float* bc, float* bs) {
        int bi=blockIdx.x*blockDim.x+threadIdx.x;
        if(bi>=nb)return;
        float mx=1e30f,my=1e30f,mz=1e30f,Mx=-1e30f,My=-1e30f,Mz=-1e30f;
        for(int s=0;s<32;s++){
            int a=ba[bi*32+s]; if(a<0)continue;
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
                (d_px, d_py, d_pz, block_atoms, np.int32(num_blocks),
                 block_center, block_size_d))

    tl = TileList(CUTOFF, SKIN)
    tl._ensure_kernels()

    tl.num_particles = N
    tl.num_blocks = num_blocks
    tl.d_block_atoms = block_atoms
    tl.d_block_center = block_center
    tl.d_block_size = block_size_d
    tl._upload_pbc(pbc_matrix, pbc_inv)
    tl._box_x = box_x
    tl._box_y = box_y
    tl._box_z = box_z
    tl._inv_box_x = 1.0 / box_x
    tl._inv_box_y = 1.0 / box_y
    tl._inv_box_z = 1.0 / box_z

    num_large_blocks = (num_blocks + 31) // 32
    tl.num_large_blocks = num_large_blocks
    tl.d_large_block_center = cp.empty(num_large_blocks * 3, dtype=np.float32)
    tl.d_large_block_size = cp.empty(num_large_blocks * 3, dtype=np.float32)
    nlb = (num_large_blocks + tpb - 1) // tpb
    tl._kernels['large_block_bounds']((nlb,), (tpb,),
        (tl.d_block_center, tl.d_block_size,
         np.int32(num_blocks), np.int32(num_large_blocks),
         np.float32(box_x), np.float32(box_y), np.float32(box_z),
         np.float32(1.0/box_x), np.float32(1.0/box_y), np.float32(1.0/box_z),
         tl.d_large_block_center, tl.d_large_block_size))

    tl.d_atom_to_block = cp.full(N, -1, dtype=np.int32)
    tl.d_atom_to_slot = cp.full(N, -1, dtype=np.int32)
    tl._kernels['atom_map']((nb_grid,), (tpb,),
        (tl.d_block_atoms, np.int32(num_blocks), np.int32(W),
         tl.d_atom_to_block, tl.d_atom_to_slot))

    tl._find_interacting_blocks((d_px, d_py, d_pz), pbc_matrix)
    tl._build_masks_gpu(topology)

    sorted_pos_x = cp.empty(total_slots, dtype=np.float32)
    sorted_pos_y = cp.empty(total_slots, dtype=np.float32)
    sorted_pos_z = cp.empty(total_slots, dtype=np.float32)
    gather_kern = tl._kernels['gather_sorted']
    g_grid = ((total_slots + tpb - 1) // tpb,)
    gather_kern(g_grid, (tpb,), (d_px, tl.d_block_atoms,
                                np.int32(total_slots), np.int32(N), sorted_pos_x))
    gather_kern(g_grid, (tpb,), (d_py, tl.d_block_atoms,
                                np.int32(total_slots), np.int32(N), sorted_pos_y))
    gather_kern(g_grid, (tpb,), (d_pz, tl.d_block_atoms,
                                np.int32(total_slots), np.int32(N), sorted_pos_z))

    return tl, d_px, d_py, d_pz, sorted_pos_x, sorted_pos_y, sorted_pos_z


def count_tile_pairs(tl, d_px, d_py, d_pz, sorted_pos_x, sorted_pos_y, sorted_pos_z):
    count_kern = cp.RawKernel(_COUNT_PAIRS_KERNEL, 'count_pairs_kernel')

    num_tiles = tl.num_tiles
    if num_tiles == 0:
        return np.array([])

    d_pair_counts = cp.zeros(num_tiles, dtype=np.int32)

    cutoff_sq = CUTOFF ** 2
    tpb = 256
    n_warps = (num_tiles + 7) // 8
    grid = (n_warps,)

    count_kern(grid, (tpb,),
        (sorted_pos_x, sorted_pos_y, sorted_pos_z,
         d_px, d_py, d_pz,
         tl.d_block_atoms, tl.d_tiles, tl.d_interacting_atoms,
         tl.d_exclusion_masks,
         np.float32(cutoff_sq), np.int32(num_tiles), np.int32(tl.num_particles),
         np.float32(tl._box_x), np.float32(tl._box_y), np.float32(tl._box_z),
         np.float32(tl._inv_box_x), np.float32(tl._inv_box_y), np.float32(tl._inv_box_z),
         d_pair_counts))

    return cp.asnumpy(d_pair_counts)


def main():
    from mdpy.forcefield.charmm_forcefield import CharmmForcefield

    print("Loading 1M9Z...")
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
    print(f"Atoms: {N}, Box: {BOX_SIZE}, Cutoff: {CUTOFF}")
    print()

    for curve in ['morton', 'hilbert']:
        print(f"=== {curve.upper()} ===")
        tl, d_px, d_py, d_pz, sp_x, sp_y, sp_z = build_tile_list(
            curve, pos_x, pos_y, pos_z, pbc_matrix, pbc_inv, topology, N)

        pair_counts = count_tile_pairs(tl, d_px, d_py, d_pz, sp_x, sp_y, sp_z)
        num_tiles = len(pair_counts)

        if num_tiles == 0:
            print("  No tiles")
            continue

        total_pairs = int(pair_counts.sum())
        max_possible = num_tiles * 1024
        rate = total_pairs / max_possible

        per_tile_rate = pair_counts.astype(np.float64) / 1024.0

        print(f"  Tiles: {num_tiles}")
        print(f"  Total interacting pairs: {total_pairs:,}")
        print(f"  Total pairs checked (32x32 per tile): {max_possible:,}")
        print(f"  Hit rate (interacting / checked): {rate:.4f} ({rate*100:.2f}%)")
        print(f"  Per-tile hit rate:")
        print(f"    mean:   {np.mean(per_tile_rate):.4f}")
        print(f"    median: {np.median(per_tile_rate):.4f}")
        print(f"    p10:    {np.percentile(per_tile_rate, 10):.4f}")
        print(f"    p25:    {np.percentile(per_tile_rate, 25):.4f}")
        print(f"    p75:    {np.percentile(per_tile_rate, 75):.4f}")
        print(f"    p90:    {np.percentile(per_tile_rate, 90):.4f}")
        print(f"    min:    {np.min(per_tile_rate):.4f}")
        print(f"    max:    {np.max(per_tile_rate):.4f}")

        hist, edges = np.histogram(pair_counts, bins=[0,1,50,100,200,400,600,800,1024])
        print(f"  Distribution of pair count per tile:")
        labels = ['0', '1-49', '50-99', '100-199', '200-399', '400-599', '600-799', '800-1023']
        for i, (label, count) in enumerate(zip(labels, hist)):
            pct = count / num_tiles * 100
            print(f"    {label:>10s}: {count:6d} ({pct:5.1f}%)")
        print()


if __name__ == '__main__':
    main()
