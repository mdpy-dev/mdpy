"""Quantify Hilbert vs Morton: block shape AND full tile list comparison.

Uses the correct Skilling algorithm for 3D Hilbert encoding on GPU.
Compares block AABB stats + tile occupancy + tile count for both curves.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n md_analysis python benchmark/profile_hilbert_vs_morton.py
"""
import os
import time

import cupy as cp
import numpy as np

from benchmark._data_path import DATA_DIR
PSF_PATH = os.path.join(DATA_DIR, '1M9Z.psf')
PDB_PATH = os.path.join(DATA_DIR, '1M9Z_minimized.pdb')
PRM_PATH = os.path.join(DATA_DIR, 'par_all36_prot.prm')
STR_PATH = os.path.join(DATA_DIR, 'toppar_water_ions.str')

BOX_SIZE = 108.0
CUTOFF = 12.0
SKIN = 1.0
W = 32

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

    float px = pos_x[index];
    float py = pos_y[index];
    float pz = pos_z[index];

    float fx = px * pbc_inv[0] + py * pbc_inv[3] + pz * pbc_inv[6];
    float fy = px * pbc_inv[1] + py * pbc_inv[4] + pz * pbc_inv[7];
    float fz = px * pbc_inv[2] + py * pbc_inv[5] + pz * pbc_inv[8];

    fx = fx - floorf(fx);
    fy = fy - floorf(fy);
    fz = fz - floorf(fz);

    float wx = fx * pbc_matrix[0] + fy * pbc_matrix[3] + fz * pbc_matrix[6];
    float wy = fx * pbc_matrix[1] + fy * pbc_matrix[4] + fz * pbc_matrix[7];
    float wz = fx * pbc_matrix[2] + fy * pbc_matrix[5] + fz * pbc_matrix[8];

    unsigned int ix = min((unsigned int)(wx / box_x * 1023.0f), 1023u);
    unsigned int iy = min((unsigned int)(wy / box_y * 1023.0f), 1023u);
    unsigned int iz = min((unsigned int)(wz / box_z * 1023.0f), 1023u);

    /* Skilling's algorithm for 3D Hilbert index, 10-bit coordinates.
       Reference: J. Skilling, "Programming the Hilbert curve",
       AIP Conf. Proc. 707, 381 (2004).

       Step 1: Inverse undo excess work
       Step 2: Gray encode
       Step 3: Transpose to integer
    */

    unsigned int X = ix, Y = iy, Z = iz;
    const int p = 10;

    /* Step 1: Inverse undo excess work (Skilling algorithm) */
    unsigned int M = 1u << (p - 1);
    unsigned int Q = M;
    while (Q > 1) {
        unsigned int P = Q - 1;
        /* dimension 0 (X) */
        if (X & Q) { X ^= P; }
        /* dimension 1 (Y) */
        if (Y & Q) { X ^= P; }
        else { unsigned int t = (X ^ Y) & P; X ^= t; Y ^= t; }
        /* dimension 2 (Z) */
        if (Z & Q) { X ^= P; }
        else { unsigned int t = (X ^ Z) & P; X ^= t; Z ^= t; }
        Q >>= 1;
    }

    /* Step 2: Gray encode */
    Y ^= X;
    Z ^= Y;

    /* Compute the correction t */
    unsigned int t = 0;
    Q = M;
    while (Q > 1) {
        if (Z & Q) t ^= (Q - 1);
        Q >>= 1;
    }
    X ^= t;
    Y ^= t;
    Z ^= t;

    /* Step 3: Transpose to Hilbert integer */
    unsigned long long h = 0;
    for (int i = p - 1; i >= 0; i--) {
        h = (h << 1) | ((X >> i) & 1);
        h = (h << 1) | ((Y >> i) & 1);
        h = (h << 1) | ((Z >> i) & 1);
    }

    hilbert_codes[index] = h;
}
"""


def _compute_block_stats(block_size_np):
    sx = block_size_np[:, 0]
    sy = block_size_np[:, 1]
    sz = block_size_np[:, 2]

    dims = np.stack([
        np.maximum(2.0 * sx, 1e-6),
        np.maximum(2.0 * sy, 1e-6),
        np.maximum(2.0 * sz, 1e-6)
    ], axis=-1)

    sorted_dims = np.sort(dims, axis=-1)
    d_min = sorted_dims[:, 0]
    d_max = sorted_dims[:, 2]

    aspect_ratio = d_max / np.maximum(d_min, 1e-10)

    volume = dims[:, 0] * dims[:, 1] * dims[:, 2]
    side_cube = np.cbrt(volume)

    surface_area = 2 * (dims[:, 0] * dims[:, 1] +
                        dims[:, 0] * dims[:, 2] +
                        dims[:, 1] * dims[:, 2])
    cube_sa = 6 * side_cube ** 2
    sa_ratio = surface_area / np.maximum(cube_sa, 1e-10)

    return {
        'aspect_ratio_mean': float(np.mean(aspect_ratio)),
        'aspect_ratio_median': float(np.median(aspect_ratio)),
        'aspect_ratio_p90': float(np.percentile(aspect_ratio, 90)),
        'aspect_ratio_p99': float(np.percentile(aspect_ratio, 99)),
        'volume_mean': float(np.mean(volume)),
        'volume_median': float(np.median(volume)),
        'side_mean': float(np.mean(side_cube)),
        'sa_ratio_mean': float(np.mean(sa_ratio)),
        'sa_ratio_median': float(np.median(sa_ratio)),
        'sa_ratio_p90': float(np.percentile(sa_ratio, 90)),
    }


def _compute_tile_occupancy(interacting_atoms_np, num_particles):
    valid = (interacting_atoms_np >= 0) & (interacting_atoms_np < num_particles)
    per_tile_count = valid.sum(axis=1).astype(np.float64)

    return {
        'occupancy_mean': float(np.mean(per_tile_count) / W),
        'occupancy_median': float(np.median(per_tile_count) / W),
        'occupancy_p10': float(np.percentile(per_tile_count, 10) / W),
        'occupancy_p90': float(np.percentile(per_tile_count, 90) / W),
        'mean_atoms_per_tile': float(np.mean(per_tile_count)),
        'tiles_total': int(len(per_tile_count)),
    }


def run_with_curve(curve_type, pos_x, pos_y, pos_z, pbc_matrix, pbc_inv,
                   topology, num_particles):
    tpb = 256
    box_x = abs(float(np.asarray(pbc_matrix).reshape(3, 3)[0, 0]))
    box_y = abs(float(np.asarray(pbc_matrix).reshape(3, 3)[1, 1]))
    box_z = abs(float(np.asarray(pbc_matrix).reshape(3, 3)[2, 2]))

    pbc_flat = np.ascontiguousarray(pbc_matrix.ravel(), dtype=np.float32)
    pbc_inv_flat = np.ascontiguousarray(pbc_inv.ravel(), dtype=np.float32)
    d_pbc = cp.asarray(pbc_flat)
    d_pbc_inv = cp.asarray(pbc_inv_flat)

    d_px = cp.asarray(pos_x, dtype=np.float32).copy()
    d_py = cp.asarray(pos_y, dtype=np.float32).copy()
    d_pz = cp.asarray(pos_z, dtype=np.float32).copy()

    n3 = (num_particles + tpb - 1) // tpb
    wrap_kern = cp.RawKernel(r"""
    extern "C" __global__
    void wrap(float* px, float* py, float* pz, const float* pbc, const float* pbc_inv, int N) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;
        float x = px[i], y = py[i], z = pz[i];
        float fx = x*pbc_inv[0]+y*pbc_inv[3]+z*pbc_inv[6];
        float fy = x*pbc_inv[1]+y*pbc_inv[4]+z*pbc_inv[7];
        float fz = x*pbc_inv[2]+y*pbc_inv[5]+z*pbc_inv[8];
        fx -= floorf(fx); fy -= floorf(fy); fz -= floorf(fz);
        px[i] = fx*pbc[0]+fy*pbc[3]+fz*pbc[6];
        py[i] = fx*pbc[1]+fy*pbc[4]+fz*pbc[7];
        pz[i] = fx*pbc[2]+fy*pbc[5]+fz*pbc[8];
    }
    """, 'wrap')
    wrap_kern((n3,), (tpb,), (d_px, d_py, d_pz, d_pbc, d_pbc_inv, np.int32(num_particles)))

    if curve_type == 'morton':
        from mdpy.core.tile_list import _MORTON_ENCODE_KERNEL
        kern = cp.RawKernel(_MORTON_ENCODE_KERNEL, 'morton_encode_kernel')
    else:
        kern = cp.RawKernel(_HILBERT_ENCODE_KERNEL, 'hilbert_encode_kernel')

    codes = cp.empty(num_particles, dtype=np.uint64)
    nm = (num_particles + tpb - 1) // tpb
    kern((nm,), (tpb,),
         (d_px, d_py, d_pz, d_pbc, d_pbc_inv, np.int32(num_particles),
          np.float32(box_x), np.float32(box_y), np.float32(box_z), codes))

    sorted_indices = cp.argsort(codes).astype(np.int32)

    num_blocks = (num_particles + W - 1) // W
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
    form_kern((nf,), (tpb,), (sorted_indices, np.int32(num_particles),
                              np.int32(num_blocks), block_atoms))

    block_center = cp.empty(num_blocks * 3, dtype=np.float32)
    block_size = cp.empty(num_blocks * 3, dtype=np.float32)

    bounds_kern = cp.RawKernel(r"""
    extern "C" __global__
    void bounds(const float* px, const float* py, const float* pz,
                const int* ba, int nb, float* bc, float* bs) {
        int bi = blockIdx.x * blockDim.x + threadIdx.x;
        if (bi >= nb) return;
        float mx=1e30f,my=1e30f,mz=1e30f,Mx=-1e30f,My=-1e30f,Mz=-1e30f;
        for(int s=0;s<32;s++){
            int a=ba[bi*32+s]; if(a<0)continue;
            float x=px[a],y=py[a],z=pz[a];
            mx=fminf(mx,x);Mx=fmaxf(Mx,x);
            my=fminf(my,y);My=fmaxf(My,y);
            mz=fminf(mz,z);Mz=fmaxf(Mz,z);
        }
        bc[bi*3]=.5f*(mx+Mx); bc[bi*3+1]=.5f*(my+My); bc[bi*3+2]=.5f*(mz+Mz);
        bs[bi*3]=.5f*(Mx-mx); bs[bi*3+1]=.5f*(My-my); bs[bi*3+2]=.5f*(Mz-mz);
    }
    """, 'bounds')
    nb_grid = (num_blocks + tpb - 1) // tpb
    bounds_kern((nb_grid,), (tpb,),
                (d_px, d_py, d_pz, block_atoms, np.int32(num_blocks),
                 block_center, block_size))

    block_size_np = cp.asnumpy(block_size).reshape(-1, 3)

    from mdpy.core.tile_list import TileList
    tl = TileList(CUTOFF, SKIN)
    tl._ensure_kernels()

    positions_soa = (d_px, d_py, d_pz)
    tl.num_particles = num_particles
    tl.num_blocks = num_blocks
    tl.d_block_atoms = block_atoms
    tl.d_block_center = block_center
    tl.d_block_size = block_size
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

    tl.d_atom_to_block = cp.full(num_particles, -1, dtype=np.int32)
    tl.d_atom_to_slot = cp.full(num_particles, -1, dtype=np.int32)
    tl._kernels['atom_map']((nb_grid,), (tpb,),
        (tl.d_block_atoms, np.int32(num_blocks), np.int32(W),
         tl.d_atom_to_block, tl.d_atom_to_slot))

    t0 = time.perf_counter()
    tl._find_interacting_blocks(positions_soa, pbc_matrix)
    cp.cuda.Stream.null.synchronize()
    t_find = time.perf_counter() - t0

    interacting_atoms_np = cp.asnumpy(tl.d_interacting_atoms).reshape(-1, W)

    block_stats = _compute_block_stats(block_size_np)
    tile_stats = _compute_tile_occupancy(interacting_atoms_np, num_particles)
    tile_stats['find_time_ms'] = t_find * 1000

    return block_stats, tile_stats


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
    print(f"Atoms: {N}")
    print(f"Box: {BOX_SIZE} A, Cutoff: {CUTOFF} A, Skin: {SKIN} A")
    print(f"Blocks (W={W}): {(N + W - 1) // W}")
    print()

    results = {}
    for curve in ['morton', 'hilbert']:
        print(f"=== {curve.upper()} ===")
        block_stats, tile_stats = run_with_curve(
            curve, pos_x, pos_y, pos_z, pbc_matrix, pbc_inv,
            topology, N)
        results[curve] = (block_stats, tile_stats)

        print("  Block AABB shape:")
        print(f"    Aspect ratio (max/min dim) mean:   {block_stats['aspect_ratio_mean']:.3f}")
        print(f"    Aspect ratio median:               {block_stats['aspect_ratio_median']:.3f}")
        print(f"    Aspect ratio p90:                  {block_stats['aspect_ratio_p90']:.3f}")
        print(f"    Aspect ratio p99:                  {block_stats['aspect_ratio_p99']:.3f}")
        print(f"    Mean block side (equiv. cube):      {block_stats['side_mean']:.2f} A")
        print(f"    Mean block volume:                  {block_stats['volume_mean']:.1f} A^3")
        print(f"    Median block volume:                {block_stats['volume_median']:.1f} A^3")
        print(f"    Surface area ratio (vs cube) mean:  {block_stats['sa_ratio_mean']:.4f}")
        print(f"    Surface area ratio median:          {block_stats['sa_ratio_median']:.4f}")
        print(f"    Surface area ratio p90:             {block_stats['sa_ratio_p90']:.4f}")
        print()
        print("  Tile occupancy:")
        print(f"    Total tiles:                        {tile_stats['tiles_total']}")
        print(f"    Mean occupancy:                     {tile_stats['occupancy_mean']:.3f} ({tile_stats['mean_atoms_per_tile']:.1f}/32)")
        print(f"    Median occupancy:                   {tile_stats['occupancy_median']:.3f}")
        print(f"    P10 occupancy:                      {tile_stats['occupancy_p10']:.3f}")
        print(f"    P90 occupancy:                      {tile_stats['occupancy_p90']:.3f}")
        print(f"    Tile-find time:                     {tile_stats['find_time_ms']:.1f} ms")
        print()

    m_bs, m_ts = results['morton']
    h_bs, h_ts = results['hilbert']

    print("=== DELTA (Hilbert - Morton) ===")
    print(f"  Aspect ratio mean:     {h_bs['aspect_ratio_mean']:.3f} vs {m_bs['aspect_ratio_mean']:.3f}  ({(h_bs['aspect_ratio_mean']/m_bs['aspect_ratio_mean']-1)*100:+.1f}%)")
    print(f"  Volume mean:           {h_bs['volume_mean']:.1f} vs {m_bs['volume_mean']:.1f}  ({(h_bs['volume_mean']/m_bs['volume_mean']-1)*100:+.1f}%)")
    print(f"  SA ratio mean:         {h_bs['sa_ratio_mean']:.4f} vs {m_bs['sa_ratio_mean']:.4f}  ({(h_bs['sa_ratio_mean']/m_bs['sa_ratio_mean']-1)*100:+.1f}%)")
    print(f"  Total tiles:           {h_ts['tiles_total']} vs {m_ts['tiles_total']}  ({(h_ts['tiles_total']/m_ts['tiles_total']-1)*100:+.1f}%)")
    print(f"  Mean occupancy:        {h_ts['occupancy_mean']:.3f} vs {m_ts['occupancy_mean']:.3f}  ({(h_ts['occupancy_mean']/m_ts['occupancy_mean']-1)*100:+.1f}%)")
    print(f"  Tile-find time:        {h_ts['find_time_ms']:.1f} ms vs {m_ts['find_time_ms']:.1f} ms")


if __name__ == '__main__':
    main()
