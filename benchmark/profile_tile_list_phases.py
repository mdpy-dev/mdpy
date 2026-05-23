"""mdpy tile list rebuild per-phase timing.

Measures each phase of TileList.rebuild() separately.
"""
import os
import time
import cupy as cp
import numpy as np

from mdpy.forcefield.charmm_forcefield import CharmmForcefield
from mdpy.core.tile_list import TileList
from mdpy import env

from benchmark._data_path import DATA_DIR
BOX_SIZE = 108.0
CUTOFF = 12.0


def main():
    ff = CharmmForcefield(
        os.path.join(DATA_DIR, '1M9Z.psf'),
        os.path.join(DATA_DIR, '1M9Z_minimized.pdb'),
        [os.path.join(DATA_DIR, 'par_all36_prot.prm'),
         os.path.join(DATA_DIR, 'toppar_water_ions.str')]
    )
    topology = ff.create_topology()
    N = topology.num_particles

    pbc_matrix = np.eye(3, dtype=np.float64) * BOX_SIZE
    pbc_inv = np.linalg.inv(pbc_matrix)
    pbc_f = np.ascontiguousarray(pbc_matrix, dtype=env.NUMPY_FLOAT)
    pbc_inv_f = np.ascontiguousarray(pbc_inv, dtype=env.NUMPY_FLOAT)

    raw = ff._pdb.positions.astype(np.float64)
    frac = raw @ pbc_inv
    frac -= np.floor(frac)
    wrapped = frac @ pbc_matrix
    positions = cp.asarray(
        np.ascontiguousarray(wrapped, dtype=env.NUMPY_FLOAT).ravel()
    )

    tl = TileList(CUTOFF, skin=2.0)

    tl.rebuild(positions, topology, pbc_f, pbc_inv_f)
    cp.cuda.Stream.null.synchronize()

    num_trials = 20
    phases = {
        'rebuild_core': [],
        'find_interacting': [],
        'build_masks': [],
        'total': [],
    }

    for _ in range(num_trials):
        tl._invalidate_caches()
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()

        wrapped = tl._rebuild_core(positions, topology, pbc_f, pbc_inv_f)
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()

        tl._find_interacting_blocks(wrapped, pbc_f)
        cp.cuda.Stream.null.synchronize()
        t2 = time.perf_counter()

        tl._build_masks_gpu(topology)
        cp.cuda.Stream.null.synchronize()
        t3 = time.perf_counter()

        phases['rebuild_core'].append(t1 - t0)
        phases['find_interacting'].append(t2 - t1)
        phases['build_masks'].append(t3 - t2)
        phases['total'].append(t3 - t0)

        tl._is_initialized = True

    print(f"Atoms: {N}  Blocks: {tl.num_blocks}  Tiles: {tl.num_tiles}")
    print(f"{'Phase':<25s} {'Mean (ms)':>10s} {'Std (ms)':>10s} {'%':>6s}")
    print("-" * 55)
    total_mean = np.mean(phases['total']) * 1000
    for name in ['rebuild_core', 'find_interacting', 'build_masks', 'total']:
        vals = np.array(phases[name]) * 1000
        pct = vals.mean() / total_mean * 100 if name != 'total' else 100.0
        print(f"  {name:<23s} {vals.mean():10.3f} {vals.std():10.3f} {pct:5.1f}%")


if __name__ == '__main__':
    main()
