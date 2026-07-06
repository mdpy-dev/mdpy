import os, numpy as np, pytest


def _build_ion():
    from mdpy.io.psf_parser import PSFParser
    from mdpy.io.pdb_parser import PDBParser
    from mdpy.io.charmm_toppar_parser import CharmmTopparParser, create_parameter_table
    from mdpy.force.factories.charmm import create_charmm_forces
    from mdpy.system import System
    from mdpy.utils import generate_velocity_from_temperature

    DATA = os.path.join(os.path.dirname(__file__), "..", "..", "benchmark", "data")
    psf = PSFParser(os.path.join(DATA, "ion.psf"))
    pdb = PDBParser(os.path.join(DATA, "ion.pdb"))
    tp = CharmmTopparParser(
        os.path.join(DATA, "par_sin.prm"), os.path.join(DATA, "par_water.prm")
    )
    topo = psf.topology
    pt = create_parameter_table(topo, tp)
    pbc = np.diag([75.450, 77.623, 69.668])
    forces = create_charmm_forces(topo, pt, pbc, cutoff=12.0)
    s = System(topo)
    s.upload_pbc(pbc)
    s.add_force_term(forces["bonded"])
    s.add_force_term(forces["nonbonded"])
    s.add_force_term(forces["pme"], stream="pme")
    s.upload_positions(pdb.positions)
    s.upload_velocities(
        generate_velocity_from_temperature(300.0, topo.masses, seed=42)
    )
    return s


def test_rebuild_no_cudamalloc_on_second_rebuild():
    """BlockList.rebuild() must not call cp.empty/cp.zeros once the persistent
    buffer pool is warm — every allocation is served from the pool.

    Counts cp.empty/cp.zeros calls via monkeypatch. On the unmodified code the
    warm rebuild makes 32 fresh allocations; with the pool it makes at most 1
    (the ``sorted_to_pdb`` buffer, which cannot be pooled because line 808
    captures ``prev_sorted_to_pdb`` before ``counting_scatter`` overwrites the
    same buffer — pooling would alias and corrupt the captured reference).
    """
    import cupy as cp

    s = _build_ion()
    s.update_neighbor_list(force_rebuild=True)  # init block list + populate pool

    bl = s._block_list
    positions_soa = (
        s.gpu.d_positions_x,
        s.gpu.d_positions_y,
        s.gpu.d_positions_z,
    )
    # one extra rebuild to warm the pool (first call allocates all buffers)
    bl.rebuild(positions_soa, s.topology, s.gpu)

    call_count = [0]
    orig_empty = cp.empty
    orig_zeros = cp.zeros

    def counting_empty(*args, **kwargs):
        call_count[0] += 1
        return orig_empty(*args, **kwargs)

    def counting_zeros(*args, **kwargs):
        call_count[0] += 1
        return orig_zeros(*args, **kwargs)

    cp.empty = counting_empty
    cp.zeros = counting_zeros
    try:
        bl.rebuild(positions_soa, s.topology, s.gpu)
    finally:
        cp.empty = orig_empty
        cp.zeros = orig_zeros

    assert call_count[0] <= 1, (
        f"rebuild() made {call_count[0]} cp.empty/cp.zeros calls "
        "— allocations not pooled"
    )


def test_permute_state_arrays_no_alloc_on_second_rebuild():
    """Second rebuild must not allocate new state-permutation buffers."""
    s = _build_ion()
    s.update_neighbor_list(force_rebuild=True)  # warm: allocates pool_A + pool_B
    gpu = s.gpu
    assert gpu._perm_pool_A is not None
    assert gpu._perm_pool_B is not None
    # Snapshot the pool buffer data pointers
    ptrs_A_before = [arr.data.ptr for arr in gpu._perm_pool_A]
    ptrs_B_before = [arr.data.ptr for arr in gpu._perm_pool_B]
    s.update_neighbor_list(force_rebuild=True)  # should reuse same buffers
    ptrs_A_after = [arr.data.ptr for arr in gpu._perm_pool_A]
    ptrs_B_after = [arr.data.ptr for arr in gpu._perm_pool_B]
    assert ptrs_A_before == ptrs_A_after, "pool_A was reallocated"
    assert ptrs_B_before == ptrs_B_after, "pool_B was reallocated"


def test_invert_permutation_kernel():
    import cupy as cp
    from mdpy.system import _get_invert_permutation_kernel
    kernel = _get_invert_permutation_kernel()
    perm = cp.array([3, 1, 0, 2], dtype=cp.int32)
    out = cp.empty(4, dtype=cp.int32)
    kernel((1,), (4,), (out, perm, np.int32(4)))
    cp.cuda.Device().synchronize()
    # out[perm[i]] = i => out[3]=0, out[1]=1, out[0]=2, out[2]=3
    assert (out.get() == [2, 1, 3, 0]).all()


def test_no_cp_arange_in_permute_path():
    """The permute path must not use cp.arange."""
    import inspect
    from mdpy import system
    src = inspect.getsource(system)
    count = sum(1 for line in src.split('\n') if 'cp.arange' in line and not line.strip().startswith('#'))
    assert count == 0, f"system.py still uses cp.arange ({count} occurrences)"


def test_excl_buffers_no_aliasing_across_rebuilds():
    """permute_exclusion_pairs_gpu double-buffers: a call's output buffers
    (returned as the next call's input) must never be the same memory as the
    previous call's output. Aliasing would corrupt the permute_pairs kernel,
    which reads d_cached_* and writes d_new_* simultaneously."""
    s = _build_ion()
    # Prime: first rebuild goes through the else-branch (build_exclusion_map_gpu,
    # no pool). Only subsequent rebuilds use permute_exclusion_pairs_gpu.
    s.update_neighbor_list(force_rebuild=True)
    # Now each rebuild calls permute_exclusion_pairs_gpu with double-buffering.
    s.update_neighbor_list(force_rebuild=True)  # pool A
    ptr_A = s._d_cached_unique_i.data.ptr
    s.update_neighbor_list(force_rebuild=True)  # pool B
    ptr_B = s._d_cached_unique_i.data.ptr
    assert ptr_A != ptr_B, "Double-buffering failed: same buffer reused (aliasing risk)"
    s.update_neighbor_list(force_rebuild=True)  # pool A again
    ptr_A2 = s._d_cached_unique_i.data.ptr
    assert ptr_A2 == ptr_A, "Double-buffering flip pattern wrong (should cycle A->B->A)"
