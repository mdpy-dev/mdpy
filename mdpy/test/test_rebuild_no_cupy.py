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
    # one extra rebuild to warm the pool (first call allocates all buffers)
    bl.rebuild(s.topology, s.gpu)

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
        bl.rebuild(s.topology, s.gpu)
    finally:
        cp.empty = orig_empty
        cp.zeros = orig_zeros

    assert call_count[0] <= 1, (
        f"rebuild() made {call_count[0]} cp.empty/cp.zeros calls "
        "— allocations not pooled"
    )


def test_no_cp_arange_in_permute_path():
    """The permute path must not use cp.arange."""
    import inspect
    from mdpy import system
    src = inspect.getsource(system)
    count = sum(1 for line in src.split('\n') if 'cp.arange' in line and not line.strip().startswith('#'))
    assert count == 0, f"system.py still uses cp.arange ({count} occurrences)"
