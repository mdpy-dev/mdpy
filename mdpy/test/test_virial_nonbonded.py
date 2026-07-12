import numpy as np
import pytest
import cupy as cp
from mdpy import precision
from mdpy.force.nonbonded_force import NonbondedForce
from mdpy.force.nonbonded_transpiler import nonbonded_expression


@nonbonded_expression
def harmonic_pair(pos1, pos2, k=0.0, r0=0.0):
    r = distance(pos1, pos2)
    dr = r - r0
    return 0.5 * k * dr * dr


def test_nonbonded_fshift_buffer_allocated():
    """NonbondedForce should allocate per-block-pair fshift buffers."""
    force = NonbondedForce(harmonic_pair, cutoff=5.0)
    force.set_pair_parameter('k', np.zeros((1, 1), dtype=np.float32))
    force.set_pair_parameter('r0', np.zeros((1, 1), dtype=np.float32))

    # Access internal buffer attributes after lazy compile
    class FakeState:
        num_particles = 2
        d_particle_type_indices = cp.zeros(2, dtype=np.int32)
    force._lazy_compile(FakeState())
    assert hasattr(force, '_d_fshift_bp_x')


def test_nonbonded_virial_matches_x_cross_f_reference():
    """mdpy nonbonded virial (GROMACS two-piece) vs brute-force 0.5*Σx⊗F.

    Uses a large box (100Å) with atoms clustered near origin so ALL pairs
    are central-image (shift=0). Then piece 2 (shift⊗fshift) vanishes and
    mdpy virial = 0.5*Σx⊗F, which we compute independently from mdpy's
    own PDB-order forces.
    """
    from mdpy.core.state import State
    from mdpy.core.block_list import BlockList
    from mdpy.core.topology import Topology
    from mdpy.force.markers import param

    @nonbonded_expression
    def lj_pair(pos1, pos2, epsilon=param, sigma=param):
        r = distance(pos1, pos2)
        return 4.0 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

    N = 6
    topo = Topology(); topo.num_particles = N
    state = State(N)
    box = np.diag([100.0, 100.0, 100.0]).astype(np.float32)
    state.set_pbc(box)
    rng = np.random.default_rng(42)
    positions = rng.uniform(0, 8, (N, 3)).astype(np.float32)
    state.set_positions(positions)
    state.set_particle_charges(np.zeros(N, dtype=np.float32))
    state.set_particle_masses(np.ones(N, dtype=np.float32))
    state.set_particle_type_indices(np.zeros(N, dtype=np.int32))

    bl = BlockList(cutoff=5.0, skin=1.0)
    bl.rebuild(topo, state, force=True)
    state.wrap_positions_with_prev_correction()
    bl.capture_snapshot(state)
    bl.build_block_pairs(topo, state)
    bl.refresh_sorted_posq(state)
    bl.refresh_sorted_type_indices(state)

    force = NonbondedForce(lj_pair, cutoff=5.0)
    force.set_pair_parameter('epsilon', np.array([0.2], dtype=np.float32))
    force.set_pair_parameter('sigma', np.array([1.5], dtype=np.float32))

    state.zero_forces()
    state.zero_virial()
    force.compute(state, bl, compute_energy=False, compute_virial=True)
    mdpy_virial = state.d_virial.get().reshape(3, 3)

    wrapped = np.stack([
        state.d_positions_x.get(),
        state.d_positions_y.get(),
        state.d_positions_z.get(),
    ], axis=1)
    forces = np.stack([
        state.d_forces_x.get(),
        state.d_forces_y.get(),
        state.d_forces_z.get(),
    ], axis=1)

    ref_virial = np.zeros((3, 3), dtype=np.float64)
    for i in range(N):
        for a in range(3):
            for b in range(3):
                ref_virial[a, b] += 0.5 * wrapped[i, a] * forces[i, b]

    np.testing.assert_allclose(
        mdpy_virial, ref_virial, atol=1e-3,
        err_msg="Nonbonded virial mismatch vs brute-force 0.5*Σx⊗F"
    )
