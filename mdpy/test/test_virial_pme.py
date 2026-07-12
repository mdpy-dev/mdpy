import numpy as np
import pytest
import cupy as cp
from mdpy.core.state import State
from mdpy.core.block_list import BlockList
from mdpy.core.topology import Topology
from mdpy.force.pme_reciprocal_force import PMEReciprocalForce


def _compute_pme(box_diag, charges, frac, cutoff=8.0):
    """Build a PME-only system at the given box and return (energy, virial_3x3)."""
    N = len(charges)
    topo = Topology(); topo.num_particles = N
    state = State(N)
    box = np.diag(box_diag).astype(np.float32)
    state.set_pbc(box)
    state.set_positions((frac * np.array(box_diag)).astype(np.float32))
    state.set_particle_charges(charges)
    state.set_particle_masses(np.ones(N, dtype=np.float32))
    state.set_particle_type_indices(np.zeros(N, dtype=np.int32))
    bl = BlockList(cutoff=cutoff, skin=1.0)
    bl.rebuild(topo, state, force=True)
    state.wrap_positions_with_prev_correction()
    bl.capture_snapshot(state)
    bl.build_block_pairs(topo, state)
    bl.refresh_sorted_posq(state)
    bl.refresh_sorted_type_indices(state)
    pme = PMEReciprocalForce(cutoff=cutoff)
    pme.initialize_grid(topo, None, box)
    state.zero_forces(); state.zero_energy(); state.zero_virial()
    pme.compute(state, bl, compute_energy=True, compute_virial=True)
    return float(state.d_energy[0]), state.d_virial.get().reshape(3, 3)


def test_pme_reciprocal_virial_matches_finite_difference():
    """PME reciprocal virial matches -dE/d(lnL) via finite differences.

    The virial W_{aa} = -dE/d(lnL_a) (strain derivative). mdpy stores
    half-virial (0.5*W), so d_virial[a,a] should equal 0.5*(-dE/d(lnL_a)).
    Uses fractional coordinates held fixed across box perturbations.
    """
    rng = np.random.default_rng(7)
    N = 8
    charges = rng.uniform(-0.5, 0.5, N).astype(np.float32)
    frac = rng.uniform(0.02, 0.98, (N, 3))

    L = 20.0
    eps = 1e-3
    E0, W0 = _compute_pme([L, L, L], charges, frac)
    Ex, _ = _compute_pme([L * (1 + eps), L, L], charges, frac)
    Ey, _ = _compute_pme([L, L * (1 + eps), L], charges, frac)
    Ez, _ = _compute_pme([L, L, L * (1 + eps)], charges, frac)

    fd_xx = -(Ex - E0) / eps
    fd_yy = -(Ey - E0) / eps
    fd_zz = -(Ez - E0) / eps

    # mdpy stores half-virial: d_virial = 0.5 * (-dE/dlnL)
    np.testing.assert_allclose(W0[0, 0], 0.5 * fd_xx, rtol=0.02,
                               err_msg="W_xx mismatch vs finite difference")
    np.testing.assert_allclose(W0[1, 1], 0.5 * fd_yy, rtol=0.02,
                               err_msg="W_yy mismatch vs finite difference")
    np.testing.assert_allclose(W0[2, 2], 0.5 * fd_zz, rtol=0.02,
                               err_msg="W_zz mismatch vs finite difference")
