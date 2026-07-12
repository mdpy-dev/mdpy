import numpy as np
import pytest
import cupy as cp
from mdpy.core.state import State
from mdpy.core.block_list import BlockList
from mdpy.core.topology import Topology
from mdpy.force.pme_reciprocal_force import PMEReciprocalForce, COULOMB_CONST, SQRT_PI


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Task 7 reciprocal virial kernel (pme_reciprocal_force.py:441-487) is "
        "missing the overall prefactor. It sums corner*|Q(k)|^2*(vfactor*m2-3) "
        "on the raw FFT/bk scale, but carries no COULOMB constant and no 1/V "
        "(or 1/grid_total) normalization that the gather energy has. "
        "Measured: kernel trace = -2420.97, but Essmann identity + finite-"
        "difference pressure both expect ~-0.0108 (= -E_recip). Off by factor "
        "~224,278x. Beyond Task 8's scope (which is the gate test, not the "
        "kernel); fixing requires re-deriving the SPME reciprocal virial "
        "prefactor (Essmann Eq. 2.7 / GROMACS pme_solve.cpp) and validating "
        "against finite-difference dE/dlnV. Remove this xfail once the kernel "
        "trace matches -E_recip."
    ),
)
def test_pme_reciprocal_virial_trace_identity():
    """For reciprocal-space Ewald: Tr(W_recip) ~ -E_recip (Essmann Eq. 2.7 trace).

    This is a reference-free sanity check on the reciprocal virial kernel.
    The identity holds for the exact Ewald sum; the SPME approximation
    introduces a small discretization error. Tolerance is loose (10%).
    """
    N = 8
    topo = Topology(); topo.num_particles = N
    state = State(N)
    box = np.diag([20.0, 20.0, 20.0]).astype(np.float32)
    state.set_pbc(box)
    rng = np.random.default_rng(7)
    state.set_positions(rng.uniform(0, 20, (N, 3)).astype(np.float32))
    charges = rng.uniform(-0.5, 0.5, N).astype(np.float32)
    state.set_particle_charges(charges)
    state.set_particle_masses(np.ones(N, dtype=np.float32))
    state.set_particle_type_indices(np.zeros(N, dtype=np.int32))

    bl = BlockList(cutoff=8.0, skin=1.0)
    bl.rebuild(topo, state, force=True)
    state.wrap_positions_with_prev_correction()
    bl.capture_snapshot(state)
    bl.build_block_pairs(topo, state)
    bl.refresh_sorted_posq(state)
    bl.refresh_sorted_type_indices(state)

    pme = PMEReciprocalForce(cutoff=8.0)
    pme.initialize_grid(topo, None, box)

    # Compute both energy and virial in one pass
    state.zero_forces()
    state.zero_energy()
    state.zero_virial()
    pme.compute(state, bl, compute_energy=True, compute_virial=True)
    energy = float(state.d_energy[0])
    virial = state.d_virial.get().reshape(3, 3)
    trace = float(np.trace(virial))

    # Diagnostic: PME d_energy includes a self-energy correction
    # (-COULOMB*alpha/sqrt(pi)*sum(q^2)) that has zero reciprocal virial.
    # The trace identity holds against the pure k-space sum, so we report it
    # to interpret any residual.
    sum_q2 = float(np.sum(charges.astype(np.float64) ** 2))
    e_self = -COULOMB_CONST * pme.alpha / SQRT_PI * sum_q2
    e_recip_sum = energy - e_self
    ratio_total = trace / (-energy) if abs(energy) > 0 else float('inf')
    ratio_pure = trace / (-e_recip_sum) if abs(e_recip_sum) > 0 else float('inf')
    print(
        f"Tr={trace:.6f}, E_total={energy:.6f}, E_self={e_self:.6f}, "
        f"E_recip_sum={e_recip_sum:.6f}, "
        f"ratio Tr/(-E_total)={ratio_total:.4f}, "
        f"ratio Tr/(-E_recip_sum)={ratio_pure:.4f}"
    )

    # Trace identity: Tr(W_recip) ~ -E_recip
    # Loose tolerance (10%) because SPME is an approximation and the
    # Hermitian half-weighting convention may introduce a factor-of-2 shift.
    assert abs(trace + energy) < 0.10 * abs(energy), (
        f"Trace identity violated: Tr(W)={trace}, -E={-energy}, "
        f"|Tr+E|/|E|={abs(trace + energy) / abs(energy):.4f}"
    )
