import numpy as np
import pytest
import cupy as cp
from mdpy import precision
from mdpy.core.state import State
from mdpy.force.bonded_force import BondedForce
from mdpy.force.bonded_transpiler import bonded_expression


@bonded_expression(body=2)
def harmonic_bond(p1, p2, k=0.0, r0=0.0):
    r = distance(p1, p2)
    return k * (r - r0) ** 2


@bonded_expression(body=3)
def harmonic_angle(p1, p2, p3, k=0.0, theta0=0.0):
    theta = angle(p1, p2, p3)
    return k * (theta - theta0) ** 2


@bonded_expression(body=4)
def periodic_dihedral(p1, p2, p3, p4, k=0.0, n=0.0, delta=0.0):
    phi = dihedral(p1, p2, p3, p4)
    return k * (1.0 + cos(n * phi - delta))


def _make_state(positions, box=100.0):
    N = len(positions)
    state = State(N)
    state.set_pbc(np.diag([box, box, box]).astype(np.float32))
    state.set_positions(positions.astype(np.float32))
    state.set_particle_charges(np.zeros(N, dtype=np.float32))
    state.set_particle_masses(np.ones(N, dtype=np.float32))
    state.set_particle_type_indices(np.zeros(N, dtype=np.int32))
    return state


def _get_forces(state):
    return np.stack([
        state.d_forces_x.get(), state.d_forces_y.get(), state.d_forces_z.get()
    ], axis=1)


def _half_virial_from_forces(positions, forces, ref_idx):
    """0.5 * Σ_i (r_i - r_ref) ⊗ F_i — the mdpy half-virial convention."""
    ref = positions[ref_idx]
    W = np.zeros((3, 3), dtype=np.float64)
    for i in range(len(positions)):
        disp = positions[i] - ref
        W += 0.5 * np.outer(disp, forces[i])
    return W


def test_bond_virial_matches_force_cross_position():
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 1.0, 0.0]])
    state = _make_state(positions)
    force = BondedForce(harmonic_bond)
    force.add([0, 1], k=100.0, r0=1.5)
    state.zero_virial()
    force.compute(state, compute_energy=True, compute_virial=True)
    mdpy_virial = state.d_virial.get().reshape(3, 3)
    ref = _half_virial_from_forces(positions, _get_forces(state), ref_idx=0)
    np.testing.assert_allclose(mdpy_virial, ref, atol=1e-3,
                               err_msg="Bond virial mismatch")


def test_angle_virial_matches_force_cross_position():
    positions = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    state = _make_state(positions)
    force = BondedForce(harmonic_angle)
    force.add([0, 1, 2], k=50.0, theta0=1.5707963)
    state.zero_virial()
    force.compute(state, compute_energy=True, compute_virial=True)
    mdpy_virial = state.d_virial.get().reshape(3, 3)
    ref = _half_virial_from_forces(positions, _get_forces(state), ref_idx=1)
    np.testing.assert_allclose(mdpy_virial, ref, atol=1e-3,
                               err_msg="Angle virial mismatch")


def test_dihedral_virial_matches_force_cross_position():
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ])
    state = _make_state(positions)
    force = BondedForce(periodic_dihedral)
    force.add([0, 1, 2, 3], k=2.0, n=3.0, delta=0.0)
    state.zero_virial()
    force.compute(state, compute_energy=True, compute_virial=True)
    mdpy_virial = state.d_virial.get().reshape(3, 3)
    ref = _half_virial_from_forces(positions, _get_forces(state), ref_idx=1)
    np.testing.assert_allclose(mdpy_virial, ref, atol=1e-2,
                               err_msg="Dihedral virial mismatch")
