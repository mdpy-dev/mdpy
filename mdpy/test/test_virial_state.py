import numpy as np
import cupy as cp
import pytest
from mdpy.core.state import State


def test_d_virial_initialized_to_nine_zeros():
    state = State(num_particles=4)
    virial = state.d_virial.get()
    assert virial.shape == (9,)
    assert np.all(virial == 0.0)


def test_zero_virial_resets_to_zero():
    state = State(num_particles=4)
    state.d_virial[:] = 1.0
    state.zero_virial()
    assert np.all(state.d_virial.get() == 0.0)


def test_allocate_virial_accumulator_shape():
    state = State(num_particles=4)
    state.allocate_virial_accumulator(3)
    acc = state.d_virial_accumulator.get()
    assert acc.shape == (3, 9)


def test_set_virial_slot_copies_d_virial():
    state = State(num_particles=4)
    state.allocate_virial_accumulator(2)
    state.d_virial[:] = cp.asarray(np.arange(9, dtype=np.float32))
    state.set_virial_slot(1)
    np.testing.assert_array_equal(
        state.d_virial_accumulator.get()[1],
        np.arange(9, dtype=np.float32),
    )


def test_force_term_compute_accepts_compute_virial_kwarg():
    from mdpy.force.force_term import ForceTerm
    class StubTerm(ForceTerm):
        name = "stub"
        def compute(self, state, block_list=None, compute_energy=True, compute_virial=False):
            assert compute_virial is True
    term = StubTerm()
    term.compute(state=None, compute_virial=True)


def test_system_dump_virial_returns_empty_dict_when_no_terms():
    import numpy as np
    from mdpy.core.topology import Topology
    from mdpy.system import System
    topo = Topology()
    topo.num_particles = 2
    system = System(topo)
    result = system.dump_virial()
    assert result == {}


def test_compute_kinetic_energy_ideal_gas():
    """K = 0.5 * sum(m * v^2) for a simple system."""
    import cupy as cp
    import numpy as np
    from mdpy.core.state import State

    state = State(num_particles=3)
    state.set_particle_masses(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    state.set_velocities(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
    ], dtype=np.float32))

    ke = state.compute_kinetic_energy()
    # K = 0.5*1*1 + 0.5*2*4 + 0.5*3*9 = 0.5 + 4.0 + 13.5 = 18.0
    assert abs(ke - 18.0) < 1e-3, f"Expected K=18.0, got {ke}"


def test_dump_pressure_ideal_gas():
    """For an ideal gas (no forces): P = NkT/V."""
    import numpy as np
    from mdpy.core.state import State
    from mdpy.core.topology import Topology
    from mdpy.system import System
    from mdpy.utils import generate_velocity_from_temperature
    from mdpy.unit import KB, NA, default_energy_unit, kelvin

    BOLTZMANN = float(KB.convert_to(default_energy_unit / kelvin).value)
    BAR_TO_INTERNAL = float(NA.value) * 1e-32

    N = 100
    T = 300.0
    box_len = 50.0
    volume = box_len ** 3

    topo = Topology(); topo.num_particles = N
    state = State(N)
    state.set_pbc(np.diag([box_len]*3).astype(np.float32))
    masses = np.ones(N, dtype=np.float32)
    state.set_particle_masses(masses)
    state.set_velocities(
        generate_velocity_from_temperature(T, masses, seed=42)
    )
    state.set_particle_charges(np.zeros(N, dtype=np.float32))
    state.set_particle_type_indices(np.zeros(N, dtype=np.int32))

    system = System(topo, state=state)

    P_bar = system.dump_pressure()
    P_expected_bar = (N * BOLTZMANN * T / volume) / BAR_TO_INTERNAL

    assert abs(P_bar - P_expected_bar) < 0.1 * abs(P_expected_bar), (
        f"Pressure {P_bar:.2f} bar vs expected {P_expected_bar:.2f} bar "
        f"(ideal gas NkT/V)"
    )
