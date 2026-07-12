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
