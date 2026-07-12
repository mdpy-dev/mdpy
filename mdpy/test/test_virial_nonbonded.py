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
