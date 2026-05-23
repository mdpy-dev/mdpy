import numpy as np
from mdpy.core.particle_table import ParticleTable


def test_create():
    table = ParticleTable(100)
    assert table.num_particles == 100
    assert table.positions.shape == (100, 3)
    assert table.velocities.shape == (100, 3)
    assert table.forces.shape == (100, 3)
    assert table.masses.shape == (100,)
    assert table.charges.shape == (100,)
    assert table.particle_types.shape == (100,)
    assert table.molecule_ids.shape == (100,)


def test_dtype():
    from mdpy import env
    table = ParticleTable(10)
    assert table.positions.dtype == env.NUMPY_FLOAT
    assert table.masses.dtype == env.NUMPY_FLOAT
    assert table.particle_types.dtype == env.NUMPY_INT


def test_zero_forces():
    table = ParticleTable(10)
    table.forces[:] = 42.0
    table.zero_forces()
    assert np.all(table.forces == 0)


def test_positions_writable():
    table = ParticleTable(5)
    table.positions[:] = np.arange(15).reshape(5, 3)
    assert table.positions[2, 1] == 7.0


def test_repr():
    table = ParticleTable(50)
    assert '50' in repr(table)
