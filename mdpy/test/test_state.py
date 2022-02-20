#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_state.py
created time : 2021/10/17
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest
import numpy as np
from .. import SPATIAL_DIM, env
from ..core import Particle, Topology, State
from ..error import *
from ..unit import *

class TestState:
    def setup(self):
        self.particles = []
        self.particles.append(Particle(particle_type='C', mass=12))
        self.particles.append(Particle(particle_type='N', mass=14))
        self.particles.append(Particle(particle_type='C', mass=12))
        self.particles.append(Particle(particle_type='H', mass=1))
        self.particles.append(Particle(particle_type='C', mass=12))
        self.particles.append(Particle(particle_type='H', mass=1))
        self.particles.append(Particle(particle_type='H', mass=1))
        self.particles.append(Particle(particle_type='N', mass=14))
        self.particles.append(Particle(particle_type='C', mass=12))
        _ = [self.particles.extend(self.particles) for i in range(7)]
        self.num_particles = len(self.particles)
        self.topology = Topology()
        self.topology.add_particles(self.particles)
        self.topology.join()
        self.state = State(self.topology)

    def teardown(self):
        self.particles = None
        self.state = None
        self.topology = None

    def test_attributes(self):
        assert self.state.positions.shape[0] == self.num_particles
        assert self.state.positions.shape[1] == SPATIAL_DIM
        assert self.state.velocities.shape[0] == self.num_particles
        assert self.state.velocities.shape[1] == SPATIAL_DIM

    def test_exceptions(self):
        with pytest.raises(ArrayDimError):
            self.state.set_positions(np.ones([5, 3]))

        with pytest.raises(ArrayDimError):
            self.state.set_velocities(np.ones([self.num_particles, 4]))

        with pytest.raises(PBCPoorDefinedError):
            self.state.pbc_matrix

        with pytest.raises(PBCPoorDefinedError):
            self.state.pbc_inv

        with pytest.raises(TypeError):
            self.state.set_positions([1, 2, 3])

    def test_set_positions(self):
        self.state.set_pbc_matrix(np.eye(3) * 100)
        self.state.cell_list.set_cutoff_radius(12)
        self.state.set_positions(np.ones([self.num_particles, SPATIAL_DIM]) * 10)
        assert self.state.positions[0, 2] == 10

    def test_set_velocities(self):
        self.state.set_velocities(np.ones([self.num_particles, SPATIAL_DIM]) * 10)
        assert self.state.velocities[0, 2] == 10

    def test_set_velocities_to_temperature(self):
        self.state.set_velocities_to_temperature(300)
        kinetic_energy = Quantity(0, default_energy_unit)
        for particle in range(self.num_particles):
            kinetic_energy += Quantity(0.5) * (
                Quantity(self.state._masses[particle], default_mass_unit) * 
                (Quantity(self.state.velocities[particle, :], default_velocity_unit)**2).sum()
            )
        temperature = kinetic_energy * Quantity(2 / 3 / self.num_particles) / KB
        assert temperature < Quantity(315, default_temperature_unit)
        assert temperature > Quantity(285, default_temperature_unit)

    def test_pbc(self):
        with pytest.raises(PBCPoorDefinedError):
            self.state.set_pbc_matrix(np.ones([3, 3]))

        with pytest.raises(ArrayDimError):
            self.state.set_pbc_matrix(np.ones([4, 3]))

        self.state.set_pbc_matrix(np.diag(np.ones(3)*10))
        assert self.state.pbc_inv[1, 1] == env.NUMPY_FLOAT(0.1)

        self.state.set_pbc_matrix(Quantity(np.diag(np.ones(3)), nanometer))
        assert self.state.pbc_inv[2, 2] == env.NUMPY_FLOAT(0.1)