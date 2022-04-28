#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_state.py
created time : 2021/10/17
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import pytest
import numpy as np
from mdpy import SPATIAL_DIM
from mdpy.core import Particle, Topology, State
from mdpy.error import *
from mdpy.unit import *

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
        self.state = State(self.topology, np.diag([300, 300, 300]))

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
            self.state.set_pbc_matrix(np.diag([0, 0, 0]))

    def test_set_positions(self):
        self.state.set_pbc_matrix(np.eye(3) * 100)
        self.state.neighbor_list.set_cutoff_radius(12)
        positions = np.random.randn(self.num_particles, SPATIAL_DIM) * 20
        self.state.set_positions(positions)
        assert self.state.positions.get()[0, 2] == pytest.approx(positions[0, 2])

    def test_set_velocities(self):
        self.state.set_velocities(np.ones([self.num_particles, SPATIAL_DIM]) * 10)
        assert self.state.velocities.get()[0, 2] == 10

    def test_set_velocities_to_temperature(self):
        self.state.set_velocities_to_temperature(300)
        kinetic_energy = Quantity(0, default_energy_unit)
        velocities = self.state.velocities.get()
        for particle in range(self.num_particles):
            kinetic_energy += Quantity(0.5) * (
                Quantity(self.state._topology.masses[particle], default_mass_unit) *
                (Quantity(velocities[particle, :], default_velocity_unit)**2).sum()
            )
        temperature = kinetic_energy * Quantity(2 / 3 / self.num_particles) / KB
        assert temperature < Quantity(315, default_temperature_unit)
        assert temperature > Quantity(285, default_temperature_unit)

    def test_pbc(self):
        with pytest.raises(PBCPoorDefinedError):
            self.state.set_pbc_matrix(np.ones([3, 3]))

        with pytest.raises(ArrayDimError):
            self.state.set_pbc_matrix(np.ones([4, 3]))
