#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_trajectory.py
created time : 2022/02/19
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest
import numpy as np
from .. import SPATIAL_DIM, env
from ..core import Particle, Topology, Trajectory
from ..error import *
from ..unit import *

class TestTrajectory:
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
        self.trajectory = Trajectory(self.topology)

    def teardown(self):
        self.particles = None
        self.state = None
        self.topology = None
        self.trajectory = None

    def test_attributes(self):
        assert self.trajectory.num_frames == 0

    def test_exceptions(self):
        with pytest.raises(ArrayDimError):
            self.trajectory._check_array(np.ones([self.topology.num_particles-1, SPATIAL_DIM]))

        with pytest.raises(ArrayDimError):
            self.trajectory._check_array(np.ones([self.topology.num_particles, SPATIAL_DIM+1]))

        with pytest.raises(ArrayDimError):
            self.trajectory._check_array(np.ones([1, self.topology.num_particles-1, SPATIAL_DIM]))

        with pytest.raises(ArrayDimError):
            self.trajectory._check_array(np.ones([1, self.topology.num_particles, SPATIAL_DIM+1]))

        with pytest.raises(ArrayDimError):
            self.trajectory._check_array(np.ones([1, 1, self.topology.num_particles, SPATIAL_DIM+1]))

        with pytest.raises(TrajectoryPoorDefinedError):
            self.trajectory.velocities
        
        with pytest.raises(TrajectoryPoorDefinedError):
            self.trajectory.time_series

        with pytest.raises(PBCPoorDefinedError):
            self.trajectory.pbc_matrix

        with pytest.raises(PBCPoorDefinedError):
            self.trajectory.pbc_inv

        with pytest.raises(TypeError):
            self.trajectory.append(positions=[1])

    def test_check_array(self):
        assert self.trajectory._check_array(np.ones([self.topology.num_particles, SPATIAL_DIM])).shape[0] == 1
        assert self.trajectory._check_array(np.ones([2, self.topology.num_particles, SPATIAL_DIM])).shape[0] == 2

    def test_set_time_step(self):
        self.trajectory.set_time_step(Quantity(1, femtosecond) * Quantity(1000))
        assert self.trajectory.time_step == 1000

    def test_set_pbc_matrix(self):
        with pytest.raises(PBCPoorDefinedError):
            self.trajectory.set_pbc_matrix(np.ones([3, 3]))

        with pytest.raises(ArrayDimError):
            self.trajectory.set_pbc_matrix(np.ones([4, 3]))

        self.trajectory.set_pbc_matrix(np.diag(np.ones(3)*10))
        assert self.trajectory.pbc_inv[1, 1] == env.NUMPY_FLOAT(0.1)

        self.trajectory.set_pbc_matrix(Quantity(np.diag(np.ones(3)), nanometer))
        assert self.trajectory.pbc_inv[2, 2] == env.NUMPY_FLOAT(0.1)

    def test_append(self):
        with pytest.raises(TrajectoryPoorDefinedError):
            self.trajectory.append(velocities=1)

        with pytest.raises(TrajectoryPoorDefinedError):
            self.trajectory.append(positions=None)

        with pytest.raises(ArrayDimError):
            self.trajectory.append(positions=np.ones([self.topology.num_particles, SPATIAL_DIM+1]))

        with pytest.raises(ArrayDimError):
            self.trajectory.append(positions=np.ones([3, self.topology.num_particles-1, SPATIAL_DIM]))

        self.trajectory.append(positions=np.ones([3, self.topology.num_particles, SPATIAL_DIM]))
        assert self.trajectory.num_frames == 3
        self.trajectory.append(positions=np.ones([3, self.topology.num_particles, SPATIAL_DIM]))
        assert self.trajectory.num_frames == 6
        assert self.trajectory.positions.shape[0] == 6 

        self.trajectory = Trajectory(self.topology, contain_velocities=True)
        with pytest.raises(ArrayDimError):
            self.trajectory.append(
                positions=np.ones([3, self.topology.num_particles, SPATIAL_DIM]),
                velocities=np.ones([4, self.topology.num_particles, SPATIAL_DIM])
            )
        
        self.trajectory.append(
            positions=np.ones([4, self.topology.num_particles, SPATIAL_DIM]),
            velocities=np.ones([4, self.topology.num_particles, SPATIAL_DIM])
        )
        assert self.trajectory.num_frames == 4
        self.trajectory.append(
            positions=np.ones([4, self.topology.num_particles, SPATIAL_DIM]),
            velocities=np.ones([4, self.topology.num_particles, SPATIAL_DIM])
        )
        assert self.trajectory.num_frames == 8
        assert self.trajectory.positions.shape[0] == 8 

    def test_unwrap_positions(self):
        with pytest.raises(PBCPoorDefinedError):
            self.trajectory.unwrap_positions()

        self.trajectory.set_pbc_matrix(np.diag([11]*3))
        self.trajectory.append(positions=np.ones([self.topology.num_particles, SPATIAL_DIM]))
        self.trajectory.append(positions=np.ones([self.topology.num_particles, SPATIAL_DIM])*10)
        self.trajectory.unwrap_positions()
        assert np.isclose(self.trajectory.unwrapped_position[1, 0, 0], -1)