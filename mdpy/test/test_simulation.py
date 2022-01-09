#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_simulation.py
created time : 2021/10/19
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest
import numpy as np
from ..core import Particle, Topology
from ..ensemble import Ensemble
from ..integrator import Integrator
from ..simulation import Simulation
from ..error import *

class TestSimulation:
    def setup(self):
        self.particles = []
        self.particles.append(Particle(
            particle_type='C', particle_name='CA', 
            molecule_id=0, molecule_type='ALA', chain_id='A'
        ))
        self.particles.append(Particle(
            particle_type='N', particle_name='N', 
            molecule_id=0, molecule_type='ALA', chain_id='A'
        ))
        self.particles.append(Particle(
            particle_type='C', particle_name='CB', 
            molecule_id=0, molecule_type='ALA', chain_id='A'
        ))
        self.particles.append(Particle(
            particle_type='O', particle_name='O', 
            molecule_id=0, molecule_type='ALA', chain_id='A'
        ))
        self.particles.append(Particle(
            particle_type='C', particle_name='CA', 
            molecule_id=1, molecule_type='LEU', chain_id='A'
        ))
        self.particles.append(Particle(
            particle_type='N', particle_name='N', 
            molecule_id=1, molecule_type='LEU', chain_id='A'
        ))
        self.particles.append(Particle(
            particle_type='C', particle_name='CB', 
            molecule_id=1, molecule_type='LEU', chain_id='A'
        ))
        self.particles.append(Particle(
            particle_type='O', particle_name='O', 
            molecule_id=1, molecule_type='LEU', chain_id='A'
        ))
        self.particles.append(Particle(
            particle_type='O', particle_name='O', 
            molecule_id=2, molecule_type='OHO', chain_id='W'
        ))
        self.particles.append(Particle(
            particle_type='H', particle_name='H', 
            molecule_id=2, molecule_type='OHO', chain_id='W'
        ))
        self.particles.append(Particle(
            particle_type='H', particle_name='H', 
            molecule_id=2, molecule_type='OHO', chain_id='W'
        ))
        num_particles = len(self.particles)
        self.topology = Topology()
        self.topology.add_particles(self.particles)
        self.ensemble = Ensemble(self.topology)
        self.ensemble.state.set_pbc_matrix(np.diag(np.ones(3)*100))
        self.ensemble.state.cell_list.set_cutoff_radius(12)
        positions = np.array(list(range(num_particles)))
        positions = np.vstack([positions, positions, positions]).T
        self.ensemble.state.set_positions(positions)
        self.integrator = Integrator(1)
        self.simulation = Simulation(self.ensemble, self.integrator)

    def teardown(self):
        self.particles = None
        self.topology = None
        self.ensemble = None
        self.simulation = None

    def test_attributes(self):
        assert self.simulation.cur_step == 0
        assert self.simulation.num_dumpers == 0

    def test_exceptions(self):
        with pytest.raises(DumperPoorDefinedError):
            self.simulation.integrate(1)