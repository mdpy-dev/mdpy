#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_pdb_dumper.py
created time : 2021/10/19
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest, os
import numpy as np
from ..core import Particle, Topology
from ..ensemble import Ensemble
from ..integrator import Integrator
from ..simulation import Simulation
from ..dumper import PDBDumper

cur_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(cur_dir, 'out')

class TestPDBDumper:
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
        self.file_path = os.path.join(out_dir, 'test_pdb_dumper.pdb')

    def teardown(self):
        self.particles = None
        self.topology = None
        self.ensemble = None
        self.simulation = None

    def test_attributes(self):
        pass

    def test_exceptions(self):
        pass

    def test_dump_header(self):
        dumper = PDBDumper(self.file_path, 1)
        dumper._dump_pdb_header(self.ensemble)

    def test_dump_model(self):
        dumper = PDBDumper(self.file_path, 1)
        dumper._dump_pdb_header(self.ensemble)
        dumper._dump_pdb_model(self.ensemble, 0)

    def test_dump(self):
        dumper = PDBDumper(self.file_path, 1)
        dumper.dump(self.simulation)
        new_position = self.simulation.ensemble.state.positions
        new_position[:, 0] += 1
        self.simulation.ensemble.state.set_positions(new_position)
        dumper.dump(self.simulation)