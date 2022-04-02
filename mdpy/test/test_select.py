#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_select.py
created time : 2022/02/20
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import os
import pytest
from mdpy.core import Particle, Topology, Trajectory
from mdpy.io import PSFParser, PDBParser
from mdpy.utils.select import *
from mdpy.error import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data')

def create_topology():
    particles = []
    particles.append(Particle(particle_name='C', particle_type='CA', mass=12))
    particles.append(Particle(particle_name='N', particle_type='NA', mass=14))
    particles.append(Particle(particle_name='C', particle_type='CB', mass=12))
    particles.append(Particle(particle_name='H', particle_type='HN', mass=1))
    particles.append(Particle(particle_name='C', particle_type='C', mass=12))
    particles.append(Particle(particle_name='H', particle_type='HC1', mass=1))
    particles.append(Particle(particle_name='H', particle_type='HC2', mass=1))
    particles.append(Particle(particle_name='N', particle_type='N', mass=14))
    particles.append(Particle(particle_name='C', particle_type='CD', mass=12))
    topology = Topology()
    topology.add_particles(particles)
    topology.join()
    return topology

def create_trajectory():
    topology = create_topology()
    return Trajectory(topology)

def test_check_target():
    with pytest.raises(TypeError):
        check_topology(1)

    with pytest.raises(TypeError):
        check_trajectory(1)

def test_check_selection_condition():
    with pytest.raises(SelectionConditionPoorDefinedError):
        condition = [{'earby': [[0], 3]}]
        check_selection_condition(condition)

    with pytest.raises(SelectionConditionPoorDefinedError):
        condition = [{'nearby': [[0], 3]}]
        check_topological_selection_condition(condition)

def test_parse_selection_condition():
    condition = [
        {
            'particle name': [['C', 'CA']],
            'not molecule type': [['VAL']]
        },
        {'molecule id': [[3]]}
    ]
    res = parse_selection_condition(condition)
    assert res == 'particle name: [\'C\', \'CA\'] and not molecule type: [\'VAL\'] or molecule id: [3]'

def test_select():
    topology = PSFParser(os.path.join(data_dir, '6PO6.psf')).topology
    position = PDBParser(os.path.join(data_dir, '6PO6.pdb')).positions
    condition = [
        {
            'particle name': [['C', 'CA']],
            'not molecule type': [['VAL']]
        },
        {'molecule id': [[3]]}
    ]
    res = select(topology, condition)
    assert 20 in res
    assert 46 in res
    
    trajectory = Trajectory(topology)
    trajectory.set_pbc_matrix(np.diag([10]*3))
    trajectory.append(position)
    trajectory.append(position)
    condition = [
        {'nearby': [[0], 3], 'particle name': [['C', 'CA']], 'molecule type': [['VAL']]},
        {'particle id': [[10, 11]]}
    ]
    res = select(trajectory, condition)
    assert 4 in res[0]

    with pytest.raises(SelectionConditionPoorDefinedError):
        condition = [{'nearby': [[0], 3]}]
        select(topology, condition)

    with pytest.raises(SelectionConditionPoorDefinedError):
        condition = [{'earby': [[0], 3]}]
        select(trajectory, condition)