#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_select.py
created time : 2022/02/20
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import os
import pytest
from ..core import Particle, Topology, Trajectory
from ..file import PSFFile, PDBFile
from ..utils.select import *
from ..error import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data')

def create_topology():
    particles = []
    particles.append(Particle(particle_type='C', particle_name='CA', mass=12))
    particles.append(Particle(particle_type='N', particle_name='NA', mass=14))
    particles.append(Particle(particle_type='C', particle_name='CB', mass=12))
    particles.append(Particle(particle_type='H', particle_name='HN', mass=1))
    particles.append(Particle(particle_type='C', particle_name='C', mass=12))
    particles.append(Particle(particle_type='H', particle_name='HC1', mass=1))
    particles.append(Particle(particle_type='H', particle_name='HC2', mass=1))
    particles.append(Particle(particle_type='N', particle_name='N', mass=14))
    particles.append(Particle(particle_type='C', particle_name='CD', mass=12))
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

def test_select_particle_type():
    topology = create_topology()
    selected_particles = select_particle_type(topology, ['H'])
    assert selected_particles[0] == 3
    assert len(selected_particles) == 3

    trajectory = create_trajectory()
    selected_particles = select_particle_type(trajectory, ['H', 'C'])
    assert selected_particles[0] == 0
    assert len(selected_particles) == 7

def test_parse_selection_condition():
    condition = [
        {
            'particle type': [['C', 'CA']],
            'not molecule type': [['VAL']]
        },
        {'molecule id': [[3]]}
    ]
    res = parse_selection_condition(condition)
    assert res == 'particle type: [\'C\', \'CA\'] and not molecule type: [\'VAL\'] or molecule id: [3]'

def test_select():
    topology = PSFFile(os.path.join(data_dir, '6PO6.psf')).create_topology()
    position = PDBFile(os.path.join(data_dir, '6PO6.pdb')).positions
    condition = [
        {
            'particle type': [['C', 'CA']],
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
        {'nearby': [[0], 3], 'particle type': [['C', 'CA']], 'molecule type': [['VAL']]},
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