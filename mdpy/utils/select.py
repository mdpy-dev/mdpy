#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : select_particle.py
created time : 2022/02/20
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np
from .check_quantity import check_quantity_value
from .pbc import unwrap_vec
from .. import SPATIAL_DIM
from ..core import Topology
from ..core import Trajectory
from ..unit import *
from ..error import *

def check_topology(target) -> Topology:
    if isinstance(target, Topology):
        return target
    elif isinstance(target, Trajectory):
        return target.topology
    else:
        raise TypeError(
            'The target of select_particle functions should be instance of ' \
            'mdpy.core.Topology or mdpy.core.Trajectory class '
            'while %s is provided' %type(target)
        )

def check_trajectory(target) -> Trajectory:
    if isinstance(target, Trajectory):
        return target
    else:
        raise TypeError(
            'The target of select_particle functions should be instance of ' \
            'mdpy.core.Trajectory class while %s is provided' %type(target)
        )

# Topological
def select_all(target):
    topology = check_topology(target)
    return list(range(topology.num_particles))

def select_particle_type(target, particle_types: list[str]):
    selected_matrix_ids = []
    topology = check_topology(target)
    for particle in topology.particles:
        if particle.particle_type in particle_types:
            selected_matrix_ids.append(particle.matrix_id)
    return selected_matrix_ids

def select_particle_name(target, particle_names: list[str]):
    selected_matrix_ids = []
    topology = check_topology(target)
    for particle in topology.particles:
        if particle.particle_name in particle_names:
            selected_matrix_ids.append(particle.matrix_id)
    return selected_matrix_ids

def select_particle_id(target, particle_ids: list[int]):
    selected_matrix_ids = []
    topology = check_topology(target)
    num_left_particles = len(particle_ids)
    for particle in topology.particles:
        if particle.particle_id in particle_ids:
            selected_matrix_ids.append(particle.matrix_id)
            num_left_particles -= 1
            if num_left_particles == 0:
                break
    return selected_matrix_ids

def select_molecule_type(target, molecule_types: list[str]):
    selected_matrix_ids = []
    topology = check_topology(target)
    for particle in topology.particles:
        if particle.molecule_type in molecule_types:
            selected_matrix_ids.append(particle.matrix_id)
    return selected_matrix_ids

def select_molecule_id(target, molecule_ids: list[int]):
    selected_matrix_ids = []
    topology = check_topology(target)
    for particle in topology.particles:
        if particle.molecule_id in molecule_ids:
            selected_matrix_ids.append(particle.matrix_id)
    return selected_matrix_ids

def select_chain_id(target, chain_ids: list[int]):
    selected_matrix_ids = []
    topology = check_topology(target)
    for particle in topology.particles:
        if particle.chain_id in chain_ids:
            selected_matrix_ids.append(particle.matrix_id)
    return selected_matrix_ids

def select_water(target):
    matrix_ids = []
    topology = check_topology(target)
    return matrix_ids

def select_ion():
    pass

def select_protein():
    pass

def select_nucleic_acid():
    pass

# Steric
def select_nearby(target: Trajectory, frame, matrix_ids, radius):
    trajectory = check_trajectory(target)
    radius = check_quantity_value(radius, default_length_unit)
    selected_matrix_ids = []
        
    for matrix_id in matrix_ids:
        vec = unwrap_vec(
            trajectory.positions[frame, matrix_id, :] - trajectory.positions[frame, :, :],
            trajectory.pbc_matrix, trajectory.pbc_inv
        )
        temp = list(np.where(np.sqrt((vec**2).sum(1)) < radius)[0])
        temp.remove(matrix_id)
        selected_matrix_ids.extend(temp)
    return selected_matrix_ids

def select_in_sphere(target: Trajectory, frame, center, radius):
    trajectory = check_trajectory(target)
    center = check_quantity_value(center, default_length_unit)
    radius = check_quantity_value(radius, default_length_unit)
    if center.size != SPATIAL_DIM:
        raise ArrayDimError('The shape of center should be (%d,)' %SPATIAL_DIM)
    center = center.reshape([SPATIAL_DIM])
    selected_matrix_ids = []
    vec = unwrap_vec(
        center - trajectory.positions[frame, :, :],
        trajectory.pbc_matrix, trajectory.pbc_inv
    )
    selected_matrix_ids.extend(
        list(np.where(np.sqrt((vec**2).sum(1)) < radius))
    )
    return selected_matrix_ids
    

def select_in_cubic():
    pass

def select_in_cylinder():
    pass

# select main function
SELECTION_SUPPORTED_TOPOLOGICAL_KEYWORDS = [
    'all',
    'particle type', 'particle name', 'particle id', 
    'molecule type', 'molecule id', 'chain id',
    'water', 'ion', 'protein', 'nucleic acid'
]

SELECTION_SUPPORTED_STERIC_KEYWORDS = [
    'nearby', 'in sphere', 'in cubic', 'in cylinder'
]

SELECTION_SUPPORTED_KEYWORDS = SELECTION_SUPPORTED_TOPOLOGICAL_KEYWORDS + SELECTION_SUPPORTED_STERIC_KEYWORDS

METHOD_DICT = {
    'all': select_all,
    'particle type': select_particle_type,
    'particle name': select_particle_name,
    'particle id': select_particle_id,
    'molecule type': select_molecule_type,
    'molecule id': select_molecule_id,
    'chain id': select_chain_id,

    'water': select_water,
    'ion': select_ion,
    'protein': select_protein,
    'nucleic acid': select_nucleic_acid,

    'nearby': select_nearby,
    'in sphere': select_in_sphere,
    'in cubic': select_in_cubic,
    'in cylinder': select_in_cylinder
}

def check_selection_condition(conditions: list[dict]):
    if not isinstance(conditions, list) or not isinstance(conditions[0], dict):
        raise SelectionConditionPoorDefinedError(
            'selection condition should be list[dict]'
        )
    for condition in conditions:
        for key, _ in condition.items():
            if not key in SELECTION_SUPPORTED_KEYWORDS:
                raise SelectionConditionPoorDefinedError(
                        '%s is unsupported selection keyword' \
                        ', please check mdpy.utils.SELECTION_SUPPORTED_TOPOLOGICAL_KEYWORDS' %key
                    )

def check_topological_selection_condition(conditions: list[dict]):
    if not isinstance(conditions, list) or not isinstance(conditions[0], dict):
        raise SelectionConditionPoorDefinedError(
            'selection condition should be list[dict]'
        )
    for condition in conditions:
        for key, _ in condition.items():
            if not key in SELECTION_SUPPORTED_TOPOLOGICAL_KEYWORDS:
                raise SelectionConditionPoorDefinedError(
                        '%s is unsupported selection topological keyword' \
                        ', please check mdpy.utils.SELECTION_SUPPORTED_TOPOLOGICAL_KEYWORDS' %key
                    )

def parse_selection_condition(conditions: list[dict]):
    res = []
    for condition in conditions:
        cur_string = []
        for key, value in condition.items():
            string = '%s: ' %key
            for i in value:
                string += '%s, ' %i
            cur_string.append(string[:-2]) # Remove last ', '
        cur_string = ' and '.join(cur_string)
        res.append(cur_string)
    return ' or '.join(res)

def select(target, conditions: list[dict]):
    if isinstance(target, Topology): # Single frame
        num_particles = target.num_particles
        all_set = set(range(num_particles))
        sum_matrix_ids = []
        for condition in conditions:
            current_matrix_id = all_set.copy()
            for key, value in condition.items():
                is_verse = False
                if 'not' in key:
                    is_verse = True
                    key = key.split('not')[-1].strip()
                if not key in SELECTION_SUPPORTED_TOPOLOGICAL_KEYWORDS:
                    raise SelectionConditionPoorDefinedError(
                        '%s is unsupported selection keyword for mdpy.core.Topology instance' \
                        ', please check mdpy.utils.SELECTION_SUPPORTED_TOPOLOGICAL_KEYWORDS' %key
                    )
                res = set(METHOD_DICT[key](target, *value))
                if not is_verse:
                    current_matrix_id &= res
                else:
                    current_matrix_id &= all_set ^ res
            sum_matrix_ids.extend(list(current_matrix_id))
            sum_matrix_ids = list(set(sum_matrix_ids))
            sum_matrix_ids.sort()
    elif isinstance(target, Trajectory): # Multi frame
        num_particles = target.topology.num_particles
        all_set = set(range(num_particles))
        sum_matrix_ids = [[]] * target.num_frames
        topological_res = {} # Store topological selection which only need to be calculated once
        for frame in range(target.num_frames):
            for index, condition in enumerate(conditions):
                current_matrix_id = all_set.copy()
                for key, value in condition.items():
                    is_verse = False
                    if 'not' in key:
                        is_verse = True
                        key = key.split('not')[-1].strip()
                    if key in SELECTION_SUPPORTED_TOPOLOGICAL_KEYWORDS:
                        if frame == 0:
                            res = set(METHOD_DICT[key](target, *value))
                            topological_res['%d-%s-%s' %(index, key, value)] = res
                        else:
                            res = topological_res['%d-%s-%s' %(index, key, value)]
                    elif key in SELECTION_SUPPORTED_STERIC_KEYWORDS:
                        res = set(METHOD_DICT[key](target, frame, *value))
                    else:
                        raise SelectionConditionPoorDefinedError(
                            '%s is unsupported selection keyword for mdpy.core.Trajectory instance' \
                            ', please check mdpy.utils.SELECTION_SUPPORTED_KEYWORDS' %key
                        )
                    if not is_verse:
                        current_matrix_id &= res
                    else:
                        current_matrix_id &= all_set ^ res
                sum_matrix_ids[frame].extend(list(current_matrix_id))
                sum_matrix_ids[frame] = list(set(sum_matrix_ids[frame]))
                sum_matrix_ids[frame].sort()
    else:
        raise TypeError(
            'The target of select_particle functions should be instance of ' \
            'mdpy.core.Topology or mdpy.core.Trajectory class '
            'while %s is provided' %type(target)
        )
    return sum_matrix_ids