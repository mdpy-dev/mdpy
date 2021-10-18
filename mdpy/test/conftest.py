#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : conftest.py
created time : 2021/09/28
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest

test_order = [
    'base_dimension', 'unit', 'unit_definition', 'quantity',
    'particle', 'topology', 'state', 'segment',
    'pdb_file', 'psf_file', 'charmm_param_file',
    'constraint', 
    'charmm_nonbonded_constraint', 'charmm_bond_constraint', 'charmm_angle_constraint',
    'charmm_dihedral_constraint', 'charmm_improper_constraint',
    'forcefield', 'charmm_forcefield'
    'ensemble',
    'integrator'
]

def pytest_collection_modifyitems(items):
     current_index = 0
     for test in test_order:
          indexes = []
          for id, item in enumerate(items):
               if 'test_'+test+'.py' in item.nodeid:
                    indexes.append(id)  
          for id, index in enumerate(indexes):
               items[current_index+id], items[index] = items[index], items[current_index+id]
          current_index += len(indexes)