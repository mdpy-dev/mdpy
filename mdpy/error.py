#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : error.py
created time : 2021/09/28
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

class UnitDimensionDismatchedError(Exception):
    '''This error occurs when the base dimension of two quantities is dismatched for a specific operation.
    Used in:
    - mdpy.unit.base_dimension
    '''
    pass

class ChangeDeviceBoundedDataError(Exception):
    '''This error occurs when users try to change the immutable jax.DeviceArray
    Used in:
    - mdpy.unit.quantity
    '''
    pass

class SpatialDimError(Exception):
    '''This error occurs when the dimension of quantity in cartesian reference is not 3
    Used in:
    - mdpy.core.particle
    '''
    pass

class GeomtryDimError(Exception):
    '''This error occurs when the dimension of geometry, like bond, angle, is mismatched
    Used in:
    - mdpy.core.topology
    '''
    pass

class ParticleConflictError(Exception):
    '''This error occurs when Particle is twice bounded to a Toplogy instance or Particle appera twice in bond, angle, dihedral or improper
    Used in:
    - mdpy.core.topology
    '''

class NonBoundedError(Exception):
    '''This error occurs when parent object is not bounded. 
    Used in:
    - mdpy.core.segment
    '''
    pass

class FileFormatError(Exception):
    '''This error occurs when file suffix or prefix appears in an unexpected way
    Used in:
    - mdpy.file.charmm_file
    '''
    pass