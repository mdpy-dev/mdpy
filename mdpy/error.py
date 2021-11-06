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

class EnvironmentVariableError(Exception):
    '''This error occurs when:
    - The environment variable is not supported

    Used in:
    - mdpy.environment
    '''
    pass

class UnitDimensionDismatchedError(Exception):
    '''This error occurs when:
    - The base dimension of two quantities is dismatched for a specific operation.
    
    Used in:
    - mdpy.unit.base_dimension
    '''
    pass

class ChangeDeviceBoundedDataError(Exception):
    '''This error occurs when:
    - Users try to change the immutable jax.DeviceArray
    
    Used in:
    - mdpy.unit.quantity
    '''
    pass

class SpatialDimError(Exception):
    '''This error occurs when:
    - The dimension of quantity in cartesian reference is not 3
    
    Used in:
    - mdpy.core.topology
    - mdpy.core.state
    - mdpy.ensemble
    '''
    pass

class GeomtryDimError(Exception):
    '''This error occurs when:
    - The dimension of geometry, like bond, angle, is mismatched
    
    Used in:
    - mdpy.core.topology
    '''
    pass

class ParticleConflictError(Exception):
    '''This error occurs when:
    - Particle is twice bounded to a Particle instance
    - Particle is bounded to itself
    - Particle is twice bounded to a Toplogy instance 
    - Particle appears twice in bond, angle, dihedral or improper
    - The number of particles is mismatched with the dimension of positions, velocities, forces matrix
    
    Used in:
    - mdpy.core.particle
    - mdpy.core.topology
    - mdpy.core.state
    - mdpy.ensemble
    '''
    pass

class ConstraintConflictError(Exception):
    '''This error occurs when:
    - Constraint is twice bounded to a Ensemble instance
    
    Used in:
    - mdpy.ensemble
    '''
    pass

class NonBoundedError(Exception):
    '''This error occurs when:
    - Parent object is not bounded 
    
    Used in:
    - mdpy.core.segment
    '''
    pass

class FileFormatError(Exception):
    '''This error occurs when:
    - file suffix or prefix appears in an unexpected way
    
    Used in:
    - mdpy.file.charmm_file
    '''
    pass

class PBCPoorDefinedError(Exception):
    '''This error occurs when:
    - Two or more column vector in pbc_matrix is linear corellated
    
    Used in:
    - mdpy.core.state
    '''
    pass

class CellListPoorDefinedError(Exception):
    '''This error occurs when:
    - The pbc info of cell list is not defined well
    - The cutoff_radius of cell list is not defined well

    Used in:
    - mdpy.core.cell_list
    '''

class ParameterPoorDefinedError(Exception):
    '''This error occurs when:
    - Topology connections' parameter is not defined in selected parameter file
    
    Used in:
    - mdpy.forcefield.charmm_forcefield
    '''
    pass

class DumperPoorDefinedError(Exception):
    '''This error occurs when:
    - Dump frequency of dumper object is 0
    - Simulation samples without adding dumper
    - LogDumper requires rest_time without providing total_step
    
    Used in:
    - mdpy.dumper.dumper
    - mdpy.dumper.log_dumper
    - mdpy.simulation
    '''
    pass

class AtomLossError(Exception):
    '''This error occurs when:
    - The atom go beyond the range of two PBC images
    
    Used in:
    - mdpy.core.state'''
    pass