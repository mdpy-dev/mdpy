#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : error.py
created time : 2021/09/28
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
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

class ArrayDimError(Exception):
    '''This error occurs when:
    - The dimension of argument does not meet the requirement

    Used in:
    - mdpy.core.state
    - mdpy.core.trajectory
    - mdpy.io.hdf5_parser
    - mdpy.io.pdb_parser
    - mdpy.io.dcd_parser
    - mdpy.utils.pbc
    - mdpy.analyser.mobility_analyser
    '''
    pass

class ModifyJoinedTopologyError(Exception):
    '''This error occurs when:
    - Adding particle or topology geometry to a joined Topology object

    Used in:
    - mdpy.core.topology
    '''
    pass

class FileFormatError(Exception):
    '''This error occurs when:
    - file suffix or prefix appears in an unexpected way

    Used in:
    - mdpy.io.charmm_toppar_parser
    - mdpy.io.pdb_parser
    - mdpy.io.pdb_writer
    - mdpy.io.psf_parser
    - mdpy.io.hdf5_parser
    - mdpy.io.hdf5_writer
    - mdpy.io.dcd_parser
    - mdpy.analyser.analyser_result
    - mdpy.dumper.dumper
    '''
    pass

class PBCPoorDefinedError(Exception):
    '''This error occurs when:
    - Two or more column vector in pbc_matrix is linear corellated

    Used in:
    - mdpy.utils.pbc
    - mdpy.core.trajectory
    '''
    pass

class ParameterPoorDefinedError(Exception):
    '''This error occurs when:
    - Topology connections' parameter is not defined in selected parameter file

    Used in:
    - mdpy.forcefield.charmm_forcefield
    '''
    pass

class ParserPoorDefinedError(Exception):
    '''This error occurs when:
    - A complementary property is required while parser init with keywords `is_parse_all=False`

    Used in:
    - mdpy.io.hdf5_parser
    - mdpy.io.pdb_parser
    - mdpy.io.dcd_parser
    '''
    pass

class ParticleLossError(Exception):
    '''This error occurs when:
    - The particle go beyond the range of two PBC images

    Used in:
    - mdpy.utils.pbc
    '''
    pass