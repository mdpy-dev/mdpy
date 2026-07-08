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

class PrecisionError(Exception):
    '''This error occurs when:
    - The requested numeric precision is not supported

    Used in:
    - mdpy.precision
    '''
    pass

class UnitDimensionMismatchedError(Exception):
    '''This error occurs when:
    - The base dimension of two quantities is mismatched for a specific operation.

    Used in:
    - mdpy.unit.base_dimension
    '''
    pass

class ArrayDimensionError(Exception):
    '''This error occurs when:
    - The dimension of argument does not meet the requirement

    Used in:
    - mdpy.io.pdb_parser
    '''
    pass

class FileFormatError(Exception):
    '''This error occurs when:
    - file suffix or prefix appears in an unexpected way

    Used in:
    - mdpy.io.charmm_toppar_parser
    - mdpy.io.pdb_parser
    - mdpy.io.psf_parser
    '''
    pass

class ParserPoorlyDefinedError(Exception):
    '''This error occurs when:
    - A complementary property is required while parser init with keywords `is_parse_all=False`

    Used in:
    - mdpy.io.pdb_parser
    '''
    pass
