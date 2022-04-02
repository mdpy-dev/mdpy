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

class GeomtryDimError(Exception):
    '''This error occurs when:
    - The dimension of geometry, like bond, angle, is mismatched

    Used in:
    - mdpy.core.topology
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
    - mdpy.core.trajectory
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

class ModifyJoinedTopologyError(Exception):
    '''This error occurs when:
    - Adding particle or topology geometry to a joined Topology object

    Used in:
    - mdpy.core.topology
    '''
    pass

class NonBoundedError(Exception):
    '''This error occurs when:
    - Parent object is not bounded

    Used in:
    - mdpy.constraint.constraint
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

class CellListPoorDefinedError(Exception):
    '''This error occurs when:
    - The pbc info of cell list is not defined well
    - The cutoff_radius of cell list is not defined well

    Used in:
    - mdpy.core.cell_list
    '''
    pass

class TrajectoryPoorDefinedError(Exception):
    '''This error occurs when:
    - Extrating information that have not been contained

    Used in:
    -mdpy.core.trajectory
    '''
    pass

class ParameterPoorDefinedError(Exception):
    '''This error occurs when:
    - Topology connections' parameter is not defined in selected parameter file

    Used in:
    - mdpy.forcefield.charmm_forcefield
    '''
    pass

class HDF5FilePoorDefinedError(Exception):
    '''This error occurs when:
    - Use hdf5 file that does not meet the requirement of mdpy

    Used in:
    - mdpy.io.hdf5_parser
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

class DumperPoorDefinedError(Exception):
    '''This error occurs when:
    - Dump frequency of dumper object is 0
    - Simulation integrates without adding dumper
    - LogDumper requires rest_time without providing total_step

    Used in:
    - mdpy.dumper.dumper
    - mdpy.dumper.log_dumper
    - mdpy.simulation
    '''
    pass

class SelectionConditionPoorDefinedError(Exception):
    '''This error occurs when:
    - Unsupported selection condition has been used

    Used in:
    - mdpy.utils.select_particle
    '''
    pass

class AnalyserPoorDefinedError(Exception):
    '''This error occurs when:
    - Analyser's input data does not meet analyser's initial setting

    Used in:
    - mdpy.analyser.analyser_result
    - mdpy.analyser.rmsd_analyser
    '''
    pass

class ParticleLossError(Exception):
    '''This error occurs when:
    - The particle go beyond the range of two PBC images

    Used in:
    - mdpy.utils.pbc
    '''
    pass