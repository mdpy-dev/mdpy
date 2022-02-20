#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : charmm_forcefield.py
created time : 2021/10/05
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

from . import Forcefield
from ..ensemble import Ensemble
from ..core import Topology
from ..file import CharmmParamFile
from ..utils import check_quantity_value
from ..constraint import LONG_RANGE_SOLVER
from ..constraint import *
from ..unit import *
from ..error import *

class CharmmForcefield(Forcefield):
    def __init__(
        self, topology: Topology, cutoff_radius=12,
        long_range_solver='PPPM', ewald_error=1e-6,
        is_SHAKE: bool=True
    ) -> None:
        super().__init__(topology)
        self._cutoff_radius = check_quantity_value(cutoff_radius, default_length_unit)
        self._long_range_solver = long_range_solver
        self._ewald_error = ewald_error
        self._is_SHAKE = is_SHAKE

    def set_param_files(self, *file_pathes) -> None:
        self._params = CharmmParamFile(*file_pathes).params

    def check_params(self):
        particle_keys = self._params['nonbonded'].keys()
        bond_keys = self._params['bond'].keys()
        angle_keys = self._params['angle'].keys()
        dihedral_keys = self._params['dihedral'].keys()
        improper_keys = self._params['improper'].keys()
        for particle in self._topology.particles:
            particle_name = particle.particle_name
            if not particle_name in particle_keys:
                raise ParameterPoorDefinedError(
                    'The nonbonded parameter for particle %d (%s) can not be found'
                    %(particle_name)
                )
        for bond in self._topology.bonds:
            bond_name = (
                self._topology.particles[bond[0]].particle_name + '-' + 
                self._topology.particles[bond[1]].particle_name
            )
            if not bond_name in bond_keys:
                raise ParameterPoorDefinedError(
                    'The parameter for bond %d-%d (%s) can not be found' 
                    %(*bond, bond_name)
                ) 
        for angle in self._topology.angles:
            angle_name = (
                self._topology.particles[angle[0]].particle_name + '-' + 
                self._topology.particles[angle[1]].particle_name + '-' +
                self._topology.particles[angle[2]].particle_name
            )
            if not angle_name in angle_keys:
                raise ParameterPoorDefinedError(
                    'The parameter for angle %d-%d-%d (%s) can not be found' 
                    %(*angle, angle_name)
                ) 
        for dihedral in self._topology.dihedrals:
            dihedral_name = (
                self._topology.particles[dihedral[0]].particle_name + '-' + 
                self._topology.particles[dihedral[1]].particle_name + '-' +
                self._topology.particles[dihedral[2]].particle_name + '-' +
                self._topology.particles[dihedral[3]].particle_name
            )
            if not dihedral_name in dihedral_keys:
                raise ParameterPoorDefinedError(
                    'The parameter for dihedral %d-%d-%d-%d (%s) can not be found' 
                    %(*dihedral, dihedral_name)
                ) 
        for improper in self._topology.impropers:
            improper_name = (
                self._topology.particles[improper[0]].particle_name + '-' + 
                self._topology.particles[improper[1]].particle_name + '-' +
                self._topology.particles[improper[2]].particle_name + '-' +
                self._topology.particles[improper[3]].particle_name
            )
            if not improper_name in improper_keys:
                raise ParameterPoorDefinedError(
                    'The parameter for improper %d-%d-%d-%d (%s) can not be found' 
                    %(*improper, improper_name)
                ) 
            
    def create_ensemble(self):
        self.check_params()
        ensemble = Ensemble(self._topology)
        constraints = []
        if self._topology.num_particles != 0:
            constraints.append(ElectrostaticConstraint())
            constraints.append(CharmmNonbondedConstraint(self._params['nonbonded'], self._cutoff_radius))
        if self._topology.num_bonds != 0:
            constraints.append(CharmmBondConstraint(self._params['bond']))
        if self._topology.num_angles != 0:
            constraints.append(CharmmAngleConstraint(self._params['angle']))
        if self._topology.num_dihedrals != 0:
            constraints.append(CharmmDihedralConstraint(self._params['dihedral']))
        if self._topology.num_impropers != 0:
            constraints.append(CharmmImproperConstraint(self._params['improper']))
        ensemble.add_constraints(*constraints)
        return ensemble