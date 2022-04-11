#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : charmm_forcefield.py
created time : 2021/10/05
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import numpy as np
from mdpy.forcefield import Forcefield
from mdpy.core import Ensemble
from mdpy.core import Topology
from mdpy.io import CharmmTopparParser
from mdpy.utils import check_quantity_value, check_pbc_matrix
from mdpy.constraint import *
from mdpy.unit import *
from mdpy.error import *

class CharmmForcefield(Forcefield):
    def __init__(
        self, topology: Topology, pbc_matrix: np.ndarray,
        cutoff_radius=12, long_range_solver='PME', ewald_direct_sum_error=1e-6,
        is_SHAKE: bool=True
    ) -> None:
        super().__init__(topology)
        pbc_matrix = check_quantity_value(pbc_matrix, default_length_unit)
        self._pbc_matrix = check_pbc_matrix(pbc_matrix)
        self._cutoff_radius = check_quantity_value(cutoff_radius, default_length_unit)
        self._long_range_solver = check_long_range_solver(long_range_solver)
        self._ewald_direct_sum_error = ewald_direct_sum_error
        self._is_SHAKE = is_SHAKE

    def set_param_files(self, *file_pathes) -> None:
        self._parameters = CharmmTopparParser(*file_pathes).parameters

    def check_parameters(self):
        particle_keys = self._parameters['nonbonded'].keys()
        bond_keys = self._parameters['bond'].keys()
        angle_keys = self._parameters['angle'].keys()
        dihedral_keys = self._parameters['dihedral'].keys()
        improper_keys = self._parameters['improper'].keys()
        for particle in self._topology.particles:
            particle_type = particle.particle_type
            if not particle_type in particle_keys:
                raise ParameterPoorDefinedError(
                    'The nonbonded parameter for particle %d (%s) can not be found'
                    %(particle_type)
                )
        for bond in self._topology.bonds:
            bond_name = (
                self._topology.particles[bond[0]].particle_type + '-' +
                self._topology.particles[bond[1]].particle_type
            )
            if not bond_name in bond_keys:
                raise ParameterPoorDefinedError(
                    'The parameter for bond %d-%d (%s) can not be found'
                    %(*bond, bond_name)
                )
        for angle in self._topology.angles:
            angle_name = (
                self._topology.particles[angle[0]].particle_type + '-' +
                self._topology.particles[angle[1]].particle_type + '-' +
                self._topology.particles[angle[2]].particle_type
            )
            if not angle_name in angle_keys:
                raise ParameterPoorDefinedError(
                    'The parameter for angle %d-%d-%d (%s) can not be found'
                    %(*angle, angle_name)
                )
        for dihedral in self._topology.dihedrals:
            dihedral_name = (
                self._topology.particles[dihedral[0]].particle_type + '-' +
                self._topology.particles[dihedral[1]].particle_type + '-' +
                self._topology.particles[dihedral[2]].particle_type + '-' +
                self._topology.particles[dihedral[3]].particle_type
            )
            if not dihedral_name in dihedral_keys:
                raise ParameterPoorDefinedError(
                    'The parameter for dihedral %d-%d-%d-%d (%s) can not be found'
                    %(*dihedral, dihedral_name)
                )
        for improper in self._topology.impropers:
            improper_name = (
                self._topology.particles[improper[0]].particle_type + '-' +
                self._topology.particles[improper[1]].particle_type + '-' +
                self._topology.particles[improper[2]].particle_type + '-' +
                self._topology.particles[improper[3]].particle_type
            )
            if not improper_name in improper_keys:
                raise ParameterPoorDefinedError(
                    'The parameter for improper %d-%d-%d-%d (%s) can not be found'
                    %(*improper, improper_name)
                )

    def create_ensemble(self):
        self.check_parameters()
        ensemble = Ensemble(self._topology, self._pbc_matrix)
        constraints = []
        if self._topology.num_particles != 0:
            if self._long_range_solver == 'CUTOFF':
                constraints.append(ElectrostaticCutoffConstraint(self._cutoff_radius))
            elif self._long_range_solver == 'PME':
                constraints.append(ElectrostaticPMEConstraint(
                    self._cutoff_radius, self._ewald_direct_sum_error
                ))
            constraints.append(CharmmVDWConstraint(self._parameters['nonbonded'], self._cutoff_radius))
        if self._topology.num_bonds != 0:
            constraints.append(CharmmBondConstraint(self._parameters['bond']))
        if self._topology.num_angles != 0:
            constraints.append(CharmmAngleConstraint(self._parameters['angle']))
        if self._topology.num_dihedrals != 0:
            constraints.append(CharmmDihedralConstraint(self._parameters['dihedral']))
        if self._topology.num_impropers != 0:
            constraints.append(CharmmImproperConstraint(self._parameters['improper']))
        ensemble.add_constraints(*constraints)
        return ensemble