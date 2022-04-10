#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : charmm_dihedral_constraint.py
created time : 2021/10/11
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import numpy as np
import numba as nb
from mdpy import env
from mdpy.core import Ensemble
from mdpy.constraint import Constraint
from mdpy.utils import *
from mdpy.unit import *

class CharmmDihedralConstraint(Constraint):
    def __init__(self, parameters, force_id: int = 0, force_group: int = 0) -> None:
        super().__init__(parameters, force_id=force_id, force_group=force_group)
        self._int_parameters = []
        self._float_parameters = []
        self._num_dihedrals = 0
        self._kernel = nb.njit(
            (env.NUMBA_INT[:, :], env.NUMBA_FLOAT[:, :], env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[:, ::1])
        )(self.kernel)

    def __repr__(self) -> str:
        return '<mdpy.constraint.CharmmDihedralConstraint object>'

    def __str__(self) -> str:
        return 'Dihedral constraint'

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._force_id = ensemble.constraints.index(self)
        self._int_parameters = []
        self._float_parameters = []
        self._num_dihedrals = 0
        for dihedral in self._parent_ensemble.topology.dihedrals:
            dihedral_type = '%s-%s-%s-%s' %(
                self._parent_ensemble.topology.particles[dihedral[0]].particle_type,
                self._parent_ensemble.topology.particles[dihedral[1]].particle_type,
                self._parent_ensemble.topology.particles[dihedral[2]].particle_type,
                self._parent_ensemble.topology.particles[dihedral[3]].particle_type
            )
            for float_param in self._parameters[dihedral_type]:
                self._int_parameters.append([
                    self._parent_ensemble.topology.particles[dihedral[0]].matrix_id,
                    self._parent_ensemble.topology.particles[dihedral[1]].matrix_id,
                    self._parent_ensemble.topology.particles[dihedral[2]].matrix_id,
                    self._parent_ensemble.topology.particles[dihedral[3]].matrix_id
                ])
                self._float_parameters.append(float_param)
            self._num_dihedrals += 1
        self._int_parameters = np.vstack(self._int_parameters).astype(env.NUMPY_INT)
        self._float_parameters = np.vstack(self._float_parameters).astype(env.NUMPY_FLOAT)

    @staticmethod
    def kernel(int_parameters, float_parameters, positions, pbc_matrix, pbc_inv):
        forces = np.zeros_like(positions)
        potential_energy = forces[0, 0]
        num_parameters = int_parameters.shape[0]
        for dihedral in range(num_parameters):
            id1, id2, id3, id4= int_parameters[dihedral, :]
            theta = get_pbc_dihedral(
                positions[id1, :], positions[id2, :],
                positions[id3, :], positions[id4, :],
                pbc_matrix, pbc_inv
            )
            vab = unwrap_vec(positions[id2, :] - positions[id1, :], pbc_matrix, pbc_inv)
            lab = np.linalg.norm(vab)
            vbc = unwrap_vec(positions[id3, :] - positions[id2, :], pbc_matrix, pbc_inv)
            lbc = np.linalg.norm(vbc)
            voc, loc = vbc / 2, lbc / 2
            vcd = unwrap_vec(positions[id4, :] - positions[id3, :], pbc_matrix, pbc_inv)
            lcd = np.linalg.norm(vcd)
            theta_abc = np.arccos(np.dot(-vab, vbc) / (lab * lbc))
            theta_bcd = np.arccos(np.dot(-vbc, vcd) / (lbc * lcd))
            k, n, delta = float_parameters[dihedral, :]
            # Forces
            force_val = - k * (1 - n * np.sin(n*theta - delta))
            force_a = force_val / (lab * np.sin(theta_abc)) * get_unit_vec(np.cross(-vab, vbc))
            force_d = force_val / (lcd * np.sin(theta_bcd)) * get_unit_vec(np.cross(vcd, -vbc))
            force_c =  np.cross(
                - (np.cross(voc, force_d) + np.cross(vcd, force_d) / 2 + np.cross(-vab, force_a) / 2),
                voc
            ) / loc**2
            force_b = - (force_a + force_c + force_d)
            forces[id1, :] += force_a
            forces[id2, :] += force_b
            forces[id3, :] += force_c
            forces[id4, :] += force_d
            # Potential energy
            potential_energy += k * (1 + np.cos(n*theta - delta))
        return forces, potential_energy

    def update(self):
        self._check_bound_state()
        # V(dihedral) = Kchi(1 + cos(n(chi) - delta))
        self._forces, self._potential_energy = self._kernel(
            self._int_parameters, self._float_parameters,
            self._parent_ensemble.state.positions,
            *self._parent_ensemble.state.pbc_info
        )

    @property
    def num_dihedrals(self):
        return self._num_dihedrals