#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : charmm_dihedral_constraint.py
created time : 2021/10/11
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np
import numba as nb
from . import Constraint
from .. import NUMPY_INT, NUMPY_FLOAT, NUMBA_INT, NUMBA_FLOAT
from ..ensemble import Ensemble
from ..math import *
from ..unit import *

@nb.njit((NUMBA_INT[:, :], NUMBA_FLOAT[:, :], NUMBA_FLOAT[:, :], NUMBA_FLOAT[:, :], NUMBA_FLOAT[:, :]))
def cpu_kernel(int_params, float_params, positions, pbc_matrix, pbc_inv):
    forces = np.zeros_like(positions, dtype=NUMPY_FLOAT)
    potential_energy = NUMPY_FLOAT(0)
    num_params = int_params.shape[0]
    for dihedral in range(num_params):
        id1, id2, id3, id4= int_params[dihedral, :]
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
        k, n, delta = float_params[dihedral, :]
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
        
class CharmmDihedralConstraint(Constraint):
    def __init__(self, params, force_id: int = 0, force_group: int = 0) -> None:
        super().__init__(params, force_id=force_id, force_group=force_group)
        self._int_params = []
        self._float_params = []
        self._num_dihedrals = 0

    def __repr__(self) -> str:
        return '<mdpy.constraint.CharmmDihedralConstraint object>'

    __str__ = __repr__

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._force_id = ensemble.constraints.index(self)
        self._int_params = []
        self._float_params = []
        self._num_dihedrals = 0
        for dihedral in self._parent_ensemble.topology.dihedrals:
            dihedral_type = '%s-%s-%s-%s' %(
                self._parent_ensemble.topology.particles[dihedral[0]].particle_name,
                self._parent_ensemble.topology.particles[dihedral[1]].particle_name,
                self._parent_ensemble.topology.particles[dihedral[2]].particle_name,
                self._parent_ensemble.topology.particles[dihedral[3]].particle_name
            )
            for float_param in self._params[dihedral_type]:
                self._int_params.append([
                    self._parent_ensemble.topology.particles[dihedral[0]].matrix_id,
                    self._parent_ensemble.topology.particles[dihedral[1]].matrix_id,
                    self._parent_ensemble.topology.particles[dihedral[2]].matrix_id,
                    self._parent_ensemble.topology.particles[dihedral[3]].matrix_id
                ])
                self._float_params.append(float_param)
            self._num_dihedrals += 1
        self._int_params = np.vstack(self._int_params).astype(NUMPY_INT)
        self._float_params = np.vstack(self._float_params).astype(NUMPY_FLOAT)

    def update(self):
        self._check_bound_state()
        # V(dihedral) = Kchi(1 + cos(n(chi) - delta))
        self._forces, self._potential_energy = cpu_kernel(
            self._int_params, self._float_params, 
            self._parent_ensemble.state.positions, 
            *self._parent_ensemble.state.pbc_info
        )

    @property
    def num_dihedrals(self):
        return self._num_dihedrals