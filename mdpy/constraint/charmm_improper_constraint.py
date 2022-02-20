#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : charmm_improper_constraint.py
created time : 2021/10/12
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np
import numba as nb
from . import Constraint
from .. import env
from ..ensemble import Ensemble
from ..utils import *
class CharmmImproperConstraint(Constraint):
    def __init__(self, params, force_id: int = 0, force_group: int = 0) -> None:
        super().__init__(params, force_id=force_id, force_group=force_group)
        self._int_params = []
        self._float_params = []
        self._num_impropers = 0
        self._kernel = nb.njit(
            (env.NUMBA_INT[:, :], env.NUMBA_FLOAT[:, :], env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[:, ::1], env.NUMBA_FLOAT[:, ::1])
        )(self.kernel)

    def __repr__(self) -> str:
        return '<mdpy.constraint.CharmmImproperConstraint object>'

    __str__ = __repr__

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._force_id = ensemble.constraints.index(self)
        self._int_params = []
        self._float_params = []
        self._num_impropers = 0
        for improper in self._parent_ensemble.topology.impropers:
            improper_type = '%s-%s-%s-%s' %(
                self._parent_ensemble.topology.particles[improper[0]].particle_name,
                self._parent_ensemble.topology.particles[improper[1]].particle_name,
                self._parent_ensemble.topology.particles[improper[2]].particle_name,
                self._parent_ensemble.topology.particles[improper[3]].particle_name
            )
            self._int_params.append([
                self._parent_ensemble.topology.particles[improper[0]].matrix_id,
                self._parent_ensemble.topology.particles[improper[1]].matrix_id,
                self._parent_ensemble.topology.particles[improper[2]].matrix_id,
                self._parent_ensemble.topology.particles[improper[3]].matrix_id
            ])
            self._float_params.append(self._params[improper_type])
            self._num_impropers += 1
        self._int_params = np.vstack(self._int_params).astype(env.NUMPY_INT)
        self._float_params = np.vstack(self._float_params).astype(env.NUMPY_FLOAT)

    @staticmethod
    def kernel(int_params, float_params, positions, pbc_matrix, pbc_inv):
        forces = np.zeros_like(positions)
        potential_energy = forces[0, 0]
        num_params = int_params.shape[0]
        for improper in range(num_params):
            id1, id2, id3, id4 = int_params[improper, :]
            k, psi0 = float_params[improper, :] 
            psi = get_pbc_dihedral(
                positions[id1, :], positions[id2, :],
                positions[id3, :], positions[id4, :],
                pbc_matrix, pbc_inv
            )
            # Forces
            force_val = - 2 * k * (psi - psi0)
            vab = unwrap_vec(positions[id2, :] - positions[id1, :], pbc_matrix, pbc_inv)
            lab = np.linalg.norm(vab)
            vbc = unwrap_vec(positions[id3, :] - positions[id2, :], pbc_matrix, pbc_inv)
            lbc = np.linalg.norm(vbc)
            voc, loc = vbc / 2, lbc / 2
            vcd = unwrap_vec(positions[id4, :] - positions[id3, :], pbc_matrix, pbc_inv)
            lcd = np.linalg.norm(vcd)
            theta_abc = np.arccos(np.dot(-vab, vbc) / (lab * lbc))
            theta_bcd = np.arccos(np.dot(-vbc, vcd) / (lbc * lcd))
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
            potential_energy += k * (psi - psi0)**2
        return forces, potential_energy

    def update(self):
        self._check_bound_state()
        # V(improper) = Kpsi(psi - psi0)**2
        self._forces, self._potential_energy = self._kernel(
            self._int_params, self._float_params, 
            self._parent_ensemble.state.positions, 
            *self._parent_ensemble.state.pbc_info
        )

    @property
    def num_impropers(self):
        return self._num_impropers