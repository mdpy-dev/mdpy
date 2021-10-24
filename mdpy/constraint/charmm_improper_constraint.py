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
from . import Constraint
from .. import SPATIAL_DIM
from ..ensemble import Ensemble
from ..math import *

class CharmmImproperConstraint(Constraint):
    def __init__(self, params, force_id: int = 0, force_group: int = 0) -> None:
        super().__init__(params, force_id=force_id, force_group=force_group)
        self._improper_info = []
        self._num_impropers = 0

    def __repr__(self) -> str:
        return '<mdpy.constraint.CharmmImproperConstraint object>'

    __str__ = __repr__

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._force_id = ensemble.constraints.index(self)
        self._improper_info = []
        self._num_impropers = 0
        for improper in self._parent_ensemble.topology.impropers:
            improper_type = '%s-%s-%s-%s' %(
                self._parent_ensemble.topology.particles[improper[0]].particle_name,
                self._parent_ensemble.topology.particles[improper[1]].particle_name,
                self._parent_ensemble.topology.particles[improper[2]].particle_name,
                self._parent_ensemble.topology.particles[improper[3]].particle_name
            )
            matrix_id = [
                self._parent_ensemble.topology.particles[improper[0]].matrix_id,
                self._parent_ensemble.topology.particles[improper[1]].matrix_id,
                self._parent_ensemble.topology.particles[improper[2]].matrix_id,
                self._parent_ensemble.topology.particles[improper[3]].matrix_id
            ]
            self._improper_info.append(matrix_id + self._params[improper_type])
            self._num_impropers += 1

    def get_forces(self):
        self._check_bound_state()
        # V(improper) = Kpsi(psi - psi0)**2
        forces = np.zeros([self._parent_ensemble.topology.num_particles, SPATIAL_DIM])
        for improper_info in self._improper_info:
            id1, id2, id3, id4, k, psi0 = improper_info
            psi = get_pbc_dihedral(
                self._parent_ensemble.state.positions[id1, :], 
                self._parent_ensemble.state.positions[id2, :],
                self._parent_ensemble.state.positions[id3, :], 
                self._parent_ensemble.state.positions[id4, :],
                self._parent_ensemble.state.pbc_matrix,
                self._parent_ensemble.state.pbc_inv,
                is_angular=False
            )
            force_val = 2 * k * (psi - psi0)

            vab = self._parent_ensemble.state.positions[id2, :] - self._parent_ensemble.state.positions[id1, :]
            lab = np.linalg.norm(vab)
            vbc = self._parent_ensemble.state.positions[id3, :] - self._parent_ensemble.state.positions[id2, :]
            lbc = np.linalg.norm(vbc)
            voc, loc = vbc / 2, lbc / 2
            vcd = self._parent_ensemble.state.positions[id4, :] - self._parent_ensemble.state.positions[id3, :]
            lcd = np.linalg.norm(vcd)
            theta_abc = get_pbc_angle(
                self._parent_ensemble.state.positions[id1, :], 
                self._parent_ensemble.state.positions[id2, :],
                self._parent_ensemble.state.positions[id3, :],
                self._parent_ensemble.state.pbc_matrix,
                self._parent_ensemble.state.pbc_inv
            )
            theta_bcd = get_pbc_angle(
                self._parent_ensemble.state.positions[id2, :], 
                self._parent_ensemble.state.positions[id3, :],
                self._parent_ensemble.state.positions[id4, :],
                self._parent_ensemble.state.pbc_matrix,
                self._parent_ensemble.state.pbc_inv
            )

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
        return forces

    def get_potential_energy(self):
         # V(improper) = Kpsi(psi - psi0)**2
        potential_energy = 0
        for improper_info in self._improper_info:
            id1, id2, id3, id4, k, psi0 = improper_info
            psi = get_pbc_dihedral(
                self._parent_ensemble.state.positions[id1, :], 
                self._parent_ensemble.state.positions[id2, :],
                self._parent_ensemble.state.positions[id3, :], 
                self._parent_ensemble.state.positions[id4, :],
                self._parent_ensemble.state.pbc_matrix,
                self._parent_ensemble.state.pbc_inv,
                is_angular=False
            )
            potential_energy += k * (psi - psi0)**2
        return potential_energy

    @property
    def num_impropers(self):
        return self._num_impropers