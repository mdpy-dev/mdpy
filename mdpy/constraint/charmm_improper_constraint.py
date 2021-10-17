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
        self._improper_type, self._improper_matrix_id, self._improper_info = [], [], []
        self._num_impropers = 0

    def bind_ensemble(self, ensemble: Ensemble):
        ensemble.add_constraints(self)
        self._improper_type, self._improper_matrix_id, self._improper_info = [], [], []
        self._num_impropers = 0
        for improper in self._parent_ensemble.topology.impropers:
            self._improper_type.append('%s-%s-%s-%s' %(
                self._parent_ensemble.topology.particles[improper[0]].particle_name,
                self._parent_ensemble.topology.particles[improper[1]].particle_name,
                self._parent_ensemble.topology.particles[improper[2]].particle_name,
                self._parent_ensemble.topology.particles[improper[3]].particle_name
            ))
            self._improper_matrix_id.append([
                self._parent_ensemble.topology.particles[improper[0]].matrix_id,
                self._parent_ensemble.topology.particles[improper[1]].matrix_id,
                self._parent_ensemble.topology.particles[improper[2]].matrix_id,
                self._parent_ensemble.topology.particles[improper[3]].matrix_id
            ])
            self._num_impropers += 1

        for index, improper in enumerate(self._improper_type):
            self._improper_info.append(self._improper_matrix_id[index] + self._params[improper])

    def get_forces(self):
        self._check_bound_state()
        # V(improper) = Kpsi(psi - psi0)**2
        forces = np.zeros([self._parent_ensemble.topology.num_particles, SPATIAL_DIM])
        for improper_info in self._improper_info:
            id1, id2, id3, id4, k, psi0 = improper_info
            psi = get_dihedral(
                self._parent_ensemble.positions[id1, :], 
                self._parent_ensemble.positions[id2, :],
                self._parent_ensemble.positions[id3, :], 
                self._parent_ensemble.positions[id4, :],
                is_angular=False
            )
            force_val = 2 * k * (psi - psi0)

            vab = self._parent_ensemble.positions[id2, :] -self._parent_ensemble.positions[id1, :]
            lab = np.linalg.norm(vab)
            vbc = self._parent_ensemble.positions[id3, :] -self._parent_ensemble.positions[id2, :]
            lbc = np.linalg.norm(vbc)
            voc, loc = vbc / 2, lbc / 2
            vcd = self._parent_ensemble.positions[id4, :] -self._parent_ensemble.positions[id3, :]
            lcd = np.linalg.norm(vcd)
            theta_abc = get_angle(
                self._parent_ensemble.positions[id1, :], 
                self._parent_ensemble.positions[id2, :],
                self._parent_ensemble.positions[id3, :]
            )
            theta_bcd = get_angle(
                self._parent_ensemble.positions[id2, :], 
                self._parent_ensemble.positions[id3, :],
                self._parent_ensemble.positions[id4, :]
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
            psi = get_dihedral(
                self._parent_ensemble.positions[id1, :], 
                self._parent_ensemble.positions[id2, :],
                self._parent_ensemble.positions[id3, :], 
                self._parent_ensemble.positions[id4, :],
                is_angular=False
            )
            potential_energy += k * (psi - psi0)**2
        return potential_energy

    @property
    def num_impropers(self):
        return self._num_impropers