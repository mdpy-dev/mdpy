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
from . import Constraint
from .. import SPATIAL_DIM
from ..ensemble import Ensemble
from ..math import *
from ..unit import *

class CharmmDihedralConstraint(Constraint):
    def __init__(self, params, force_id: int = 0, force_group: int = 0) -> None:
        super().__init__(params, force_id=force_id, force_group=force_group)
        self._dihedral_info = []
        self._num_dihedrals = 0

    def __repr__(self) -> str:
        return '<mdpy.constraint.CharmmDihedralConstraint object>'

    __str__ = __repr__

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._force_id = ensemble.constraints.index(self)
        self._dihedral_info = []
        self._num_dihedrals = 0
        for dihedral in self._parent_ensemble.topology.dihedrals:
            dihedral_type = '%s-%s-%s-%s' %(
                self._parent_ensemble.topology.particles[dihedral[0]].particle_name,
                self._parent_ensemble.topology.particles[dihedral[1]].particle_name,
                self._parent_ensemble.topology.particles[dihedral[2]].particle_name,
                self._parent_ensemble.topology.particles[dihedral[3]].particle_name
            )
            matrix_id = [
                self._parent_ensemble.topology.particles[dihedral[0]].matrix_id,
                self._parent_ensemble.topology.particles[dihedral[1]].matrix_id,
                self._parent_ensemble.topology.particles[dihedral[2]].matrix_id,
                self._parent_ensemble.topology.particles[dihedral[3]].matrix_id
            ]
            self._dihedral_info.append(matrix_id + self._params[dihedral_type])
            self._num_dihedrals += 1

    def update(self):
        i = 0
        self._check_bound_state()
        # V(dihedral) = Kchi(1 + cos(n(chi) - delta))
        self._forces = np.zeros([self._parent_ensemble.topology.num_particles, SPATIAL_DIM])
        self._potential_energy = 0
        for dihedral_info in self._dihedral_info:
            id1, id2, id3, id4= dihedral_info[:4]
            theta = get_pbc_dihedral(
                self._parent_ensemble.state.positions[id1, :], 
                self._parent_ensemble.state.positions[id2, :],
                self._parent_ensemble.state.positions[id3, :], 
                self._parent_ensemble.state.positions[id4, :],
                *self._parent_ensemble.state.pbc_info
            )
            vab = unwrap_vec(
                self._parent_ensemble.state.positions[id2, :] - 
                self._parent_ensemble.state.positions[id1, :],
                *self._parent_ensemble.state.pbc_info
            )
            lab = np.linalg.norm(vab)
            vbc = unwrap_vec(
                self._parent_ensemble.state.positions[id3, :] - 
                self._parent_ensemble.state.positions[id2, :],
                *self._parent_ensemble.state.pbc_info
            )
            lbc = np.linalg.norm(vbc)
            voc, loc = vbc / 2, lbc / 2
            vcd = unwrap_vec(
                self._parent_ensemble.state.positions[id4, :] - 
                self._parent_ensemble.state.positions[id3, :],
                *self._parent_ensemble.state.pbc_info
            )
            lcd = np.linalg.norm(vcd)
            theta_abc = get_pbc_angle(
                self._parent_ensemble.state.positions[id1, :], 
                self._parent_ensemble.state.positions[id2, :],
                self._parent_ensemble.state.positions[id3, :],
                *self._parent_ensemble.state.pbc_info
            )
            theta_bcd = get_pbc_angle(
                self._parent_ensemble.state.positions[id2, :], 
                self._parent_ensemble.state.positions[id3, :],
                self._parent_ensemble.state.positions[id4, :],
                *self._parent_ensemble.state.pbc_info
            )
            for param in dihedral_info[4:]:
                k, n, delta = param
                # Forces
                force_val = - k * (1 - n * np.sin(n*theta - delta))
                force_a = force_val / (lab * np.sin(theta_abc)) * get_unit_vec(np.cross(-vab, vbc))
                force_d = force_val / (lcd * np.sin(theta_bcd)) * get_unit_vec(np.cross(vcd, -vbc))
                force_c =  np.cross(
                    - (np.cross(voc, force_d) + np.cross(vcd, force_d) / 2 + np.cross(-vab, force_a) / 2),
                    voc
                ) / loc**2
                force_b = - (force_a + force_c + force_d)
                self._forces[id1, :] += force_a
                self._forces[id2, :] += force_b
                self._forces[id3, :] += force_c
                self._forces[id4, :] += force_d
                # Potential energy
                self._potential_energy += k * (1 + np.cos(n*theta - delta))

    @property
    def num_dihedrals(self):
        return self._num_dihedrals