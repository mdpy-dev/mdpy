#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : topology.py
created time : 2021/09/28
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import numpy as np
import cupy as cp
from mdpy import env
from mdpy.core import MAX_NUM_EXCLUDED_PARTICLES, MAX_NUM_SCALED_PARTICLES
from mdpy.core import Particle
from mdpy.environment import CUPY_FLOAT, CUPY_INT
from mdpy.error import *
from mdpy.unit import *

class Topology:
    def __init__(self) -> None:
        self._particles = []
        self._num_particles = 0
        self._bonds = []
        self._num_bonds = 0
        self._angles = []
        self._num_angles = 0
        self._dihedrals = []
        self._num_dihedrals = 0
        self._impropers = []
        self._num_impropers = 0
        self._is_joined = False
        self._masses = []
        self._device_masses = []
        self._charges = []
        self._device_charges = []
        self._device_sorted_charges = []
        self._excluded_particles = []
        self._device_excluded_particles = []
        self._device_sorted_excluded_particles = []
        self._scaled_particles = []
        self._device_scaled_particles = []
        self._device_sorted_scaled_particles = []

    def __repr__(self) -> str:
        return '<mdpy.core.Toplogy object: %d particles at %x>' %(self._num_particles, id(self))

    def __str__(self) -> str:
        return(
            'Toplogy with %d particles, %d bonds, %d angles, %d dihedrals, %d impropers'
            %(self._num_particles, self._num_bonds, self._num_angles, self._num_dihedrals, self._num_impropers)
        )

    def _check_matrix_ids(self, *matrix_ids):
        for index, matrix_id in enumerate(matrix_ids):
            if matrix_id >= self._num_particles:
                raise ParticleConflictError(
                    'Matrix id %d beyonds the range of particles contain in toplogy, ' \
                    'can not be added as part of topology connection' %matrix_id
                )
            if matrix_id in matrix_ids[index+1:]:
                raise ParticleConflictError('Particle appears twice in a topology connection')

    def _check_joined(self):
        if self._is_joined:
            raise  ModifyJoinedTopologyError(
                '%s has been joined. No change can be made.' %self
            )

    def join(self):
        self._masses = np.zeros([self._num_particles, 1], dtype=env.NUMPY_FLOAT)
        self._charges = np.zeros([self._num_particles, 1], dtype=env.NUMPY_FLOAT)
        for index, particle in enumerate(self._particles):
            self._masses[index, 0] = particle.mass
            self._charges[index, 0] = particle.charge
        self._excluded_particles = np.ones([
            self._num_particles, MAX_NUM_EXCLUDED_PARTICLES
        ], dtype=env.NUMPY_INT) * -1
        self._scaled_particles = np.ones([
            self._num_particles, MAX_NUM_SCALED_PARTICLES
        ], dtype=env.NUMPY_INT) * -1
        for index, particle in enumerate(self._particles):
            self._excluded_particles[index, :particle.num_excluded_particles] = particle.excluded_particles
            self._scaled_particles[index, :particle.num_scaled_particles] = particle.scaled_particles
        self._device_masses = cp.array(self._masses, CUPY_FLOAT)
        self._device_charges = cp.array(self._charges, CUPY_FLOAT)
        self._device_excluded_particles = cp.array(self._excluded_particles, CUPY_INT)
        self._device_scaled_particles = cp.array(self._scaled_particles, CUPY_INT)
        self._is_joined = True

    def split(self):
        self._masses = []
        self._charges = []
        self._excluded_particles = []
        self._scaled_particles = []
        self._is_joined = False

    def add_particles(self, particles):
        self._check_joined()
        for particle in particles:
            if not isinstance(particle, Particle):
                raise TypeError('mdpy.core.Particle type is excepted, while %s provided' %type(particle))
            # if particle in self._particles:
            #     raise ParticleConflictError('Particle %s is added twice to Toplogy instance' %particle)
            # particle.change_particle_id(self._num_particles) # Deprecated because this work should be done by modeling software
            particle.change_matrix_id(self._num_particles)
            self._particles.append(particle)
            self._num_particles += 1

    def del_particles(self, particles):
        self._check_joined()
        particle_list, bond_list, angle_list, dihedral_list, improper_list = [], [], [], [], []
        for index, particle in enumerate(particles):
            if particle in self._particles:
                particle_list.append(index)
                bond_list.extend([index for index, bond in enumerate(self._bonds) if particle.matrix_id in bond])
                angle_list.extend([index for index, angle in enumerate(self._angles) if particle.matrix_id in angle])
                dihedral_list.extend([index for index, dihedral in enumerate(self._dihedrals) if particle.matrix_id in dihedral])
                improper_list.extend([index for index, improper in enumerate(self._impropers) if particle.matrix_id in improper])
        self._particles = [self._particles[i] for i in set(range(self._num_particles))^set(particle_list)]
        self._num_particles = len(self._particles)
        self._bonds = [self._bonds[i] for i in set(range(self._num_bonds))^set(bond_list)]
        self._num_bonds = len(self._bonds)
        self._angles = [self._angles[i] for i in set(range(self._num_angles))^set(angle_list)]
        self._num_angles = len(self._angles)
        self._dihedrals = [self._dihedrals[i] for i in set(range(self._num_dihedrals))^set(dihedral_list)]
        self._num_dihedrals = len(self._dihedrals)
        self._impropers = [self._impropers[i] for i in set(range(self._num_impropers))^set(improper_list)]
        self._num_impropers = len(self._impropers)

    def add_bond(self, bond):
        self._check_joined()
        num_particles = len(bond)
        if num_particles != 2:
            raise GeomtryDimError('Bond should be a matrix id list of 2 Particles, instead of %d' %num_particles)
        p1, p2 = bond
        self._check_matrix_ids(p1, p2)
        # bond_replica = [p2, p1]
        # if not bond in self._bonds and not bond_replica in self._bonds:
        self._bonds.append(bond)
        self._particles[p1].add_excluded_particle(p2)
        self._particles[p2].add_excluded_particle(p1)
        self._num_bonds += 1

    def del_bond(self, bond):
        self._check_joined()
        num_particles = len(bond)
        if num_particles != 2:
            raise GeomtryDimError('Bond should be a matrix id list of 2 Particles, instead of %d' %num_particles)
        p1, p2 = bond
        self._check_matrix_ids(p1, p2)
        bond_replica = [p2, p1]
        if bond in self._bonds:
            self._bonds.remove(bond)
            self._particles[p1].del_excluded_particle(p2)
            self._particles[p2].del_excluded_particle(p1)
            self._num_bonds -= 1
        elif bond_replica in self._bonds:
            self._bonds.remove(bond_replica)
            self._particles[p1].del_excluded_particle(p2)
            self._particles[p2].del_excluded_particle(p1)
            self._num_bonds -= 1

    def add_angle(self, angle):
        self._check_joined()
        num_particles = len(angle)
        if num_particles != 3:
            raise GeomtryDimError('Angle should be a matrix id list of 3 Particles, instead of %d' %num_particles)
        p1, p2, p3 = angle
        self._check_matrix_ids(p1, p2, p3)
        # angle_replica = [p3, p2, p1]
        # if not angle in self._angles and not angle_replica in self._angles:
        self._angles.append(angle)
        self._particles[p1].add_excluded_particle(p3)
        self._particles[p3].add_excluded_particle(p1)
        self._num_angles += 1

    def del_angle(self, angle):
        self._check_joined()
        num_particles = len(angle)
        if num_particles != 3:
            raise GeomtryDimError('Angle should be a matrix id list of 3 Particles, instead of %d' %num_particles)
        p1, p2, p3 = angle
        self._check_matrix_ids(p1, p2, p3)
        angle_replica = [p3, p2, p1]
        if angle in self._angles:
            self._angles.remove(angle)
            self._particles[p1].del_excluded_particle(p3)
            self._particles[p3].del_excluded_particle(p1)
            self._num_angles -= 1
        elif angle_replica in self._angles:
            self._angles.remove(angle_replica)
            self._particles[p1].del_excluded_particle(p3)
            self._particles[p3].del_excluded_particle(p1)
            self._num_angles -= 1

    def add_dihedral(self, dihedral, scaling_factor=1):
        self._check_joined()
        num_particles = len(dihedral)
        if num_particles != 4:
            raise GeomtryDimError('Dihedral should be a matrix id list of 4 Particles, instead of %d' %num_particles)
        p1, p2, p3, p4 = dihedral
        self._check_matrix_ids(p1, p2, p3, p4)
        # dihedral_replica = [p4, p3, p2, p1]
        # if not dihedral in self._dihedrals and not dihedral_replica in self._dihedrals:
        self._dihedrals.append(dihedral)
        self._particles[p1].add_scaled_particle(p4, scaling_factor)
        self._particles[p4].add_scaled_particle(p1, scaling_factor)
        self._num_dihedrals += 1

    def del_dihedral(self, dihedral):
        self._check_joined()
        num_particles = len(dihedral)
        if num_particles != 4:
            raise GeomtryDimError('Dihedral should be a matrix id list of 4 Particles, instead of %d' %num_particles)
        p1, p2, p3, p4 = dihedral
        self._check_matrix_ids(p1, p2, p3, p4)
        dihedral_replica = [p4, p3, p2, p1]
        if dihedral in self._dihedrals:
            self._dihedrals.remove(dihedral)
            self._particles[p1].del_scaled_particle(p4)
            self._particles[p4].del_scaled_particle(p1)
            self._num_dihedrals -= 1
        elif dihedral_replica in self._dihedrals:
            self._dihedrals.remove(dihedral_replica)
            self._particles[p1].del_scaled_particle(p4)
            self._particles[p4].del_scaled_particle(p1)
            self._num_dihedrals -= 1

    def add_improper(self, improper):
        self._check_joined()
        num_particles = len(improper)
        if num_particles != 4:
            raise GeomtryDimError('Improper should be a matrix id list of 4 Particles, instead of %d' %num_particles)
        p1, p2, p3, p4 = improper
        self._check_matrix_ids(p1, p2, p3, p4)
        # if not improper in self._impropers:
        self._impropers.append(improper)
        self._num_impropers += 1

    def del_improper(self, improper):
        self._check_joined()
        num_particles = len(improper)
        if num_particles != 4:
            raise GeomtryDimError('Improper should be a matrix id list of 4 Particles, instead of %d' %num_particles)
        p1, p2, p3, p4 = improper
        self._check_matrix_ids(p1, p2, p3, p4)
        if improper in self._impropers:
            self._impropers.remove(improper)
            self._num_impropers -= 1

    @property
    def masses(self) -> np.ndarray:
        return self._masses

    @property
    def device_masses(self) -> cp.ndarray:
        return self._device_masses

    @property
    def charges(self) -> np.ndarray:
        return self._charges

    @property
    def device_charges(self) -> cp.ndarray:
        return self._device_charges

    @property
    def device_sorted_charges(self) -> cp.ndarray:
        return self._device_sorted_charges

    @device_sorted_charges.setter
    def device_sorted_charges(self, sorted_charges: cp.ndarray):
        self._device_sorted_charges = sorted_charges

    @property
    def excluded_particles(self) -> np.ndarray:
        return self._excluded_particles

    @property
    def device_excluded_particles(self) -> cp.ndarray:
        return self._device_excluded_particles

    @property
    def device_sorted_excluded_particles(self) -> cp.ndarray:
        return self._device_sorted_excluded_particles

    @device_sorted_excluded_particles.setter
    def device_sorted_excluded_particles(self, sorted_excluded_particles: cp.ndarray):
        self._device_sorted_excluded_particles = sorted_excluded_particles

    @property
    def scaled_particles(self) -> np.ndarray:
        return self._scaled_particles

    @property
    def device_scaled_particles(self) -> cp.ndarray:
        return self._device_scaled_particles

    @property
    def device_sorted_scaled_particles(self) -> cp.ndarray:
        return self._device_sorted_scaled_particles

    @device_sorted_scaled_particles.setter
    def device_sorted_scaled_particles(self, sorted_scaled_particles: cp.ndarray):
        self._device_sorted_scaled_particles = sorted_scaled_particles

    @property
    def particles(self) -> list[Particle]:
        return self._particles

    @property
    def num_particles(self) -> int:
        return self._num_particles

    @property
    def bonds(self) -> list:
        return self._bonds

    @property
    def num_bonds(self) -> int:
        return self._num_bonds

    @property
    def angles(self) -> list:
        return self._angles

    @property
    def num_angles(self) -> int:
        return self._num_angles

    @property
    def dihedrals(self) -> list:
        return self._dihedrals

    @property
    def num_dihedrals(self) -> int:
        return self._num_dihedrals

    @property
    def impropers(self) -> list:
        return self._impropers

    @property
    def num_impropers(self) -> int:
        return self._num_impropers

    @property
    def is_joined(self) -> bool:
        return self._is_joined