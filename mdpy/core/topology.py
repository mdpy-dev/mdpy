#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : topology.py
created time : 2021/09/28
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np
from . import Particle
from .. import env
from ..error import *
from ..unit import *

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
        self._charges = []
        self._bonded_particles = []
        self._scaling_pactiles = []
        
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
                    'Matrix id %d beyonds the range of particles contain in toplogy, can not be added as part of topology connection' 
                    %matrix_id
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
        max_num_bonded_particles = 0
        max_num_scaling_particles = 0
        for index, particle in enumerate(self._particles):
            if particle.num_bonded_particles > max_num_bonded_particles:
                max_num_bonded_particles = particle.num_bonded_particles
            if particle.num_scaling_particles > max_num_scaling_particles:
                max_num_scaling_particles = particle.num_scaling_particles
            self._masses[index, 0] = particle.mass
            self._charges[index, 0] = particle.charge
        self._bonded_particles = np.ones([
            self._num_particles, max_num_bonded_particles
        ], dtype=env.NUMPY_INT) * -1
        self._scaling_particles = np.ones([
            self._num_particles, max_num_bonded_particles
        ], dtype=env.NUMPY_INT) * -1
        for index, particle in enumerate(self._particles):
            self._bonded_particles[index, :particle.num_bonded_particles] = particle.bonded_particles
            self._scaling_particles[index, :particle.num_scaling_particles] = particle.scaling_particles
        self._is_joined = True

    def split(self):
        self._masses = []
        self._charges = []
        self._bonded_particles = []
        self._scaling_pactiles = []
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
        self._particles[p1].add_bonded_particle(p2)
        self._particles[p2].add_bonded_particle(p1)
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
            self._particles[p1].del_bonded_particle(p2)
            self._particles[p2].del_bonded_particle(p1)
            self._num_bonds -= 1
        elif bond_replica in self._bonds:
            self._bonds.remove(bond_replica)
            self._particles[p1].del_bonded_particle(p2)
            self._particles[p2].del_bonded_particle(p1)
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
        self._particles[p1].add_bonded_particle(p3)
        self._particles[p3].add_bonded_particle(p1)
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
            self._particles[p1].del_bonded_particle(p3)
            self._particles[p3].del_bonded_particle(p1)
            self._num_angles -= 1
        elif angle_replica in self._angles:
            self._angles.remove(angle_replica)
            self._particles[p1].del_bonded_particle(p3)
            self._particles[p3].del_bonded_particle(p1)
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
        self._particles[p1].add_scaling_particle(p4, scaling_factor)
        self._particles[p4].add_scaling_particle(p1, scaling_factor)
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
            self._particles[p1].del_scaling_particle(p4)
            self._particles[p4].del_scaling_particle(p1)
            self._num_dihedrals -= 1
        elif dihedral_replica in self._dihedrals:
            self._dihedrals.remove(dihedral_replica)
            self._particles[p1].del_scaling_particle(p4)
            self._particles[p4].del_scaling_particle(p1)
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
    def masses(self):
        return self._masses

    @property
    def charges(self):
        return self._charges

    @property
    def bonded_particles(self):
        return self._bonded_particles

    @property
    def scaling_particles(self):
        return self._scaling_particles

    @property
    def particles(self) -> list[Particle]:
        return self._particles
    
    @property
    def num_particles(self):
        return self._num_particles

    @property
    def bonds(self):
        return self._bonds
    
    @property
    def num_bonds(self):
        return self._num_bonds

    @property
    def angles(self):
        return self._angles
    
    @property
    def num_angles(self):
        return self._num_angles

    @property
    def dihedrals(self):
        return self._dihedrals
    
    @property
    def num_dihedrals(self):
        return self._num_dihedrals

    @property
    def impropers(self):
        return self._impropers
    
    @property
    def num_impropers(self):
        return self._num_impropers

    @property
    def is_joined(self):
        return self._is_joined