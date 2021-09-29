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

from . import Particle
from ..error import *

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
        
    def __repr__(self) -> str:
        return '<Toplogy object: %d particles at %x>' %(self._num_particles, id(self))
    
    def __str__(self) -> str:
        return(
            'Toplogy with %d particles, %d bonds, %d angles, %d dihedrals, %d impropers'
            %(self._num_particles, self._num_bonds, self._num_angles, self._num_dihedrals, self._num_impropers)
        )
        
    def _add_particle(self, particle: Particle):
        if particle in self._particles:
            raise ParticleConflictError('Particle %s is added twice to Toplogy instance' %particle)
        # particle.change_particle_id(self._num_particles) # Deprecated because this work should be done by modeling software
        self._particles.append(particle)
        self._num_particles += 1
        
    def add_particles(self, *p_list):
        for p in p_list:
            self._add_particle(p)
    
    def select_particles(self, keywords):
        ''' particle_id=1 and molecule_type=ASN or particle_type=CB
        '''
        target_partiles = self._particles.copy()
        selected_particles = []
        or_selections = [i.strip() for i in keywords.split('or')]
        for or_selection in or_selections:
            and_particles = []
            and_selections = [i.strip() for i in or_selection.split('and')]
            key_val_pairs = []
            for and_selection in and_selections:
                key_val_pair = [i.strip() for i in and_selection.split('=')]
                if 'not' in key_val_pair[0]:
                    key_val_pair = ['not', key_val_pair[0].split('not')[-1].strip(), key_val_pair[-1]]
                try:
                    key_val_pair[-1] = float(key_val_pair[-1][-1]) # Turn str to int if it is possible
                except:
                    pass
                if not 'all' in key_val_pair[0]:
                    key_val_pairs.append(key_val_pair) # Don't select atom to delete in the following section
                
            # Select particle doesn't match condition to delete
            and_particles = target_partiles.copy()
            for particle in target_partiles:
                is_deleted = False # First assume will not be deleted, once condition is not matched turn to True
                for key_val in key_val_pairs:
                    if not 'not' in key_val:
                        if particle.__getattribute__(key_val[0]) != key_val[-1]:
                            is_deleted = True
                            break
                    elif 'not' in key_val:
                        if particle.__getattribute__(key_val[1]) == key_val[-1]:
                            is_deleted = True
                            break
                if is_deleted == True:
                    and_particles.remove(particle)
            [target_partiles.remove(particle) for particle in and_particles] # Remove selected particles, reduce computation time for next or condition
            selected_particles.extend(and_particles)
        return selected_particles

    def del_particles(self, particles):
        for particle in particles:
            if particle in self._particles:
                self._particles.remove(particle)
                [self.del_bond(bond) for bond in self._bonds if particle in bond]
                [self.del_angle(angle) for angle in self._angles if particle in angle]
                [self.del_dihedral(dihedral) for dihedral in self._dihedrals if particle in dihedral]
                [self.del_improper(improper) for improper in self._impropers if particle in improper]
                self._num_particles -= 1

    def _check_particles(self, *particles, is_patch_mode=True):
        for i, j in enumerate(particles):
            if j in particles[i+1:]:
                raise ParticleConflictError('Particle appears twice in a topology connection')
        if is_patch_mode:
            for p in particles:
                if not p in self._particles:
                    self.add_particles(p)

    def add_bond(self, bond):
        num_particles = len(bond)
        if num_particles != 2:
            raise GeomtryDimError('Bond should be a list of 2 Particles, instead of %d' %num_particles)
        p1, p2 = bond
        self._check_particles(p1, p2)
        bond_replica = [p2, p1]
        if not bond in self._bonds and not bond_replica in self._bonds:
            self._bonds.append(bond)
            self._num_bonds += 1
        
    def del_bond(self, bond):
        num_particles = len(bond)
        if num_particles != 2:
            raise GeomtryDimError('Bond should be a list of 2 Particles, instead of %d' %num_particles)
        p1, p2 = bond
        self._check_particles(p1, p2, is_patch_mode=False)
        bond_replica = [p2, p1]
        if bond in self._bonds:
            self._bonds.remove(bond)
            self._num_bonds -= 1
        elif bond_replica in self._bonds:
            self._bonds.remove(bond_replica)
            self._num_bonds -= 1

    def add_angle(self, angle):
        num_particles = len(angle)
        if num_particles != 3:
            raise GeomtryDimError('Angle should be a list of 3 Particles, instead of %d' %num_particles)
        p1, p2, p3 = angle
        self._check_particles(p1, p2, p3)
        angle_replica = [p3, p2, p1]
        if not angle in self._angles and not angle_replica in self._angles:
            self._angles.append(angle)  
            self._num_angles += 1
        
    def del_angle(self, angle):
        num_particles = len(angle)
        if num_particles != 3:
            raise GeomtryDimError('Angle should be a list of 3 Particles, instead of %d' %num_particles)
        p1, p2, p3 = angle
        self._check_particles(p1, p2, p3, is_patch_mode=False)
        angle_replica = [p3, p2, p1]
        if angle in self._angles:
            self._angles.remove(angle)
            self._num_angles -= 1
        elif angle_replica in self._angles:
            self._angles.remove(angle_replica)
            self._num_angles -= 1
        
    def add_dihedral(self, dihedral):
        num_particles = len(dihedral)
        if num_particles != 4:
            raise GeomtryDimError('Dihedral should be a list of 4 Particles, instead of %d' %num_particles)
        p1, p2, p3, p4 = dihedral
        self._check_particles(p1, p2, p3, p4)
        dihedral_replica = [p4, p3, p2, p1]
        if not dihedral in self._dihedrals and not dihedral_replica in self._dihedrals:
            self._dihedrals.append(dihedral)
            self._num_dihedrals += 1
        
    def del_dihedral(self, dihedral):
        num_particles = len(dihedral)
        if num_particles != 4:
            raise GeomtryDimError('Dihedral should be a list of 4 Particles, instead of %d' %num_particles)
        p1, p2, p3, p4 = dihedral
        self._check_particles(p1, p2, p3, p4, is_patch_mode=False)
        dihedral_replica = [p4, p3, p2, p1]
        if dihedral in self._dihedrals:
            self._dihedrals.remove(dihedral)
            self._num_dihedrals -= 1
        elif dihedral_replica in self._dihedrals:
            self._dihedrals.remove(dihedral_replica)
            self._num_dihedrals -= 1

    def add_improper(self, improper):
        num_particles = len(improper)
        if num_particles != 4:
            raise GeomtryDimError('Improper should be a list of 4 Particles, instead of %d' %num_particles)
        p1, p2, p3, p4 = improper
        self._check_particles(p1, p2, p3, p4)
        if not improper in self._impropers:
            self._impropers.append(improper)
            self._num_impropers += 1
        
    def del_improper(self, improper):
        num_particles = len(improper)
        if num_particles != 4:
            raise GeomtryDimError('Improper should be a list of 4 Particles, instead of %d' %num_particles)
        p1, p2, p3, p4 = improper
        self._check_particles(p1, p2, p3, p4, is_patch_mode=False)
        if improper in self._impropers:
            self._impropers.remove(improper)
            self._num_impropers -= 1

    @property
    def particles(self):
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