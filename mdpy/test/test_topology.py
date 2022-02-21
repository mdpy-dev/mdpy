#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
file: test_topology.py
created time : 2021/09/29
author : Zhenyu Wei 
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest
import numpy as np
from ..core import Particle, Topology
from ..error import *
from ..unit import *

class TestTopology:
    def setup(self):
        self.topology = Topology()
    
    def teardown(self):
        pass
    
    def test_attributes(self):
        assert self.topology.particles == []
        assert self.topology.num_particles == 0
        assert self.topology.bonds == []
        assert self.topology.num_bonds == 0
        assert self.topology.angles == []
        assert self.topology.num_angles == 0
        assert self.topology.dihedrals == []
        assert self.topology.num_dihedrals == 0
        assert self.topology.impropers == []
        assert self.topology.num_impropers == 0
        assert self.topology.charges == []
        assert self.topology.masses == []

    def test_particle_properties(self):
        assert self.topology.charges == []
        assert self.topology.masses == []
        p1 = Particle(mass=1, charge=2)
        p2 = Particle(mass=2, charge=1)
        p3 = Particle(mass=3, charge=0)
        self.topology.add_particles([p1, p2, p3])
        self.topology.join()
        assert self.topology.charges[0] == 2
        assert self.topology.charges[-1] == 0
        assert self.topology.masses[1] == 2

        assert self.topology.masses.shape[0] == 3
        assert self.topology.masses.shape[1] == 1
        assert self.topology.charges.shape[0] == 3
        assert self.topology.charges.shape[1] == 1

    def test_add_particles(self):
        p1 = Particle(0, 'CA')
        p2 = Particle(0, 'CB')
        p3 = Particle(0, 'N')
        self.topology.add_particles([p1, p2, p3])
        assert self.topology.num_particles == 3
        assert self.topology.particles[0].particle_id == 0
        assert self.topology.particles[1].particle_id == 0
        assert self.topology.particles[0].matrix_id == 0
        assert self.topology.particles[1].matrix_id == 1

        # with pytest.raises(ParticleConflictError):
        #     self.topology.add_particles([p1])

    def test_del_particles(self):
        p1 = Particle(0, 'CA', molecule_id=0, molecule_type='ASN')
        p2 = Particle(1, 'CB', molecule_id=0, molecule_type='ASN')
        p3 = Particle(2, 'N', molecule_id=0, molecule_type='ASN')
        p4 = Particle(3, 'H', molecule_id=1, molecule_type='HOH')
        self.topology.add_particles([p1, p2, p3, p4])
        self.topology.del_particles([p1, p2, p3])
        assert self.topology.num_particles == 1

        self.topology.del_particles([p4])
        assert self.topology.num_particles == 0

        self.topology.add_particles([p1, p2, p3, p4])
        self.topology.add_bond([0, 1])
        self.topology.add_angle([0, 1, 2])
        self.topology.add_dihedral([0, 1, 2, 3])
        assert self.topology.num_particles == 4
        assert self.topology.num_bonds == 1
        assert self.topology.num_angles == 1
        assert self.topology.num_dihedrals == 1
        self.topology.del_particles([p4])
        assert self.topology.num_particles == 3
        assert self.topology.num_bonds == 1
        assert self.topology.num_angles == 1
        assert self.topology.num_dihedrals == 0

    def test_add_bond(self):
        p1 = Particle(0, 'CA', molecule_id=0, molecule_type='ASN')
        p2 = Particle(1, 'CB', molecule_id=0, molecule_type='ASN')
        p3 = Particle(2, 'N', molecule_id=0, molecule_type='ASN')
        self.topology.add_particles([p1, p2, p3])
        self.topology.add_bond([0, 1])
        assert p1.num_bonded_particles == 1
        assert self.topology.num_bonds == 1

        self.topology.add_bond([1, 2])
        assert p2.num_bonded_particles == 2
        assert self.topology.num_bonds == 2

        with pytest.raises(GeomtryDimError):
            self.topology.add_bond([0, 1, 2])

        with pytest.raises(ParticleConflictError):
            self.topology.add_bond([1, 1])
        
        with pytest.raises(ParticleConflictError):
            self.topology.add_bond([1, 4])

    def test_del_bond(self):
        p1 = Particle(0, 'CA', molecule_id=0, molecule_type='ASN')
        p2 = Particle(1, 'CB', molecule_id=0, molecule_type='ASN')
        p3 = Particle(2, 'N', molecule_id=0, molecule_type='ASN')
        self.topology.add_particles([p1, p2, p3])
        self.topology.add_bond([0, 1])
        assert p1.num_bonded_particles == 1
        assert self.topology.num_bonds == 1
        self.topology.del_bond([0, 1])
        assert p2.num_bonded_particles == 0
        assert self.topology.num_bonds == 0

        self.topology.add_bond([0, 1])
        assert self.topology.num_bonds == 1
        self.topology.del_bond([1, 0])
        assert self.topology.num_bonds == 0

        self.topology.add_bond([1, 2])
        assert self.topology.num_bonds == 1
        self.topology.del_bond([0, 1])
        assert self.topology.num_bonds == 1

        with pytest.raises(GeomtryDimError):
            self.topology.del_bond([1, 2, 3])

        with pytest.raises(ParticleConflictError):
            self.topology.del_bond([1, 1])

        with pytest.raises(ParticleConflictError):
            self.topology.del_bond([1, 9])

    def test_add_angle(self):
        p1 = Particle(0, 'CA', molecule_id=0, molecule_type='ASN')
        p2 = Particle(1, 'CB', molecule_id=0, molecule_type='ASN')
        p3 = Particle(2, 'N', molecule_id=0, molecule_type='ASN')
        self.topology.add_particles([p1, p2, p3])
        self.topology.add_angle([0, 1, 2])
        assert self.topology.num_angles == 1

        self.topology.add_angle([2, 0, 1])
        assert self.topology.num_angles == 2

        with pytest.raises(GeomtryDimError):
            self.topology.add_angle([1, 1, 2, 3])

        with pytest.raises(ParticleConflictError):
            self.topology.add_angle([2, 1, 1])

        with pytest.raises(ParticleConflictError):
            self.topology.add_angle([2, 1, 9])

    def test_del_angle(self):
        p1 = Particle(0, 'CA', molecule_id=0, molecule_type='ASN')
        p2 = Particle(1, 'CB', molecule_id=0, molecule_type='ASN')
        p3 = Particle(2, 'N', molecule_id=0, molecule_type='ASN')
        self.topology.add_particles([p1, p2, p3])
        self.topology.add_angle([0, 1, 2])
        assert self.topology.num_angles == 1
        self.topology.del_angle([0, 1, 2])
        assert self.topology.num_angles == 0

        self.topology.add_angle([0, 1, 2])
        assert self.topology.num_angles == 1
        self.topology.del_angle([2, 1, 0])
        assert self.topology.num_angles == 0

        self.topology.add_angle([0, 1, 2])
        assert self.topology.num_angles == 1
        self.topology.del_angle([0, 2, 1])
        assert self.topology.num_angles == 1

        with pytest.raises(GeomtryDimError):
            self.topology.del_angle([1, 2, 3, 4])

        with pytest.raises(ParticleConflictError):
            self.topology.del_angle([1, 2, 1])
        
        with pytest.raises(ParticleConflictError):
            self.topology.del_angle([1, 2, 9])

    def test_add_dihedral(self):
        p1 = Particle(0, 'CA', molecule_id=0, molecule_type='ASN')
        p2 = Particle(1, 'CB', molecule_id=0, molecule_type='ASN')
        p3 = Particle(2, 'N', molecule_id=0, molecule_type='ASN')
        p4 = Particle(3, 'H', molecule_id=0, molecule_type='ASN')
        self.topology.add_particles([p1, p2, p3, p4])
        self.topology.add_dihedral([0, 1, 2, 3])
        assert self.topology.particles[0].scaling_particles[0] == 3
        assert self.topology.particles[3].scaling_factors[0] == 1
        assert self.topology.num_dihedrals == 1

        self.topology.add_dihedral([3, 2, 1, 0])
        assert self.topology.particles[0].num_scaling_particles == 1
        assert self.topology.num_dihedrals == 2

        self.topology.add_dihedral([0, 2, 1, 3])
        assert self.topology.num_dihedrals == 3

        with pytest.raises(GeomtryDimError):
            self.topology.add_dihedral([0, 1])

        with pytest.raises(ParticleConflictError):
            self.topology.add_dihedral([1, 1, 2, 0])

        with pytest.raises(ParticleConflictError):
            self.topology.add_dihedral([1, 1, 2, 9])

    def test_del_dihedral(self):
        p1 = Particle(0, 'CA', molecule_id=0, molecule_type='ASN')
        p2 = Particle(1, 'CB', molecule_id=0, molecule_type='ASN')
        p3 = Particle(2, 'N', molecule_id=0, molecule_type='ASN')
        p4 = Particle(3, 'H', molecule_id=0, molecule_type='ASN')
        self.topology.add_particles([p1, p2, p3, p4])
        self.topology.add_dihedral([0, 1, 2, 3])
        assert self.topology.num_dihedrals == 1
        self.topology.del_dihedral([0, 1, 2, 3])
        assert self.topology.num_dihedrals == 0
        assert self.topology.particles[0].num_scaling_particles == 0

        self.topology.add_dihedral([0, 1, 2, 3])
        assert self.topology.num_dihedrals == 1
        self.topology.del_dihedral([3, 2, 1, 0])
        assert self.topology.num_dihedrals == 0

        self.topology.add_dihedral([0, 1, 2, 3])
        assert self.topology.num_dihedrals == 1
        self.topology.del_dihedral([1, 2, 3, 0])
        assert self.topology.num_dihedrals == 1

        with pytest.raises(GeomtryDimError):
            self.topology.del_dihedral([1, 2, 3])

        with pytest.raises(ParticleConflictError):
            self.topology.del_dihedral([0, 0, 1, 2])
        
        with pytest.raises(ParticleConflictError):
            self.topology.del_dihedral([0, 0, 1, 9])

    def test_add_improper(self):
        p1 = Particle(0, 'CA', molecule_id=0, molecule_type='ASN')
        p2 = Particle(1, 'CB', molecule_id=0, molecule_type='ASN')
        p3 = Particle(2, 'N', molecule_id=0, molecule_type='ASN')
        p4 = Particle(3, 'H', molecule_id=0, molecule_type='ASN')
        self.topology.add_particles([p1, p2, p3, p4])
        self.topology.add_improper([0, 1, 2, 3])
        assert self.topology.num_impropers == 1

        self.topology.add_improper([3, 2, 1, 0])
        assert self.topology.num_impropers == 2

        with pytest.raises(GeomtryDimError):
            self.topology.add_improper([1, 2, 3])

        with pytest.raises(ParticleConflictError):
            self.topology.add_improper([2, 1, 1, 3])

        with pytest.raises(ParticleConflictError):
            self.topology.add_improper([2, 1, 2, 9])

    def test_del_improper(self):
        p1 = Particle(0, 'CA', molecule_id=0, molecule_type='ASN')
        p2 = Particle(1, 'CB', molecule_id=0, molecule_type='ASN')
        p3 = Particle(2, 'N', molecule_id=0, molecule_type='ASN')
        p4 = Particle(3, 'H', molecule_id=0, molecule_type='ASN')
        self.topology.add_particles([p1, p2, p3, p4])
        self.topology.add_improper([0, 1, 2, 3])
        assert self.topology.num_impropers == 1
        self.topology.del_improper([0, 1, 2, 3])
        assert self.topology.num_impropers == 0

        self.topology.add_improper([0, 1, 2, 3])
        assert self.topology.num_impropers == 1
        self.topology.del_improper([3, 2, 1, 0])
        assert self.topology.num_impropers == 1

        with pytest.raises(GeomtryDimError):
            self.topology.del_improper([0, 1, 2])

        with pytest.raises(ParticleConflictError):
            self.topology.del_improper([1, 2, 1, 3])

        with pytest.raises(ParticleConflictError):
            self.topology.del_improper([1, 2, 9, 3])
    