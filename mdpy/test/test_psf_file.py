#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_psf_file.py
created time : 2021/10/05
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest, os
from .. import env
from ..file import PSFFile

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data')

class TestPSFFile:
    def setup(self):
        self.file_path = os.path.join(data_dir, '1M9Z.psf')

    def teardown(self):
        pass 

    def test_attributes(self):
        psf = PSFFile(self.file_path)
        assert psf.num_particles == 95567
        assert psf.particle_ids[1] == 1
        assert psf.particle_types[3] == 'HT3'
        assert psf.particle_names[15] == 'HB1'
        assert psf.molecule_ids[9] == 26
        assert psf.molecule_types[21] == 'LEU'
        assert psf.chain_ids[1616] == 'WT1'
        assert psf._charges[2] == pytest.approx(0.33)
        assert psf._masses[5] == env.NUMPY_FLOAT(1.008)

    def test_create_topology(self):
        psf = PSFFile(self.file_path)
        topology = psf.create_topology()
        assert topology.num_particles == psf.num_particles
        assert topology.num_bonds == psf.num_bonds
        assert topology.num_angles == psf.num_angles
        assert topology.num_dihedrals == psf.num_dihedrals
        assert topology.num_impropers == psf.num_impropers