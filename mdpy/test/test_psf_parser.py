#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_psf_parser.py
created time : 2021/10/05
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import pytest, os
from mdpy import env
from mdpy.io import PSFParser

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data')

class TestPSFParser:
    def setup(self):
        self.file_path = os.path.join(data_dir, '1M9Z.psf')

    def teardown(self):
        pass

    def test_attributes(self):
        psf = PSFParser(self.file_path)
        assert psf.num_particles == 95567
        assert psf.particle_ids[1] == 1
        assert psf.particle_names[3] == 'HT3'
        assert psf.particle_types[15] == 'HB1'
        assert psf.molecule_ids[9] == 26
        assert psf.molecule_types[21] == 'LEU'
        assert psf.chain_ids[1616] == 'WT1'
        assert psf._charges[2] == pytest.approx(0.33)
        assert psf._masses[5] == env.NUMPY_FLOAT(1.008)

    def test_topology(self):
        psf = PSFParser(self.file_path)
        topology = psf.topology
        assert topology.num_particles == psf.num_particles
        assert topology.num_bonds == psf.num_bonds
        assert topology.num_angles == psf.num_angles
        assert topology.num_dihedrals == psf.num_dihedrals
        assert topology.num_impropers == psf.num_impropers