#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_pdb_file.py
created time : 2021/10/05
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''


import pytest, os
from ..file import PDBFile

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data')

class TestPDBFile:
    def setup(self):
        self.file_path = os.path.join(data_dir, '1M9Z.pdb')

    def teardown(self):
        pass 

    def test_attributes(self):
        pdb = PDBFile(self.file_path)
        assert pdb.num_particles == 95567
        assert pdb.particle_ids[2] == 3
        assert pdb.particle_types[4] == 'C'
        assert pdb.particle_names[pdb.get_matrix_id(15)] == 'CA'
        assert pdb.molecule_ids[10] == 26
        assert pdb.molecule_types[22] == 'LEU'
        assert pdb.chain_ids[pdb.get_matrix_id(1617)] == 'W'

        assert pdb.pbc_matrix[0, 0] == 100
        assert pdb.pbc_matrix[1, 1] == 100
        assert pdb.pbc_matrix[2, 2] == 100
        assert pdb.pbc_matrix[0, 1] == 0
        assert pdb.pbc_matrix[2, 1] == 0

    def test_exceptions(self):
        pass

    def test_get_particle_info(self):
        pdb = PDBFile(self.file_path)
        particle_info = pdb.get_particle_info(1)
        assert particle_info['particle_id'] == 1
        assert particle_info['particle_type'] == 'N'
        assert particle_info['particle_name'] == 'N'
        assert particle_info['molecule_id'] == 26
        assert particle_info['molecule_type'] == 'ALA'
        assert particle_info['chain_id'] == 'A'

        assert particle_info['position'][0] == pytest.approx(0.337)
        assert particle_info['position'][1] == pytest.approx(-11.334)
        assert particle_info['position'][2] == pytest.approx(11.207)