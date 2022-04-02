#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_dcd_parser.py
created time : 2022/03/10
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import pytest, os
from mdpy.io import DCDParser
from mdpy.unit import *
from mdpy.error import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data')

class TestDCDParser:
    def setup(self):
        self.file_path = os.path.join(data_dir, 'test_dcd_parser.dcd')

    def teardown(self):
        pass

    def test_attributes(self):
        parser = DCDParser(self.file_path)
        # Positions
        assert parser.positions.shape[0] == 20
        assert parser.positions.shape[1] == 14984
        # PBC Matrix
        assert parser.pbc_matrix[0, 0] == 100

    def test_exceptions(self):
        with pytest.raises(FileFormatError):
            DCDParser('te.dc')

        with pytest.raises(ParserPoorDefinedError):
            DCDParser(self.file_path, is_parse_all=False).positions

        with pytest.raises(ArrayDimError):
            DCDParser(self.file_path, is_parse_all=False).get_positions(100)

        with pytest.raises(ArrayDimError):
            DCDParser(self.file_path, is_parse_all=False).get_positions(100, 101)