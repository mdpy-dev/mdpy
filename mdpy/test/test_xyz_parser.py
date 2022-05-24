#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_xyz_parser.py
created time : 2022/04/26
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import os
import pytest
from mdpy.io import XYZParser
from mdpy.error import ArrayDimError, FileFormatError, IOPoorDefinedError

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data/xyz_parser')

class TestXYZParser:
    def setup(self):
        pass

    def teardown(self):
        pass

    def test_attributes(self):
        pass

    def test_exceptions(self):
        with pytest.raises(FileFormatError):
            XYZParser('test.xy')

        with pytest.raises(IOPoorDefinedError):
            XYZParser(
                os.path.join(data_dir, 'single_frame.xyz'), False
            ).positions

        with pytest.raises(ArrayDimError):
            XYZParser(os.path.join(data_dir, 'missing_particles.xyz'))

    def test_parse_single_frame_file(self):
        parser = XYZParser(os.path.join(data_dir, 'single_frame.xyz'))
        assert parser.num_particles == 400
        assert parser.particle_types[0] == 'DPDC'
        assert parser.positions[10, 0] == pytest.approx(0.8732404541)
        assert parser.positions[15, 1] == pytest.approx(-0.0719176835)
        assert parser.positions[16, 2] == pytest.approx(-1.0761971104)
        assert parser.num_frames == 1

    def test_parse_multi_frames_file(self):
        parser = XYZParser(os.path.join(data_dir, 'multi_frames.xyz'))
        assert parser.num_particles == 400
        assert parser.particle_types[20] == 'DPDC'
        assert parser.positions[0, 10, 0] == pytest.approx(0.8732404541)
        assert parser.positions[1, 15, 1] == pytest.approx(-0.0719176835)
        assert parser.positions[0, 16, 2] == pytest.approx(-1.0761971104)
        assert parser.num_frames == 2

test  = TestXYZParser()
test.test_parse_multi_frames_file()