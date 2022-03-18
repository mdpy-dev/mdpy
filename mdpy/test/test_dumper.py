#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_dumper.py
created time : 2021/10/19
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest, os
from mdpy.dumper import Dumper
from mdpy.error import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(cur_dir, 'out')

class TestDumper:
    def setup(self):
        self.file = os.path.join(out_dir, 'test_dumper.txt')

    def teardown(self):
        pass

    def test_attributes(self):
        dumper = Dumper(self.file, 1, 'txt')
        assert dumper.file_path == os.path.join(out_dir, 'test_dumper.txt')
        assert dumper.dump_frequency == 1

    def test_exceptions(self):
        dumper = Dumper(self.file, 1, 'txt')
        with pytest.raises(NotImplementedError):
            dumper.dump(1)

        with pytest.raises(FileFormatError):
            Dumper(self.file, 1, 'tt')

    def test_dump_info(self):
        dumper = Dumper(self.file, 1, 'txt')
        dumper._dump_info('test dump info')