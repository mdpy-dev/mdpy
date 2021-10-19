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

import pytest
from ..dumper import Dumper

class TestDumper:
    def setup(self):
        pass

    def teardown(self):
        pass

    def test_attributes(self):
        dumper = Dumper('test.dump', 1)
        assert dumper.file_path == 'test.dump'
        assert dumper.dump_frequency == 1

    def test_exceptions(self):
        dumper = Dumper('test.dump', 1)
        with pytest.raises(NotImplementedError):
            dumper.dump(1)