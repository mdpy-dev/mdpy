#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_analyser_result.py
created time : 2022/02/22
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest
import os
import numpy as np
from ..analyser import AnalyserResult, load_analyser_result
from ..error import *
from ..unit import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data')
out_dir = os.path.join(cur_dir, 'out')

class TestAnalyserResult:
    def setup(self):
        pass

    def teardown(self):
        pass

    def test_attributes(self):
        pass

    def test_exceptions(self):
        pass

    def test_save(self):
        title = 'a'
        description = {'a': 'test'}
        data = {'a': np.ones([10, 1])}
        result = AnalyserResult(title=title, description=description, data=data)
        with pytest.raises(FileFormatError):
            result.save('a')
        result.save(os.path.join(out_dir, 'analyser_result.npz'))

def test_load_analyser_result():
    result = load_analyser_result(os.path.join(data_dir, 'analyser_result.npz'))
    assert result.title == 'a'
    assert result.description['a'] == 'test'
    assert result.data['a'][0] == 1