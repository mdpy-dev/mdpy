#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_analyser_result.py
created time : 2022/02/22
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import pytest
import os
import numpy as np
from mdpy.analyser import AnalyserResult, load_analyser_result
from mdpy.error import *
from mdpy.unit import *

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
        title = 'a'
        description = {'a': 'test'}
        data = {'b': np.ones([10, 1])}
        with pytest.raises(AnalyserPoorDefinedError):
            AnalyserResult(title=title, description=description, data=data)

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