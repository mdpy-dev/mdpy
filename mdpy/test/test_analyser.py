#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_analyser.py
created time : 2022/02/20
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest
from ..analyser import Analyser

class TestAnalyser:
    def setup(self):
        pass

    def teardown(self):
        pass

    def test_analysis(self):
        analyser = Analyser()
        with pytest.raises(NotImplementedError):
            analyser.analysis(1)