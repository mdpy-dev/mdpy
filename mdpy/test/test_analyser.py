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
from ..error import *

class TestAnalyser:
    def setup(self):
        pass

    def teardown(self):
        pass

    def test_attributes(self):
        pass

    def test_exceptions(self):
        with pytest.raises(SelectionConditionPoorDefinedError):
            Analyser(1)

        with pytest.raises(SelectionConditionPoorDefinedError):
            Analyser([{'neab': 1}])

    def test_analysis(self):
        analyser = Analyser([{'particle type': ['CA']}])
        with pytest.raises(NotImplementedError):
            analyser.analysis(1)