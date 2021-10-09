#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_geometry.py
created time : 2021/10/09
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest
import numpy as np
from ..math import *

def test_bond():
    position1 = [0, 1, 0]
    position2 = [0, 0, 0]
    assert bond(position1, position2) == 1

    position2 = [3, 2, 2]
    assert bond(position2, position1) == pytest.approx(np.sqrt(14))