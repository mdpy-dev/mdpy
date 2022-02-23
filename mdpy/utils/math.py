#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : math.py
created time : 2022/02/22
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np

@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.exp(-x))