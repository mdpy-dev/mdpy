#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : geometry.py
created time : 2021/10/09
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np

def bond(position1, position2):
    return np.sqrt(((np.array(position1) - np.array(position2))**2).sum())