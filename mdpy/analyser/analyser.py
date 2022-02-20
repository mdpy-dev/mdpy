#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : analyser.py
created time : 2022/02/20
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

from ..core import Trajectory

class Analyser:
    def __init__(self) -> None:
        pass

    def analysis(self, trajectory: Trajectory):
        raise NotImplementedError(
            'The subclass of mdpy.analyser.Analyser class should overload analysis method.'
        )

    def save(self, file_path):
        raise NotImplementedError(
            'The subclass of mdpy.analyser.Analyser class should overload save method.'
        )