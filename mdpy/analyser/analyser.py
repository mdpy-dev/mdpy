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
from ..utils import check_selection_condition

class AnalyserResult:
    def __init__(self, title: str, description: dict, result: dict) -> None:
        self._title = title
        self._description = description
        self._result = result

    def __repr__(self):
        return '<mdpy.analyser.AnalyserResult object of %s at %x>' %(self._title, id(self))

    __str__ = __repr__

    @property
    def title(self):
        return self._title
    
    @property
    def description(self):
        descrption = '------------\nDescription of %s\n' %self
        descrption += 'Title: \n- %s\n' %self._title
        descrption += 'Keys: \n'
        for key, value in self._description.items():
            descrption += '- %s: %s\n' %(key, value)
        return descrption + '------------'

    @property
    def result(self):
        return self._result

class Analyser:
    def __init__(self, selection_condition: list[dict]) -> None:
        check_selection_condition(selection_condition)
        self._selection_condition = selection_condition

    def analysis(self, trajectory: Trajectory) -> AnalyserResult:
        raise NotImplementedError(
            'The subclass of mdpy.analyser.Analyser class should overload analysis method.'
        )

    def save(self, file_path):
        raise NotImplementedError(
            'The subclass of mdpy.analyser.Analyser class should overload save method.'
        )

    @property
    def selection_condition(self):
        return self._selection_condition 

    @selection_condition.setter
    def selection_condition(self, condition: list[dict]):
        self._selection_condition = condition