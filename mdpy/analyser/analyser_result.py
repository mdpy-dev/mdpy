#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : analyser_result.py
created time : 2022/02/20
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np
from ..unit import *
from ..error import *

class AnalyserResult:
    def __init__(self, title: str, description: dict, result: dict) -> None:
        self._title = title
        self._description = description
        self._result = result

    def __repr__(self):
        descrption = '------------\nDescription of AnalyserResult object at %x\n' %id(self)
        descrption += 'Title: \n- %s\n' %self._title
        descrption += 'Keys: \n'
        for key, value in self._description.items():
            descrption += '- %s: %s\n' %(key, value)
        return descrption + '------------'

    __str__ = __repr__

    def save(self, file_path: str):
        if not file_path.endswith('npz'):
            raise FileFormatError('mdpy.analyser.AnalyerResult should be save to a .npz file')
        save_dict = {}
        save_dict['title'] = self._title
        for key, value in self._description.items():
            save_dict['description-%s' %key] = value
        for key, value in self._result.items():
            save_dict[key] = value.value if isinstance(value, Quantity) else value
        np.savez(file_path, **save_dict)

    @property
    def title(self):
        return self._title
    
    @property
    def description(self):
        return self._description
    
    @property
    def result(self):
        return self._result

def load_analyser_result(file_path: str):
    if not file_path.endswith('npz'):
            raise FileFormatError('mdpy.analyser.AnalyerResult should be save to a .npz file')
    data = np.load(file_path)
    title = data['title'].item()
    description, result = {}, {}
    for key in data.keys():
        if key.startswith('description'):
            target_key = key.split('description-')[-1]
            description[target_key] = data[key].item()
            result[target_key] = data[target_key]
    return AnalyserResult(title=title, description=description, result=result)