#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : analyser_data.py
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
    def __init__(self, title: str, description: dict, data: dict) -> None:
        self._title = title
        self._description = description
        self._data = data

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
        for key, value in self._data.items():
            save_dict[key] = value.value if isinstance(value, Quantity) else value
        np.savez(file_path, **save_dict)

    @property
    def title(self):
        return self._title
    
    @property
    def description(self):
        return self._description
    
    @property
    def data(self):
        return self._data

def load_analyser_result(file_path: str):
    if not file_path.endswith('npz'):
            raise FileFormatError('mdpy.analyser.AnalyerResult should be save to a .npz file')
    raw_data = np.load(file_path)
    title = raw_data['title'].item()
    description, data = {}, {}
    for key in raw_data.keys():
        if key.startswith('description'):
            target_key = key.split('description-')[-1]
            description[target_key] = raw_data[key].item()
            data[target_key] = raw_data[target_key]
    return AnalyserResult(title=title, description=description, data=data)