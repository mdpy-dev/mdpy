#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : check_quantity.py
created time : 2021/10/10
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

from mdpy.unit import Unit, Quantity

def check_quantity(val, target_unit: Unit):
    if isinstance(val, Quantity):
        return val.convert_to(target_unit)
    elif isinstance(val, type(None)):
        return None
    else:
        return Quantity(val, target_unit)

def check_quantity_value(val, target_unit: Unit):
    if isinstance(val, Quantity):
        return val.convert_to(target_unit).value
    elif isinstance(val, type(None)):
        return None
    else:
        return Quantity(val, target_unit).value