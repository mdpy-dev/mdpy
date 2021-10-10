#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_utils.py
created time : 2021/10/10
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest
from ..math import check_quantity, check_quantity_value
from ..unit import *
from ..error import *

def test_check_quantity():
    assert check_quantity(1, default_length_unit) == Quantity(1, default_length_unit)
    assert check_quantity(None, default_length_unit) == None
    assert check_quantity(Quantity(1, meter), default_length_unit).unit == default_length_unit
    assert check_quantity(Quantity(1, meter), default_length_unit).value == 1e10

    with pytest.raises(UnitDimensionDismatchedError):
        check_quantity(Quantity(1, default_charge_unit), default_length_unit)

def test_check_quantity_value():
    assert check_quantity_value(1, default_length_unit) == 1
    assert check_quantity_value(None, default_length_unit) == None
    assert check_quantity_value(Quantity(1, meter), default_length_unit) == 1e10

    with pytest.raises(UnitDimensionDismatchedError):
        check_quantity_value(Quantity(1, default_length_unit), default_mass_unit)