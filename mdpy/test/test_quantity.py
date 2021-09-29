#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_quantity.py
created_time : 2021/08/15
last edit time : 2021/08/15
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest
import numpy as np
import jax.numpy as jnp
from ..unit import *
from ..unit import Quantity
from ..error import UnitDimensionDismatchedError, ChangeDeviceBoundedDataError

class TestQuantity:
    def setup(self):
        pass

    def teardown(self):
        pass
    def test_attributes(self):
        quantity = Quantity(1) * angstrom
        assert quantity.value == 1
        assert quantity.unit == angstrom

        quantity = Quantity(np.array([1, 2, 3])) * angstrom
        assert quantity[0].value == 1
        assert quantity[0].unit == angstrom

        quantity = Quantity([1, 2, 3, 4]) * angstrom
        assert quantity[0].value == 1
        assert quantity[0].unit == angstrom

        assert isinstance(quantity.value, np.ndarray)

        quantity = Quantity(quantity, meter)
        assert quantity.unit == meter
        assert quantity[0].value == 1e-10
        assert quantity.value[0] == 1e-10

    def test_exceptions(self):
        with pytest.raises(ChangeDeviceBoundedDataError):
            quantity = Quantity(np.array([1, 2, 3])) * angstrom
            quantity.to_device()
            quantity[1] = 0
    def test_to_device(self):
        quantity = Quantity(1, angstrom)
        assert isinstance(quantity.value, np.ndarray)

        quantity.to_device()
        assert isinstance(quantity.value, jnp.DeviceArray)

    def test_indice(self):
        quantity = Quantity([1, 2, 3, 4], angstrom)
        quantity[0] == Quantity(1, angstrom)
        quantity[0] = Quantity(1, nanometer)
        quantity[0].value == 10

        quantity.to_device()
        with pytest.raises(ChangeDeviceBoundedDataError):
            quantity[0] = 1


    def test_convert_to(self):
        quantity = Quantity(1) * angstrom
        quantity_m = quantity.convert_to(meter)
        assert quantity_m.unit == meter
        assert quantity_m.value == 1e-10

        with pytest.raises(UnitDimensionDismatchedError):
            quantity.convert_to(second)

        quantity = Quantity(1) * meter / second
        quantity_an_per_fs = quantity.convert_to(angstrom/femtosecond)
        assert quantity_an_per_fs.unit == (angstrom/femtosecond)
        assert quantity_an_per_fs.value == 1e-5
        with pytest.raises(UnitDimensionDismatchedError):
            quantity.convert_to(second)

        del quantity
        del quantity_m
        del quantity_an_per_fs

    def test_is_dimension_less(self):
        quantity = Quantity(1, angstrom)
        assert not quantity.is_dimension_less()
        
        quantity = Quantity(1)
        assert quantity.is_dimension_less()

        quantity = Quantity([1, 2, 3, 4, 5])
        assert quantity.is_dimension_less()

        del quantity

    def test_eq(self):
        assert Quantity(1) * nanometer == Quantity(10) * angstrom
        assert (Quantity([1, 2, 3, 4]) * angstrom == Quantity([.1, .2, .3, .4], nanometer)).all()

        with pytest.raises(UnitDimensionDismatchedError):
            Quantity(1) * nanometer == Quantity(1) * nanosecond

    def test_ne(self):
        assert Quantity(1) * nanometer != Quantity(10) * nanometer
        assert Quantity(10) * nanometer != Quantity(10) * angstrom

        assert not Quantity(1) * nanometer != Quantity(1) * nanometer
        assert not (Quantity([1, 2, 3, 4]) * angstrom != Quantity([2, 2, 3, 4], angstrom)).all()

        with pytest.raises(UnitDimensionDismatchedError):
            Quantity(1) * nanometer != Quantity(1) * nanosecond

    def test_lt(self):
        assert Quantity(1) * nanometer < Quantity(10) * nanometer
        assert not Quantity(12) * nanometer < Quantity(10) * nanometer
        with pytest.raises(UnitDimensionDismatchedError):
            Quantity(1) * nanometer < Quantity(1) * nanosecond

    def test_le(self):
        assert Quantity(1) * nanometer <= Quantity(10) * nanometer
        assert not Quantity(11) * nanometer <= Quantity(10) * nanometer
        with pytest.raises(UnitDimensionDismatchedError):
            Quantity(1) * nanometer <= Quantity(1) * nanosecond

    def test_gt(self):
        assert Quantity(10) * nanometer > Quantity(1) * nanometer
        assert not Quantity(1, angstrom) > Quantity(1) * nanometer
        with pytest.raises(UnitDimensionDismatchedError):
            Quantity(1) * nanometer > Quantity(1) * nanosecond

    def test_ge(self):
        assert Quantity(10) * nanometer >= Quantity(1) * angstrom
        assert not Quantity(1, angstrom) > Quantity(1) * nanometer
        with pytest.raises(UnitDimensionDismatchedError):
            Quantity(1) * nanometer > Quantity(1) * nanosecond

    def test_add(self):
        quantity = Quantity([1, 2, 3, 4]) * angstrom
        quantity = quantity + Quantity(1, angstrom)
        assert quantity[0] == Quantity(2) * angstrom
        assert quantity[-1] == Quantity(5) * angstrom
        assert (quantity == Quantity([1, 2, 3, 4]) * angstrom + Quantity(1) * angstrom).all()

        quantity = Quantity([1, 2, 3, 4]) * angstrom
        quantity = Quantity(1, angstrom) + quantity
        assert quantity[0] == Quantity(2) * angstrom
        assert quantity[-1] == Quantity(5) * angstrom
        assert (quantity == Quantity([1, 2, 3, 4]) * angstrom + Quantity(1) * angstrom).all()

        quantity = Quantity([1, 2, 3, 4]) * angstrom
        quantity = quantity + Quantity(1, nanometer)
        assert quantity[0] == Quantity(11) * angstrom
        assert quantity[-1] == Quantity(14) * angstrom
        assert (quantity == Quantity([1, 2, 3, 4]) * angstrom + Quantity(1) * nanometer).all()

        with pytest.raises(UnitDimensionDismatchedError):
            Quantity(1) * nanometer + Quantity(1) * nanosecond

    def test_sub(self):
        quantity = Quantity([1, 2, 3, 4]) * angstrom
        quantity = quantity - Quantity(1, angstrom)
        assert quantity[0] == Quantity(0) * angstrom
        assert quantity[-1] == Quantity(3) * angstrom
        assert (quantity == Quantity([1, 2, 3, 4]) * angstrom - Quantity(1) * angstrom).all()

        quantity = Quantity([1, 2, 3, 4]) * angstrom
        quantity = Quantity(1, angstrom) - quantity
        assert quantity[0] == Quantity(0) * angstrom
        assert quantity[-1] == Quantity(-3) * angstrom
        assert (quantity == Quantity(1) * angstrom - Quantity([1, 2, 3, 4]) * angstrom).all()

        quantity = Quantity([1, 2, 3, 4]) * angstrom
        quantity = quantity - Quantity(1, nanometer)
        assert quantity[0] == Quantity(-9) * angstrom
        assert quantity[-1] == Quantity(-6) * angstrom
        assert (quantity == Quantity([1, 2, 3, 4]) * angstrom - Quantity(1) * nanometer).all()

        with pytest.raises(UnitDimensionDismatchedError):
            Quantity(1) * nanometer - Quantity(1) * nanosecond

    def test_mul(self):
        quantity = Quantity([1, 2, 3, 4]) * angstrom
        quantity = quantity * Quantity(1, angstrom)
        assert quantity[0] == Quantity(1) * angstrom**2
        assert quantity[-1] == Quantity(4) * angstrom**2

        quantity = Quantity([1, 2, 3, 4]) * angstrom
        quantity = Quantity(1, angstrom) * quantity
        assert quantity[0] == Quantity(1) * angstrom**2
        assert quantity[-1] == Quantity(4) * angstrom**2

        quantity = Quantity([1, 2, 3, 4]) * angstrom
        quantity = quantity * quantity
        assert quantity[0] == Quantity(1) * angstrom**2
        assert quantity[-1] == Quantity(16) * angstrom**2

        quantity = Quantity([1, 2, 3, 4]) * angstrom
        quantity = quantity * Quantity([1, 2, 3, 4]) 
        assert quantity[0] == Quantity(1) * angstrom
        assert quantity[-1] == Quantity(16) * angstrom

        quantity = Quantity([1, 2, 3, 4]) * angstrom
        quantity = quantity * Quantity([1, 2, 3, 4]) * newton
        assert quantity[0] == Quantity(1) * newton * angstrom
        assert quantity[-1] == Quantity(16) * newton * angstrom

    def test_div(self):
        quantity = Quantity([1, 2, 3, 4]) * angstrom
        quantity = quantity / Quantity(1, angstrom)
        assert quantity[0] == Quantity(1) 
        assert quantity[0] == 1
        assert quantity[-1] == Quantity(4) 

        quantity = Quantity([1, 2, 3, 4]) * angstrom
        quantity = Quantity(1, angstrom) / quantity
        assert quantity[0] == Quantity(1)
        assert quantity[-1] == Quantity(0.25)

        quantity = Quantity([1, 2, 3, 4]) * angstrom
        quantity = quantity / quantity
        assert quantity[0] == Quantity(1) 
        assert quantity[-1] == Quantity(1) 

        quantity = Quantity([1, 2, 3, 4]) * angstrom
        quantity = quantity / Quantity([1, 2, 3, 4]) 
        assert quantity[0] == Quantity(1) * angstrom
        assert quantity[-1] == Quantity(1) * angstrom

        quantity = Quantity([1, 2, 3, 4]) * angstrom
        quantity = quantity / (Quantity([1, 2, 3, 4]) * newton)
        assert quantity[0] == Quantity(1) / newton * angstrom
        assert quantity[-1] == Quantity(1) / newton * angstrom
