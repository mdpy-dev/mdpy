#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_unit.py
created time : 2021/09/28
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest
import numpy as np
from .. import env
from ..unit import BaseDimension, Unit
from ..error import UnitDimensionDismatchedError

class TestUnit:
    def setup(self):
        self.unit = Unit(BaseDimension(length_dimension=1), 1e-10)

    def teardown(self):
        self.unit = None

    def test_attributes(self):
        assert self.unit.unit_name == 'm'
        assert self.unit.base_dimension == BaseDimension(length_dimension=1)
        assert self.unit.relative_value == env.UNIT_FLOAT(1e-10)

    def test_exceptions(self):
        with pytest.raises(AttributeError):
            self.unit.unit_name = 1
        
        with pytest.raises(AttributeError):
            self.unit.base_dimension = 1

        with pytest.raises(AttributeError):
            self.unit.relative_value = 1        

    def test_is_dimension_less(self):
        assert not self.unit.is_dimension_less()
        assert BaseDimension().is_dimension_less()

    def test_set_relative_value_to_one(self):
        self.unit.set_relative_value_to_one()
        assert self.unit.relative_value == 1

    def test_eq(self):
        assert self.unit == Unit(BaseDimension(length_dimension=1), 1e-10)
        assert not self.unit == Unit(BaseDimension(length_dimension=2), 1e-10)

    def test_ne(self):
        assert self.unit != Unit(BaseDimension(length_dimension=2), 1e-10)
        assert not self.unit != Unit(BaseDimension(length_dimension=1), 1e-10)

    def test_add(self):
        angstrom = Unit(BaseDimension(length_dimension=1), 1e-10)
        gram = Unit(BaseDimension(mass_dimension=1), 1e-3)
        assert angstrom + angstrom == angstrom

        with pytest.raises(UnitDimensionDismatchedError):
            gram + angstrom

        with pytest.raises(NotImplementedError):
            1 + angstrom
        
        with pytest.raises(NotImplementedError):
            angstrom + 1

        with pytest.raises(NotImplementedError):
            angstrom += 1
            
    def test_sub(self):
        angstrom = Unit(BaseDimension(length_dimension=1), 1e-10)
        gram = Unit(BaseDimension(mass_dimension=1), 1e-3)
        assert angstrom - angstrom == angstrom

        with pytest.raises(NotImplementedError):
            1 - angstrom
        
        with pytest.raises(NotImplementedError):
            angstrom - 1

        with pytest.raises(NotImplementedError):
            angstrom -= 1

    def test_mul(self):
        angstrom = Unit(BaseDimension(length_dimension=1), 1e-10)
        angstrom_square = Unit(BaseDimension(length_dimension=2), 1e-20)
        assert angstrom * angstrom == angstrom_square

        with pytest.raises(NotImplementedError):
            1 * angstrom
        
        with pytest.raises(NotImplementedError):
            angstrom * 1

        with pytest.raises(NotImplementedError):
            angstrom *= 1

    def test_div(self):
        angstrom_reciprocal = Unit(BaseDimension(length_dimension=-1), 1e10)
        angstrom = Unit(BaseDimension(length_dimension=1), 1e-10)
        angstrom_square = Unit(BaseDimension(length_dimension=2), 1e-20)

        assert angstrom_square / angstrom == angstrom
        assert angstrom / angstrom_square == angstrom_reciprocal
        assert (angstrom / angstrom).is_dimension_less()
        assert 1 / angstrom == angstrom**-1

        with pytest.raises(NotImplementedError):
            2 / angstrom

        with pytest.raises(NotImplementedError):
            1. / angstrom

        with pytest.raises(NotImplementedError):
            angstrom / 1

    def test_pow(self):
        constant = Unit(BaseDimension(), 1)
        angstrom_reciprocal = Unit(BaseDimension(length_dimension=-1), 1e10)
        angstrom_reciprocal_square = Unit(BaseDimension(length_dimension=-2), 1e20)
        angstrom = Unit(BaseDimension(length_dimension=1), 1e-10)
        angstrom_square = Unit(BaseDimension(length_dimension=2), 1e-20)
        assert angstrom_square == angstrom**2
        assert angstrom == angstrom**1
        assert constant == angstrom**0
        assert angstrom_reciprocal == angstrom**-1
        assert angstrom_square**(1/2) == angstrom
        assert angstrom_reciprocal_square**(-1/2) == angstrom
        
        with pytest.raises(ValueError):
            constant**[1, 2]
        with pytest.raises(ValueError):
            constant**np.array([1, 2])

    def test_sqrt(self):
        angstrom_reciprocal = Unit(BaseDimension(length_dimension=-1), 1e10)
        angstrom_reciprocal_square = Unit(BaseDimension(length_dimension=-2), 1e20)
        angstrom = Unit(BaseDimension(length_dimension=1), 1e-10)
        angstrom_square = Unit(BaseDimension(length_dimension=2), 1e-20)

        assert angstrom_reciprocal_square.sqrt() == angstrom_reciprocal
        assert angstrom_square.sqrt() == angstrom