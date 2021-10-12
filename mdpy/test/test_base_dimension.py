#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_base_dimension.py
created time : 2021/09/28
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest
import numpy as np
from ..unit import BaseDimension

class TestUnit:
    def setup(self):
        pass

    def teardown(self):
        pass

    def test_attributes(self):
        constant = BaseDimension()
        assert constant.length_dimension == 0
        assert constant.time_dimension == 0
        assert constant.mass_dimension == 0
        assert constant.temperature_dimension == 0
        assert constant.charge_dimension == 0
        assert constant.mol_dimension == 0

    def test_exceptions(self):
        constant = BaseDimension()
        with pytest.raises(AttributeError):
            constant.length_dimension = 1

        with pytest.raises(AttributeError):
            constant.time_dimension = 1
        
        with pytest.raises(AttributeError):
            constant.mass_dimension = 1
        
        with pytest.raises(AttributeError):
            constant.temperature_dimension = 1
        
        with pytest.raises(AttributeError):
            constant.charge_dimension = 1
        
        with pytest.raises(AttributeError):
            constant.mol_dimension = 1

    def test_eq(self):
        length = BaseDimension(length_dimension=1)
        mass = BaseDimension(mass_dimension=1)
        assert length == length
        assert not mass == length

    def test_ne(self):
        length = BaseDimension(length_dimension=1)
        mass = BaseDimension(mass_dimension=1)
        assert mass != length
        assert not length != length

    def test_mul(self):
        constant = BaseDimension()
        length = BaseDimension(length_dimension=1)
        force = BaseDimension(mass_dimension=1, length_dimension=1, time_dimension=-2)
        energy = BaseDimension(mass_dimension=1, length_dimension=2, time_dimension=-2)
        assert energy == force * length
        assert 1*energy == BaseDimension(mass_dimension=1, length_dimension=2, time_dimension=-2)
        assert energy == energy * constant

    def test_div(self):
        constant = BaseDimension()
        length = BaseDimension(length_dimension=1)
        force = BaseDimension(mass_dimension=1, length_dimension=1, time_dimension=-2)
        energy = BaseDimension(mass_dimension=1, length_dimension=2, time_dimension=-2)
        assert force == energy / length
        assert 1/length == BaseDimension(length_dimension=-1)
        assert force == force / constant

    def test_pow(self):
        length = BaseDimension(length_dimension=1)
        square = BaseDimension(length_dimension=2)
        volume = BaseDimension(length_dimension=3)
        assert square == length**2
        assert volume == length**3

        with pytest.raises(ValueError):
            length**[1, 2]
        with pytest.raises(ValueError):
            length**np.array([1, 2])

    def test_sqrt(self):
        length = BaseDimension(length_dimension=1)
        square = BaseDimension(length_dimension=2)
        volume = BaseDimension(length_dimension=3)
        assert length == square.sqrt()
        assert volume.sqrt() == BaseDimension(length_dimension=1.5)

    def test_generate_dimension_name(self):
        dimension = BaseDimension()
        assert dimension.name == ''

        dimension = BaseDimension(length_dimension=1)
        assert dimension.name == 'm'

        dimension *= dimension
        assert dimension.name == 'm^2'

        dimension = BaseDimension(length_dimension=-1)
        assert dimension.name == '1/m'

        dimension *= dimension
        assert dimension.name == '1/m^2'

        dimension = BaseDimension(time_dimension=1, length_dimension=1)
        assert dimension.name == 'm*s'

        dimension *= dimension
        assert dimension.name == 'm^2*s^2'

        dimension = BaseDimension(time_dimension=1, length_dimension=-1)
        assert dimension.name == 's/m'

        dimension *= dimension
        assert dimension.name == 's^2/m^2'

        dimension = BaseDimension(time_dimension=-1, length_dimension=-1)
        assert dimension.name == '1/m*s'

        dimension *= dimension
        assert dimension.name == '1/m^2*s^2'