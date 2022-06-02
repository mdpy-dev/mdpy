#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_recipe.py
created time : 2021/10/16
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import pytest
from mdpy.core import Topology
from mdpy.recipe import Recipe


class TestRecipe:
    def setup(self):
        self.topology = Topology()

    def teardown(self):
        pass

    def test_attributes(self):
        pass

    def test_exceptions(self):
        recipe = Recipe(self.topology)

        with pytest.raises(NotImplementedError):
            recipe.set_parameter_files()

        with pytest.raises(NotImplementedError):
            recipe.check_parameters()

        with pytest.raises(NotImplementedError):
            recipe.create_ensemble()
