#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_unit_definition.py
created time : 2021/09/28
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest
import numpy as np
from ..unit import energy, force
from ..unit import *

def test_constants():
    # Avogadro Constant
    assert Quantity(1) * mol * NA == 6.0221e23
    assert Quantity(6.0221e23) / NA == Quantity(1) * mol

    # Boltzmann Constant
    assert Quantity(300) * kelvin * KB / kilojoule_permol == 2.494321
    assert Quantity(300) * kelvin * KB / kilocalorie_permol == 0.596157
    assert Quantity(300) * kelvin * KB / ev == 0.02585199
    assert Quantity(300) * kelvin * KB / hartree == 9.500431e-04

def test_length():
    assert Quantity(1) * meter / decimeter == 10
    assert Quantity(1) * meter / centermeter == 100
    assert Quantity(1) * meter / millimeter == 1000
    assert Quantity(1) * meter / micrometer == 1e6
    assert Quantity(1) * meter / nanometer == 1e9
    assert Quantity(1) * meter / angstrom == 1e10

def test_mass():
    assert Quantity(1) * kilogram / gram == 1e3
    assert Quantity(1) * kilogram / amu == 1/1.66053904e-27
    assert Quantity(1) * kilogram / dalton == 1/1.66053904e-27

    assert Quantity(1) * gram / kilogram == 1e-3
    assert Quantity(1) * amu / kilogram == 1.66053904e-27
    assert Quantity(1) * dalton / kilogram == 1.66053904e-27

def test_time():
    assert Quantity(1) * second / millisecond == 1e3
    assert Quantity(1) * second / microsecond == 1e6
    assert Quantity(1) * second / nanosecond == 1e9
    assert Quantity(1) * second / picosecond == 1e12
    assert Quantity(1) * second / femtosecond == 1e15

def test_temperature():
    assert Quantity(1) * kelvin / kelvin == 1

def test_charge():
    assert Quantity(1) * coulomb / e == 1/1.602176634e-19
    q1, q2 = Quantity(1, e), Quantity(1, e)
    length = Quantity(1, angstrom)
    energy_val = q1 * q2 / Quantity(4*np.pi) / EPSILON0 / length
    assert energy_val.unit.base_dimension == energy
    force_val = q1 * q2 / Quantity(4*np.pi) / EPSILON0 / length**2
    assert force_val.unit.base_dimension == force

    factor = 4 * np.pi * EPSILON0.value
    q1 = q1.convert_to(default_charge_unit).value
    q2 = q2.convert_to(default_charge_unit).value
    length = length.convert_to(default_length_unit).value
    force_val_no_unit = q1 * q2 / factor / length**2
    assert force_val.convert_to(default_force_unit).value == pytest.approx(force_val_no_unit)

    force_val_newton = force_val.convert_to(newton)
    assert force_val_newton.value == pytest.approx(2.3e-8, abs=1e-10)

    assert Quantity(1, ohm) == Quantity(1, volt / ampere)
    assert Quantity(1, default_electric_intensity_unit * default_charge_unit) == Quantity(1, default_force_unit)
    assert Quantity(1, volt / meter * coulomb) == Quantity(1, newton)

def test_mol():
    assert Quantity(1) * kilomol / mol == 1e3

def test_energy():
    assert Quantity(1) * joule / kilojoule == 1e-3
    assert Quantity(1) * joule / joule_permol == 6.0221e23
    assert Quantity(1) * joule / kilojoule_permol == 6.0221e20

    assert Quantity(1) * joule / calorie == 1/4.184
    assert Quantity(1) * joule / kilocalorie == 1/4.184e3
    assert Quantity(1) * joule / calorie_premol == 6.0221e23/4.184
    assert Quantity(1) * joule / kilocalorie_permol == 6.0221e23/4.184e3
    
    assert Quantity(1) * joule / ev == 1/1.60217662e-19
    assert Quantity(1) * joule / hartree == 1/4.3597447222071e-18

def test_force():
    assert Quantity(1) * newton / newton == 1
    assert Quantity(1) * kilocalorie_permol_over_nanometer / kilojoule_permol_over_nanometer == 4.184
    assert Quantity(1) * kilocalorie_permol_over_nanometer / kilojoule_permol_over_angstrom == 0.4184

def test_power():
    assert Quantity(1) * watt / kilowatt == 1e-3

def test_mixture():
    assert Quantity(1) * newton * meter == Quantity(1) * joule
    assert Quantity(1) * joule / (NA * mol) == Quantity(1) * joule_permol
    assert Quantity(1) * joule / second == Quantity(1) * watt
    assert Quantity(1) * kilojoule / second == Quantity(1) * kilowatt
    assert Quantity(1) * kilojoule == Quantity(1000) * joule
    assert Quantity(1) * angstrom == Quantity(1e-10) * meter
    assert not Quantity(1) * angstrom == Quantity(1e-9) * meter
    assert Quantity(1) * millisecond == Quantity(1e-3) * second