#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : unitDefinition.py
created time : 2021/09/28
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

from . import BaseDimension, Unit

######################
## BaseDimensions   ##
######################
constant = BaseDimension()
length = BaseDimension(length_dimension=1)
mass = BaseDimension(mass_dimension=1)
time = BaseDimension(time_dimension=1)
temperature = BaseDimension(temperature_dimension=1)
charge = BaseDimension(charge_dimension=1)
mol_dimension = BaseDimension(mol_dimension=1)

force = mass * length / time**2
energy = force * length
power = energy / time
velocity = length / time
accelration = length / time**2

######################
## Constant Unit    ##
######################

no_unit = Unit(constant, 1)

######################
## Length Unit      ##
######################

meter = Unit(length, 1)
decimeter = Unit(length, 1e-1)
centermeter = Unit(length, 1e-2)
millimeter = Unit(length, 1e-3)
micrometer = Unit(length, 1e-6)
nanometer = Unit(length, 1e-9)
angstrom = Unit(length, 1e-10)

######################
## Mass Unit        ##
######################

kilogram = Unit(mass, 1)
gram = Unit(mass, 1e-3)
amu = Unit(mass, 1.66053904e-27)
dalton = Unit(mass, 1.66053904e-27)

######################
## Time Unit        ##
######################

day = Unit(time, 60*60*24)
hour = Unit(time, 60*60)
minute = Unit(time, 60)
second = Unit(time, 1)
millisecond = Unit(time, 1e-3)
microsecond = Unit(time, 1e-6)
nanosecond = Unit(time, 1e-9)
picosecond = Unit(time, 1e-12)
femtosecond = Unit(time, 1e-15)

hertz = 1 / second

######################
## Temperature Unit ##
######################

kelvin = Unit(temperature, 1)

######################
## Power Unit       ##
######################

watt = Unit(power, 1)
kilowatt = Unit(power, 1e3)

######################
## Electrical Unit      ##
######################

coulomb = Unit(charge, 1)
e = Unit(charge, 1.602176634e-19)

ampere = coulomb / second
volt = watt / ampere
ohm = volt / ampere
farad = coulomb / volt
siemens = 1 / ohm

######################
## Mol Unit         ##
######################

mol = Unit(mol_dimension, 1)
kilomol = Unit(mol_dimension, 1e3)

######################
## Energy Unit      ##
######################

joule = Unit(energy, 1)
kilojoule = Unit(energy, 1e3)
joule_permol = Unit(energy, 1/6.0221e23)
kilojoule_permol = Unit(energy, 1e3/6.0221e23)

calorie = Unit(energy, 4.184)
kilocalorie = Unit(energy, 4.184e3)
calorie_premol = Unit(energy, 4.184/6.0221e23)
kilocalorie_permol = Unit(energy, 4.184e3/6.0221e23)

ev = Unit(energy, 1.60217662e-19)
hartree = Unit(energy, 4.3597447222071e-18)

######################
## Force Unit       ##
######################

newton = Unit(force, 1)
kilonewton = Unit(force, 100)
kilojoule_permol_over_angstrom = kilojoule_permol / angstrom
kilojoule_permol_over_nanometer = kilojoule_permol / nanometer
kilocalorie_permol_over_angstrom = kilocalorie_permol / angstrom
kilocalorie_permol_over_nanometer = kilocalorie_permol / nanometer