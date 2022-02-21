__author__ = "Zhenyu Wei"
__maintainer__ = "Zhenyu Wei"
__email__ = "zhenyuwei99@gmail.com"
__copyright__ = "Copyright 2021-2021, Southeast University and Zhenyu Wei"
__license__ = "GPLv3"


UNIT_PRECISION = 1e-6
QUANTITY_PRECISION = 1e-6

from .base_dimension import BaseDimension
from .unit import Unit

# BaseDimension
from .unit_definition import length, mass, time, temperature, charge, mol_dimension
from .unit_definition import force, energy, power, velocity, accelration

# Unit
from .unit_definition import meter, decimeter, centermeter, millimeter, micrometer, nanometer, angstrom
from .unit_definition import kilogram, gram, amu, dalton
from .unit_definition import day, hour, minute
from .unit_definition import second, millisecond, microsecond, nanosecond, picosecond, femtosecond
from .unit_definition import kelvin
from .unit_definition import coulomb, e, ampere, volt, ohm, farad, siemens, hertz
from .unit_definition import mol, kilomol
from .unit_definition import joule, kilojoule, joule_permol, kilojoule_permol, calorie, kilocalorie, calorie_premol, kilocalorie_permol, ev, hartree
from .unit_definition import newton, kilonewton
from .unit_definition import kilojoule_permol_over_angstrom, kilojoule_permol_over_nanometer, kilocalorie_permol_over_angstrom, kilocalorie_permol_over_nanometer
from .unit_definition import watt, kilowatt

# Default Unit
default_length_unit = angstrom
default_mass_unit = dalton
default_time_unit = femtosecond
default_temperature_unit = kelvin
default_charge_unit = e
default_mol_unit = mol

default_frequency_unit = 1 / default_time_unit
default_velocity_unit = default_length_unit / default_time_unit
default_accelrated_velocity_unit = default_velocity_unit / default_time_unit
default_energy_unit = default_mass_unit * default_length_unit**2 / default_time_unit**2
default_power_unit = default_energy_unit / default_time_unit
default_force_unit = default_energy_unit / default_length_unit
default_current_unit = default_charge_unit / default_time_unit
default_voltage_unit = default_power_unit / default_current_unit
default_resistance_unit = default_voltage_unit / default_current_unit
default_capacitance_unit = default_charge_unit / default_voltage_unit
default_conductance_unit = 1 / default_resistance_unit
default_electric_intensity_unit = default_voltage_unit / default_length_unit


from .quantity import Quantity
# Constant
KB = Quantity(1.38064852e-23, Unit(energy/temperature, 1))
NA = Quantity(6.0221e23, Unit(1/mol_dimension, 1))
EPSILON0 = Quantity(8.85418e-12, second**2*coulomb**2/meter**3/kilogram).convert_to(
    default_time_unit**2*default_charge_unit**2/default_length_unit**3/default_mass_unit
)

__all__ = [
    'Quantity',

    'default_length_unit', 'default_mass_unit', 'default_time_unit', 
    'default_temperature_unit', 'default_charge_unit', 'default_mol_unit',
    'default_frequency_unit', 'default_velocity_unit', 'default_accelrated_velocity_unit',
    'default_energy_unit', 'default_power_unit', 'default_force_unit', 
    'default_current_unit', 'default_voltage_unit', 'default_resistance_unit',
    'default_capacitance_unit', 'default_conductance_unit', 'default_electric_intensity_unit',

    'meter', 'decimeter', 'centermeter', 'millimeter', 'micrometer', 'nanometer', 'angstrom',
    'kilogram', 'gram', 'amu', 'dalton',
    'day', 'hour', 'minute',
    'second', 'millisecond', 'microsecond', 'nanosecond', 'picosecond', 'femtosecond',
    'kelvin',
    'coulomb', 'e', 'ampere', 'volt', 'ohm', 'farad', 'siemens', 'hertz',
    'mol', 'kilomol',
    'joule', 'kilojoule',  'joule_permol', 'kilojoule_permol', 'calorie', 'kilocalorie',  'calorie_premol', 'kilocalorie_permol', 'ev', 'hartree',
    'newton', 'kilonewton',
    'kilojoule_permol_over_angstrom', 'kilojoule_permol_over_nanometer', 
    'kilocalorie_permol_over_angstrom', 'kilocalorie_permol_over_nanometer',
    'watt', 'kilowatt',
    
    'NA', 'KB', 'EPSILON0'
]