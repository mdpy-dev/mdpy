#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : quantity.py
created time : 2021/09/28
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import jax
import jax.numpy as jnp
import numpy as np
from copy import deepcopy
from . import Unit
from .unit_definition import *
from ..error import UnitDimensionDismatchedError, ChangeDeviceBoundedDataError

class Quantity:
    def __init__(self, value, unit: Unit=no_unit) -> None:
        '''
        Parameters
        ----------
        value : int, float, array
            the value of quantity
        unit : Unit
            the unit of quantity
        '''        
        if isinstance(value, np.ndarray):
            self._value = value.astype(np.float64)
        else:        
            self._value = np.array(value).astype(np.float64)

        if unit.is_dimension_less():
            self._value *= unit.relative_value
            self._unit = deepcopy(no_unit)
        else:
            self._unit = deepcopy(unit)

        self._in_device = False
        
    def __repr__(self) -> str:
        return (
            '<Quantity object: %s %s at 0x%x>' 
            %(self._value*self._unit.relative_value, self._unit.base_dimension, id(self))
        )

    def __str__(self) -> str:
        return (
            '%s %s' %(self._value*self._unit.relative_value, self._unit.base_dimension)
        )

    def to_device(self):
        self._value = jax.device_put(self._value)
        self._in_device = True

    def is_dimension_less(self):
        '''
        is_dimension_less judges wether ``self`` is dimensionless

        Returns
        -------
        bool
            - True, the quantity is dimensionless
            - False, the quantity isn't dimensionless
        '''      
        if self._unit.is_dimension_less():
            return True
        else:
            return False

    def convertTo(self, target_unit: Unit):
        '''
        convertTo converts ``self`` to the unit of ``target_unit``

        Parameters
        ----------
        target_unit : mdpy.Unit
            the unit defined by mdpy or users

        Returns
        -------
        mdpy.Quantity
            Quantity with the same absolute value but new unit

        Raises
        ------
        ValueError
            If ``self._unit.base_dimension != target_unit.unit.base_dimension``. E.g ``(10*meter).convertTo(second)``
        '''        
        if self._unit.base_dimension != target_unit.base_dimension:
            raise UnitDimensionDismatchedError(
                'Quantity in %s can not be converted to %s'
                %(self._unit.base_dimension, target_unit.base_dimension)
            )
        else:
            return self / target_unit * target_unit

    def __getitem__(self, key):
        return Quantity(
            self._value[key],
            self._unit
        )

    def __setitem__(self, key, value):
        if self._in_device:
            raise ChangeDeviceBoundedDataError(
                'mdpy.Quantity object does not support item assignment. JAX arrays are immutable;'
            )
        else:
            self._value[key] = (value / self._unit * self._unit).value

    def __eq__(self, other) -> bool:
        if self._in_device:
            eq_judge = jnp.isclose
        else:
            eq_judge = np.isclose
        if isinstance(other, Quantity):
            if self._unit == other.unit:
                return eq_judge(self._value, other.value)
            elif self._unit.base_dimension == other.unit.base_dimension:
                return eq_judge(self._value, other.unit.relative_value / self._unit.relative_value * other.value)
            else:
                raise UnitDimensionDismatchedError(
                    'Quantity in %s can not be compared with quantity in %s'
                    %(self._unit.base_dimension, other.unit.base_dimension)
                )
        # Value judgement, without relative value like 10*angstrom == 10
        elif self.is_dimension_less():
            return eq_judge(self._value, other)
        else:
            return NotImplemented

    def __ne__(self, other) -> bool:
        return ~(self == other) # Invert the result

    def __lt__(self, other) -> bool:
        if isinstance(other, Quantity):
            if self._unit == other.unit:
                return self._value < other.value
            elif self._unit.base_dimension == other.unit.base_dimension:
                return self._value < other.unit.relative_value / self._unit.relative_value * other.value
            else:
                raise UnitDimensionDismatchedError(
                    'Quantity in %s can not be compared with quantity in %s'
                    %(self._unit.base_dimension, other.unit.base_dimension)
                )
        else:
            return NotImplemented

    def __le__(self, other) -> bool:
        if isinstance(other, Quantity):
            if self._unit == other.unit:
                return self._value <= other.value
            elif self._unit.base_dimension == other.unit.base_dimension:
                return self._value <= other.unit.relative_value / self._unit.relative_value * other.value
            else:
                raise UnitDimensionDismatchedError(
                    'Quantity in %s can not be compared with quantity in %s'
                    %(self._unit.base_dimension, other.unit.base_dimension)
                )
        else:
            return NotImplemented
    
    def __gt__(self, other) -> bool:
        if isinstance(other, Quantity):
            if self._unit == other.unit:
                return self._value > other.value
            elif self._unit.base_dimension == other.unit.base_dimension:
                return self._value > other.unit.relative_value / self._unit.relative_value * other.value
            else:
                raise UnitDimensionDismatchedError(
                    'Quantity in %s can not be compared with quantity in %s'
                    %(self._unit.base_dimension, other.unit.base_dimension)
                )
        else:
            return NotImplemented

    def __ge__(self, other) -> bool:
        if isinstance(other, Quantity):
            if self._unit == other.unit:
                return self._value >= other.value
            elif self._unit.base_dimension == other.unit.base_dimension:
                return self._value >= other.unit.relative_value / self._unit.relative_value * other.value
            else:
                raise UnitDimensionDismatchedError(
                    'Quantity in %s can not be compared with quantity in %s'
                    %(self._unit.base_dimension, other.unit.base_dimension)
                )
        else:
            return NotImplemented

    def __add__(self, other):
        if isinstance(other, Quantity):
            return Quantity(
                self._value + other.value * (other.unit.relative_value / self._unit.relative_value),
                self._unit + other.unit # Test wether the base dimension is same Or the dimension will be changed in the next step
            )
        else:
            return NotImplemented

    __iadd__ = __add__
    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Quantity):
            return Quantity(
                self._value - other.value * (other.unit.relative_value / self._unit.relative_value),
                self._unit - other.unit # Test wether the base dimension is same Or the dimension will be changed in the next step
            )
        else:
            return NotImplemented

    __isub__ = __sub__

    def __rsub__(self, other):
        if isinstance(other, Quantity):
            return Quantity(
                other.value - self._value * (self._unit.relative_value / other.unit.relative_value),
                other.unit - self._unit
            )
        else:
            return NotImplemented
            
    def __neg__(self):
        return Quantity(
            - self._value,
            self._unit
        )

    def __mul__(self, other):
        if isinstance(other, Quantity):
            return Quantity(
                self._value * other.value,
                self._unit * other.unit
            )
        elif isinstance(other, Unit):
            return Quantity(
                self._value,
                self._unit * other
            )
        else:
            return NotImplemented

    __imul__ = __mul__
    __rmul__ = __mul__
    
    def __truediv__(self, other):
        if isinstance(other, Quantity):
            return Quantity(
                self._value / other.value,
                self._unit / other.unit
            )
        elif isinstance(other, Unit):
            return Quantity(
                self._value,
                self._unit / other
            )
        else:
            return NotImplemented

    __itruediv__ = __truediv__

    def __rtruediv__(self, other):
        if isinstance(other, Quantity):
            return Quantity(
                other.value / self._value,
                other.unit / self._unit
            )
        elif isinstance(other, Unit):
            return Quantity(
                1 / self._value,
                other / self._unit
            )
        else:
            return NotImplemented

    def __pow__(self, value):
        try:
            len(value)
        except:
            return Quantity(
                self._value**value,
                self._unit**value
            )
        raise ValueError('The power term should be a single number')
        

    def sqrt(self):
        '''
        sqrt returns square root of Quantity

        Returns
        -------
        Unit
            square root of ``self``
        '''   
        if self._in_device:
            return Quantity(
                jnp.sqrt(self._value),
                self._unit.sqrt()
            )
        else:
            return Quantity(
                np.sqrt(self._value),
                self._unit.sqrt()
            )

    def __abs__(self):
        return Quantity(
            abs(self._value),
            self._unit
        )

    @property
    def value(self):
        return self._value

    @property
    def unit(self):
        return self._unit

    @value.setter
    def value(self, val):
        if self._in_device:
            raise ChangeDeviceBoundedDataError('mdpy.Quantity object does not support item assignment. JAX arrays are immutable;')
        else:
            self._value = np.array(val).astype(np.float64)