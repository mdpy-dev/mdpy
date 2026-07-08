#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : unit.py
created time : 2021/09/28
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import numpy as np
from . import BaseDimension, UNIT_PRECISION
from .base_dimension import format_dimension
from .. import precision

class Unit:
    def __init__(self, base_dimension:BaseDimension, relative_value) -> None:
        '''
        Parameters
        ----------
        base_dimension : BaseDimension
            the dimension of unit
        relative_value : int or float
            the relative value of ``self`` to the basic unit of ``base_dimension``
        '''        
        self._base_dimension = base_dimension
        self._relative_value = precision.UNIT_FLOAT(relative_value) # The relative value to the normal unit like angstrom in Length 

    def is_dimensionless(self):
        '''
        is_dimensionless judges wether ``self`` is dimensionless

        Returns
        -------
        bool
            - True, the unit is dimensionless
            - False, the unit isn't dimensionless
        '''        
        if self._base_dimension.is_dimensionless():
            return True
        else:
            return False
    
    def __repr__(self):
        return (
            '<Unit object: %.2e %s at 0x%x>'
            %(self._relative_value, format_dimension(self._base_dimension), id(self))
        )

    def __str__(self):
        return (
            '%.2e %s' %(self._relative_value, format_dimension(self._base_dimension))
        )

    def __eq__(self, other) -> bool:
        err = np.abs((self._relative_value - other.relative_value)/self._relative_value)
        if (
            self._base_dimension == other.base_dimension and
            err < UNIT_PRECISION
        ):
            return True
        else:
            return False

    def __ne__(self, other) -> bool:
        return not self == other

    def __mul__(self, other):
        if isinstance(other, Unit):
            return Unit(
                self._base_dimension * other.base_dimension,
                self._relative_value * other.relative_value
            )
        else:
            raise NotImplementedError(
                '* between %s and mdpy.unit.Unit is not implemented' 
                %(type(other))
            )

    __imul__ = __mul__

    def __rmul__(self, other):
        if isinstance(other, Unit):
            return Unit(
                other.base_dimension * self._base_dimension,
                other.relative_value * self._relative_value
            )
        elif isinstance(other, (int, float, np.ndarray)):
            from .quantity import Quantity
            return Quantity(other, self)
        else:
            raise TypeError(
                '* between %s and mdpy.unit.Unit is not supported'
                %(type(other))
            )

    def __truediv__(self, other):
        if isinstance(other, Unit):
            return Unit(
                self._base_dimension / other.base_dimension,
                self._relative_value / other.relative_value
            )
        else:
            raise NotImplementedError(
                '/ between %s and mdpy.unit.Unit is not implemented' 
                %(type(other))
            )

    __itruediv__ = __truediv__

    def __rtruediv__(self, other):
        if isinstance(other, Unit):
            return Unit(
                other.base_dimension / self._base_dimension,
                other.relative_value / self._relative_value
            )
        elif isinstance(other, int) and other == 1:
            return Unit(
                self._base_dimension**-1,
                1 / self._relative_value
            )
        else:
            raise NotImplementedError(
                '/ between mdpy.unit.Unit and %s is not implemented' 
                %(type(other))
            )

    def __pow__(self, value):
        try:
            len(value)
        except:
            return Unit(
                self._base_dimension**value,
                self._relative_value**value
            )
        raise ValueError('The power term should be a single number')
        

    def sqrt(self):
        '''
        sqrt returns square root of Unit

        Returns
        -------
        Unit
            square root of ``self``
        '''        
        return Unit(
            self._base_dimension.sqrt(),
            np.sqrt(self._relative_value)
        )
            
    @property
    def base_dimension(self):
        '''
        base_dimension gets the dimension of the unit

        Returns
        -------
        BaseDimension
            the dimension of unit
        '''        
        return self._base_dimension

    @property
    def relative_value(self):
        '''
        relative_value gets the relative value to the basic unit

        Returns
        -------
        int or float
            the relative value to the basic unit
        '''        
        return self._relative_value