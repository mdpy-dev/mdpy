#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : baseDimension.py
created time : 2021/09/28
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

class BaseDimension:
    def __init__(
            self, length_dimension=0, 
            time_dimension=0, 
            mass_dimension=0, 
            temperature_dimension=0, 
            charge_dimension = 0,
            mol_dimension=0
        ) -> None:
        '''
        Parameters
        ----------
        length_dimension : int, optional
            dimension of length, by default 0
        time_dimension : int, optional
            dimension of time, by default 0
        mass_dimension : int, optional
            dimension of mass, by default 0
        temperature_dimension : int, optional
            dimension of temperature, by default 0
        charge_dimension : int, optional
            dimension of charge, by default 0
        mol_dimension : int, optional
            dimension of amount of substance, by default 0
        '''        
        self._length_dimension = length_dimension
        self._time_dimension = time_dimension
        self._mass_dimension = mass_dimension
        self._temperature_dimension = temperature_dimension
        self._charge_dimension = charge_dimension
        self._mol_dimension = mol_dimension

    def __repr__(self) -> str:
        dim_str = format_dimension(self)
        return (
            '<BaseDimension object: %s at 0x%x>'
            %(dim_str, id(self))
        )

    def __str__(self) -> str:
        return format_dimension(self)

    def __eq__(self, base_unit):
        if (
            self._length_dimension == base_unit.length_dimension and 
            self._time_dimension == base_unit.time_dimension and 
            self._mass_dimension == base_unit.mass_dimension and 
            self._temperature_dimension == base_unit.temperature_dimension and 
            self._charge_dimension == base_unit.charge_dimension and
            self._mol_dimension == base_unit.mol_dimension
        ):
            return True
        else:
            return False

    def __ne__(self, base_unit) -> bool:
        if (
            self._length_dimension != base_unit.length_dimension or 
            self._time_dimension != base_unit.time_dimension or
            self._mass_dimension != base_unit.mass_dimension or 
            self._temperature_dimension != base_unit.temperature_dimension or 
            self._charge_dimension != base_unit.charge_dimension or
            self._mol_dimension != base_unit.mol_dimension
        ):
            return True
        else:
            return False

    def __mul__(self, base_unit):
        return BaseDimension(
            self._length_dimension + base_unit.length_dimension,
            self._time_dimension + base_unit.time_dimension,
            self._mass_dimension + base_unit.mass_dimension,
            self._temperature_dimension + base_unit.temperature_dimension,
            self._charge_dimension + base_unit.charge_dimension,
            self._mol_dimension + base_unit.mol_dimension
        )

    def __rmul__(self, other):
        return BaseDimension(
            self._length_dimension,
            self._time_dimension,
            self._mass_dimension,
            self._temperature_dimension,
            self._charge_dimension,
            self._mol_dimension
        )

    def __truediv__(self, base_unit):
        return BaseDimension(
            self._length_dimension - base_unit.length_dimension,
            self._time_dimension - base_unit.time_dimension,
            self._mass_dimension - base_unit.mass_dimension,
            self._temperature_dimension - base_unit.temperature_dimension,
            self._charge_dimension - base_unit.charge_dimension,
            self._mol_dimension - base_unit.mol_dimension
        )

    def __rtruediv__(self, other):
        return BaseDimension(
            -self._length_dimension,
            -self._time_dimension,
            -self._mass_dimension,
            -self._temperature_dimension,
            -self._charge_dimension,
            -self._mol_dimension
        )

    def __pow__(self, value):
        try:
            len(value)
        except:
            return BaseDimension(
                self._length_dimension * value,
                self._time_dimension * value,
                self._mass_dimension * value,
                self._temperature_dimension * value,
                self._charge_dimension * value,
                self._mol_dimension * value
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
        return BaseDimension(
            self._length_dimension / 2,
            self._time_dimension / 2,
            self._mass_dimension / 2,
            self._temperature_dimension / 2,
            self._charge_dimension / 2,
            self._mol_dimension / 2
        )

    def is_dimensionless(self):
        '''
        is_dimensionless judges wether ``self`` is a dimensionless

        Returns
        -------
        bool
            If dimension less, return True
            Else: return False
        '''        
        if (
            self._length_dimension == 0 and
            self._time_dimension == 0 and
            self._mass_dimension == 0 and
            self._temperature_dimension == 0 and
            self._charge_dimension == 0 and
            self._mol_dimension == 0 
        ):
            return True
        else:
            return False

    @property
    def length_dimension(self):
        '''
        length_dimension gets the dimension of length

        Returns
        -------
        int
            dimension of length
        '''        
        return self._length_dimension

    @property
    def time_dimension(self):
        '''
        time_dimension gets the dimension of time

        Returns
        -------
        int
            dimension of time
        '''   
        return self._time_dimension

    @property
    def mass_dimension(self):
        '''
        mass_dimension gets the dimension of mass

        Returns
        -------
        int
            dimension of mass
        '''   
        return self._mass_dimension

    @property
    def temperature_dimension(self):
        '''
        temperature_dimension gets the dimension of temperature

        Returns
        -------
        int
            dimension of temperature
        '''   
        return self._temperature_dimension

    @property
    def charge_dimension(self):
        '''
        charge_dimension gets the dimension of charge

        Returns
        -------
        int
            dimension of charge
        '''   
        return self._charge_dimension

    @property
    def mol_dimension(self):
        '''
        mol_dimension gets the dimension of amount of substance

        Returns
        -------
        int
            dimension of amount of substance
        '''   
        return self._mol_dimension


_SUPERSCRIPTS = {
    '0': '\u2070', '1': '\u00b9', '2': '\u00b2', '3': '\u00b3',
    '4': '\u2074', '5': '\u2075', '6': '\u2076', '7': '\u2077',
    '8': '\u2078', '9': '\u2079', '-': '\u207b',
}

def _superscript_exponent(exponent):
    if exponent == 1 or exponent == -1:
        return ''
    result = ''
    for char in str(abs(int(exponent)) if int(exponent) == exponent else abs(exponent)):
        result += _SUPERSCRIPTS.get(char, char)
    if exponent < 0:
        result = '\u207b' + result
    return result

def format_dimension(dimension, use_unicode=True):
    if dimension.is_dimensionless():
        return ''

    pairs = [
        ('m',   dimension.length_dimension),
        ('s',   dimension.time_dimension),
        ('kg',  dimension.mass_dimension),
        ('K',   dimension.temperature_dimension),
        ('C',   dimension.charge_dimension),
        ('mol', dimension.mol_dimension),
    ]

    positive_pairs = [(name, exponent) for name, exponent in pairs if exponent > 0]
    negative_pairs = [(name, -exponent) for name, exponent in pairs if exponent < 0]

    if use_unicode:
        separator = '\u00b7'
        division_sign = '/'
        pos_parts = []
        for name, exponent in positive_pairs:
            superscript_str = _superscript_exponent(exponent)
            pos_parts.append(f'{name}{superscript_str}')
        numerator = separator.join(pos_parts)

        neg_parts = []
        for name, exponent in negative_pairs:
            superscript_str = _superscript_exponent(exponent)
            neg_parts.append(f'{name}{superscript_str}')
        denominator = separator.join(neg_parts)

        if numerator and denominator:
            return f'{numerator}{division_sign}{denominator}'
        elif numerator:
            return numerator
        elif denominator:
            return f'1{division_sign}{denominator}'
        else:
            return ''
    else:
        def _format_ascii(items):
            parts = []
            for name, exponent in items:
                if exponent == 1:
                    parts.append(name)
                else:
                    parts.append(f'{name}^{exponent}')
            return '*'.join(parts)
        pos_str = _format_ascii(positive_pairs)
        neg_str = _format_ascii(negative_pairs)
        if pos_str and neg_str:
            return f'{pos_str}/{neg_str}'
        elif pos_str:
            return pos_str
        elif neg_str:
            return f'1/{neg_str}'
        else:
            return ''
