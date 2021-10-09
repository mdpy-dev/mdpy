#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_charmm_file.py
created time : 2021/10/08
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest, os

from ..unit import *
from ..error import *
from ..file import CharmmParamFile

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data')

class TestCharmmParamFile:
    def setup(self):
        pass

    def teardown(self):
        pass

    def test_attributes(self):
        pass

    def test_exceptions(self):
        with pytest.raises(FileFormatError):
            charmm = CharmmParamFile('x.p')

        with pytest.raises(FileFormatError):
            charmm = CharmmParamFile('to.p')

    def test_parse_par_file(self):
        charmm = CharmmParamFile()
        charmm.parse_par_file(os.path.join(data_dir, 'par_all36_prot.prm'))
        # Mass
        assert charmm.param['mass']['H'] == Quantity(1.00800, dalton)
        # Bond
        assert charmm.param['bond']['CE1-CE1'][0] == Quantity(440, kilocalorie_permol/angstrom**2)
        assert charmm.param['bond']['CC-CP1'][1] == Quantity(1.49, angstrom) # Test for opposite order
        # Angle
        assert charmm.param['angle']['NH2-CT1-HB1'][0] == Quantity(38, kilocalorie_permol)
        assert charmm.param['angle']['CPH1-CPH1-CT2'][1] == Quantity(130)
        # Dihedral
        assert charmm.param['dihedral']['H-NH2-CT1-HB1'][0] == Quantity(0.11, kilocalorie_permol)
        assert charmm.param['dihedral']['CT2-CT2-CT2-CT2'][1] == Quantity(6)
        assert charmm.param['dihedral']['C-CT1-NH1-H'][2] == Quantity(0)
        # Improper
        assert charmm.param['improper']['HR1-NR1-NR2-CPH2'][0] == Quantity(0.5, kilocalorie_permol).convert_to(kilojoule_permol)
        assert charmm.param['improper']['O-NH2-CT1-CC'][1] == Quantity(0)
        # Nonbonded
        assert charmm.param['nonbonded']['CA'][0] == Quantity(-0.07, kilocalorie_permol)
        assert charmm.param['nonbonded']['SM'][1] == Quantity(1.975, angstrom)

    def test_parse_top_file(self):
        charmm = CharmmParamFile()
        charmm.parse_top_file(os.path.join(data_dir, 'top_all36_na.rtf'))
        # Charge
        assert charmm.param['charge']['GUA-NN2B'] == Quantity(-0.02, e)
        assert charmm.param['charge']['3POM-ON3'] == Quantity(-0.78, e)
        assert charmm.param['charge']['URA-CN7B'] == Quantity(0.14, e)

    def test_parse_toppar_file(self):
        charmm = CharmmParamFile()
        charmm.parse_toppar_file(os.path.join(data_dir, 'toppar_water_ions_namd.str'))
        # Charge
        assert charmm.param['charge']['LIT'] == Quantity(1, e)
        # Mass
        assert charmm.param['mass']['RUB'] == Quantity(85.46780, dalton)
        # Bond
        assert charmm.param['bond']['HT-HT'][0] == Quantity(0, kilocalorie_permol/angstrom**2)
        assert charmm.param['bond']['OX-HX'][1] == Quantity(0.97, angstrom)
        # Angle
        assert charmm.param['angle']['HT-OT-HT'][0] == Quantity(55, kilocalorie_permol)
        assert charmm.param['angle']['HT-OT-HT'][1] == Quantity(104.52)
        # Nonbonded
        assert charmm.param['nonbonded']['OX'][0] == Quantity(-0.12, kilocalorie_permol)
        assert charmm.param['nonbonded']['BAR'][1] == Quantity(1.890, angstrom)

    def test_parse_multi_files(self):
        f1 = os.path.join(data_dir, 'toppar_water_ions_namd.str')
        f2 = os.path.join(data_dir, 'par_all36_prot.prm')
        f3 = os.path.join(data_dir, 'top_all36_na.rtf')
        charmm = CharmmParamFile(f1, f2, f3)
        # Charge
        assert charmm.param['charge']['LIT'] == Quantity(1, e)
        assert charmm.param['charge']['GUA-NN2B'] == Quantity(-0.02, e)
        assert charmm.param['charge']['3POM-ON3'] == Quantity(-0.78, e)
        assert charmm.param['charge']['URA-CN7B'] == Quantity(0.14, e)
        # Mass
        assert charmm.param['mass']['RUB'] == Quantity(85.46780, dalton)
        assert charmm.param['mass']['H'] == Quantity(1.00800, dalton)
        # Bond
        assert charmm.param['bond']['HT-HT'][0] == Quantity(0, kilocalorie_permol/angstrom**2)
        assert charmm.param['bond']['OX-HX'][1] == Quantity(0.97, angstrom)
        assert charmm.param['bond']['CE1-CE1'][0] == Quantity(440, kilocalorie_permol/angstrom**2)
        assert charmm.param['bond']['CC-CP1'][1] == Quantity(1.49, angstrom)
        # Angle
        assert charmm.param['angle']['HT-OT-HT'][0] == Quantity(55, kilocalorie_permol)
        assert charmm.param['angle']['HT-OT-HT'][1] == Quantity(104.52)
        assert charmm.param['angle']['NH2-CT1-HB1'][0] == Quantity(38, kilocalorie_permol)
        assert charmm.param['angle']['CPH1-CPH1-CT2'][1] == Quantity(130)
        # Dihedral
        assert charmm.param['dihedral']['H-NH2-CT1-HB1'][0] == Quantity(0.11, kilocalorie_permol)
        assert charmm.param['dihedral']['CT2-CT2-CT2-CT2'][1] == Quantity(6)
        assert charmm.param['dihedral']['C-CT1-NH1-H'][2] == Quantity(0)
        # Improper
        assert charmm.param['improper']['HR1-NR1-NR2-CPH2'][0] == Quantity(0.5, kilocalorie_permol).convert_to(kilojoule_permol)
        assert charmm.param['improper']['O-NH2-CT1-CC'][1] == Quantity(0)
        # Nonbonded
        assert charmm.param['nonbonded']['OX'][0] == Quantity(-0.12, kilocalorie_permol)
        assert charmm.param['nonbonded']['BAR'][1] == Quantity(1.890, angstrom)
        assert charmm.param['nonbonded']['CA'][0] == Quantity(-0.07, kilocalorie_permol)
        assert charmm.param['nonbonded']['SM'][1] == Quantity(1.975, angstrom)
        