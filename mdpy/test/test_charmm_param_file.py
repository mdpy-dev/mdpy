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
import numpy as np
from .. import env
from ..unit import *
from ..error import *
from ..file import CharmmParamFile
from ..file.charmm_param_file import RMIN_TO_SIGMA_FACTOR

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

    def test_embed_x_element(self):
        charmm = CharmmParamFile()
        charmm._params['atom'] = 'C CA N H S'.split()
        embed_res = charmm._embed_x_element('SS-CS')
        assert len(embed_res) == 1

        embed_res = charmm._embed_x_element('X-A')
        assert len(embed_res) == 5
        assert embed_res[0] == 'C-A'
        assert embed_res[-1] == 'S-A'

        embed_res = charmm._embed_x_element('X-A-X')
        assert len(embed_res) == 25
        assert embed_res[0] == 'C-A-C'
        assert embed_res[5] == 'CA-A-C'
        assert embed_res[-1] == 'S-A-S'

        embed_res = charmm._embed_x_element('X-A-X-X')
        assert len(embed_res) == 125
        assert embed_res[2] == 'C-A-C-N'

    def test_parse_par_file(self):
        charmm = CharmmParamFile()
        charmm.parse_par_file(os.path.join(data_dir, 'par_all36_prot.prm'))
        # Mass
        assert charmm.params['mass']['H'] == Quantity(1.00800, dalton).value
        # Bond
        assert charmm.params['bond']['CE1-CE1'][0] == Quantity(440, kilocalorie_permol/angstrom**2).convert_to(default_energy_unit/default_length_unit**2).value
        assert charmm.params['bond']['CC-CP1'][1] == Quantity(1.49, angstrom).convert_to(default_length_unit).value # Test for opposite order
        # Angle
        assert charmm.params['angle']['NH2-CT1-HB1'][0] == Quantity(38, kilocalorie_permol).convert_to(default_energy_unit).value
        assert charmm.params['angle']['CPH1-CPH1-CT2'][1] == env.NUMPY_FLOAT(np.deg2rad(130))
        assert charmm.params['angle']['NH2-CT1-HB1'][2] == Quantity(50, kilocalorie_permol).convert_to(default_energy_unit).value
        assert charmm.params['angle']['CAI-CA-HP'][3] == Quantity(2.15250, angstrom).convert_to(default_length_unit).value
        # Dihedral
        assert charmm.params['dihedral']['H-NH2-CT1-HB1'][0][0] == Quantity(0.11, kilocalorie_permol).convert_to(default_energy_unit).value
        assert charmm.params['dihedral']['CT2-CT2-CT2-CT2'][0][1] == Quantity(2).value
        assert charmm.params['dihedral']['C-CT1-NH1-H'][0][2] == Quantity(0).value
        # Improper
        assert charmm.params['improper']['HR1-NR1-NR2-CPH2'][0] == Quantity(0.5, kilocalorie_permol).convert_to(default_energy_unit).value
        assert charmm.params['improper']['O-NH2-CT1-CC'][1] == Quantity(0).value
        # Nonbonded
        assert charmm.params['nonbonded']['CA'][0] == Quantity(0.07, kilocalorie_permol).convert_to(default_energy_unit).value
        assert charmm.params['nonbonded']['SM'][1] == pytest.approx(Quantity(1.975, angstrom).convert_to(default_length_unit).value * RMIN_TO_SIGMA_FACTOR * 2)
        assert charmm.params['nonbonded']['CP1'][2] == Quantity(0.01, kilocalorie_permol).convert_to(default_energy_unit).value
        assert charmm.params['nonbonded']['CP2'][3] == pytest.approx(Quantity(1.9, angstrom).convert_to(default_length_unit).value * RMIN_TO_SIGMA_FACTOR * 2)

    def test_parse_top_file(self):
        charmm = CharmmParamFile()
        charmm.parse_top_file(os.path.join(data_dir, 'top_all36_na.rtf'))
        # Charge
        assert charmm.params['charge']['GUA-NN2B'] == Quantity(-0.02, e).convert_to(default_charge_unit).value
        assert charmm.params['charge']['3POM-ON3'] == Quantity(-0.78, e).convert_to(default_charge_unit).value
        assert charmm.params['charge']['URA-CN7B'] == Quantity(0.14, e).convert_to(default_charge_unit).value

    def test_parse_toppar_file(self):
        charmm = CharmmParamFile()
        charmm.parse_toppar_file(os.path.join(data_dir, 'toppar_water_ions_namd.str'))
        # Charge
        assert charmm.params['charge']['LIT'] == Quantity(1, e).convert_to(default_charge_unit).value
        # Mass
        assert charmm.params['mass']['RUB'] == Quantity(85.46780, dalton).convert_to(default_mass_unit).value
        # Bond
        assert charmm.params['bond']['HT-HT'][0] == Quantity(0, kilocalorie_permol/angstrom**2).convert_to(default_energy_unit/default_length_unit**2).value
        assert charmm.params['bond']['OX-HX'][1] == Quantity(0.97, angstrom).convert_to(default_length_unit).value
        # Angle
        assert charmm.params['angle']['HT-OT-HT'][0] == Quantity(55, kilocalorie_permol).convert_to(default_energy_unit).value
        assert charmm.params['angle']['HT-OT-HT'][1] == np.deg2rad(Quantity(104.52).value)
        # Nonbonded
        assert charmm.params['nonbonded']['OX'][0] == Quantity(0.12, kilocalorie_permol).convert_to(default_energy_unit).value
        assert charmm.params['nonbonded']['BAR'][1] == pytest.approx(Quantity(1.890, angstrom).convert_to(default_length_unit).value * 2 * RMIN_TO_SIGMA_FACTOR)

    def test_parse_multi_files(self):
        f1 = os.path.join(data_dir, 'toppar_water_ions_namd.str')
        f2 = os.path.join(data_dir, 'par_all36_prot.prm')
        f3 = os.path.join(data_dir, 'top_all36_na.rtf')
        charmm = CharmmParamFile(f1, f2, f3)
        # Charge
        assert charmm.params['charge']['GUA-NN2B'] == Quantity(-0.02, e).convert_to(default_charge_unit).value
        assert charmm.params['charge']['3POM-ON3'] == Quantity(-0.78, e).convert_to(default_charge_unit).value
        assert charmm.params['charge']['URA-CN7B'] == Quantity(0.14, e).convert_to(default_charge_unit).value
        assert charmm.params['charge']['LIT'] == Quantity(1, e).convert_to(default_charge_unit).value
        # Mass
        assert charmm.params['mass']['RUB'] == Quantity(85.46780, dalton).convert_to(default_mass_unit).value
        assert charmm.params['mass']['H'] == Quantity(1.00800, dalton).convert_to(default_mass_unit).value
        # Bond
        assert charmm.params['bond']['HT-HT'][0] == Quantity(0, kilocalorie_permol/angstrom**2).convert_to(default_energy_unit/default_length_unit**2).value
        assert charmm.params['bond']['OX-HX'][1] == Quantity(0.97, angstrom).convert_to(default_length_unit).value
        assert charmm.params['bond']['CE1-CE1'][0] == Quantity(440, kilocalorie_permol/angstrom**2).convert_to(default_energy_unit/default_length_unit**2).value
        assert charmm.params['bond']['CC-CP1'][1] == Quantity(1.49, angstrom).convert_to(default_length_unit).value
        # Angle
        assert charmm.params['angle']['HT-OT-HT'][0] == Quantity(55, kilocalorie_permol).convert_to(default_energy_unit).value
        assert charmm.params['angle']['HT-OT-HT'][1] == np.deg2rad(Quantity(104.52).value)
        assert charmm.params['angle']['NH2-CT1-HB1'][0] == Quantity(38, kilocalorie_permol).convert_to(default_energy_unit).value
        assert charmm.params['angle']['CPH1-CPH1-CT2'][1] == np.deg2rad(Quantity(130).value)
        assert charmm.params['angle']['NH2-CT1-HB1'][2] == Quantity(50, kilocalorie_permol).convert_to(default_energy_unit).value
        assert charmm.params['angle']['CAI-CA-HP'][3] == Quantity(2.15250, angstrom).convert_to(default_length_unit).value
        # Dihedral
        assert charmm.params['dihedral']['H-NH2-CT1-HB1'][0][0] == Quantity(0.11, kilocalorie_permol).convert_to(default_energy_unit).value
        assert charmm.params['dihedral']['CT2-CT2-CT2-CT2'][0][1] == Quantity(2).value
        assert charmm.params['dihedral']['C-CT1-NH1-H'][0][2] == Quantity(0).value
        # Improper
        assert charmm.params['improper']['HR1-NR1-NR2-CPH2'][0] == Quantity(0.5, kilocalorie_permol).convert_to(default_energy_unit).value
        assert charmm.params['improper']['O-NH2-CT1-CC'][1] == Quantity(0).value
        # Nonbonded
        assert charmm.params['nonbonded']['OX'][0] == Quantity(0.12, kilocalorie_permol).convert_to(default_energy_unit).value
        assert charmm.params['nonbonded']['BAR'][1] == pytest.approx(Quantity(1.890, angstrom).convert_to(default_length_unit).value * 2 * RMIN_TO_SIGMA_FACTOR)
        assert charmm.params['nonbonded']['CA'][0] == Quantity(0.07, kilocalorie_permol).convert_to(default_energy_unit).value
        assert charmm.params['nonbonded']['SM'][1] == pytest.approx(Quantity(1.975, angstrom).convert_to(default_length_unit).value * 2 * RMIN_TO_SIGMA_FACTOR)
        assert charmm.params['nonbonded']['CP1'][2] == Quantity(0.01, kilocalorie_permol).convert_to(default_energy_unit).value
        assert charmm.params['nonbonded']['CP2'][3] == pytest.approx(Quantity(1.9, angstrom).convert_to(default_length_unit).value * RMIN_TO_SIGMA_FACTOR * 2)
        