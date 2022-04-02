#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_charmm_forcefield.py
created time : 2021/10/16
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import pytest, os
import numpy as np
from mdpy.io import PSFParser, CharmmTopparParser
from mdpy.forcefield import CharmmForcefield
from mdpy.error import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data')

class TestCharmmForcefield:
    def setup(self):
        self.psf_file_path = os.path.join(data_dir, '6PO6.psf')

    def teardown(self):
        pass

    def test_attributes(self):
        pass

    def test_exceptions(self):
        with pytest.raises(PBCPoorDefinedError):
            topology = PSFParser(self.psf_file_path).topology
            CharmmForcefield(topology, np.diag([0, 30, 30]))

    def test_check_parameters(self):
        f1 = os.path.join(data_dir, 'toppar_water_ions_namd.str')
        f2 = os.path.join(data_dir, 'par_all36_prot.prm')
        f3 = os.path.join(data_dir, 'top_all36_prot.rtf')
        charmm_file = CharmmTopparParser(f1, f2, f3)
        parameters = charmm_file.parameters
        topology = PSFParser(self.psf_file_path).topology
        forcefield = CharmmForcefield(topology, np.diag([30, 30, 30]))
        forcefield._parameters = parameters
        forcefield.check_parameters()

    def test_create_ensemble(self):
        f1 = os.path.join(data_dir, 'toppar_water_ions_namd.str')
        f2 = os.path.join(data_dir, 'par_all36_prot.prm')
        f3 = os.path.join(data_dir, 'top_all36_prot.rtf')
        topology = PSFParser(self.psf_file_path).topology
        forcefield = CharmmForcefield(topology, np.diag([30, 30, 30]))
        forcefield.set_param_files(f1, f2, f3)
        ensemble = forcefield.create_ensemble()
        assert ensemble.num_constraints == 6