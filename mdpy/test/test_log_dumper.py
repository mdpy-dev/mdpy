#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_log_dumper.py
created time : 2021/10/29
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import pytest, os
import numpy as np
from mdpy.io import PSFParser, PDBParser
from mdpy.forcefield import CharmmForcefield
from mdpy.integrator import VerletIntegrator
from mdpy.simulation import Simulation
from mdpy.dumper import LogDumper
from mdpy.error import *
from mdpy.unit import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data')
out_dir = os.path.join(cur_dir, 'out')

class TestLogDumper:
    def setup(self):
        self.log_file = os.path.join(out_dir, 'test_log_dumper.log')

    def teardown(self):
        pass

    def test_attributes(self):
        pass

    def test_exceptions(self):
        with pytest.raises(DumperPoorDefinedError):
            LogDumper(self.log_file, 100, rest_time=True)

        with pytest.raises(FileFormatError):
            LogDumper('test.lo', 10)

    def test_dump(self):
        prot_name = '6PO6'
        psf = PSFParser(os.path.join(data_dir, prot_name + '.psf'))
        pdb = PDBParser(os.path.join(data_dir, prot_name + '.pdb'))

        topology = psf.topology

        forcefield = CharmmForcefield(topology, np.eye(3) * 100)
        forcefield.set_param_files(
            os.path.join(data_dir, 'par_all36_prot.prm'),
            os.path.join(data_dir, 'toppar_water_ions_namd.str')
        )

        ensemble = forcefield.create_ensemble()
        ensemble.state.cell_list.set_cutoff_radius(12)
        ensemble.state.set_positions(pdb.positions)
        ensemble.state.set_velocities_to_temperature(Quantity(300, kelvin))

        integrator = VerletIntegrator(Quantity(0.01, femtosecond))
        simulation = Simulation(ensemble, integrator)
        dump_interval = 5
        log_dumper = LogDumper(
            self.log_file, dump_interval,
            step=True, sim_time=True,
            volume=True, density=True,
            potential_energy=True,
            kinetic_energy=True,
            total_energy=True,
            temperature=True
        )
        log_dumper.dump(simulation)