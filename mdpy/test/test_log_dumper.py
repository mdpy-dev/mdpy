#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : test_log_dumper.py
created time : 2021/10/29
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import pytest, os
import numpy as np
from ..file import PSFFile, PDBFile
from ..forcefield import CharmmForcefield
from ..integrator import VerletIntegrator
from ..simulation import Simulation
from ..dumper import LogDumper
from ..error import *
from ..unit import *

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

    def test_dump(self):
        prot_name = '6PO6'
        psf = PSFFile(os.path.join(data_dir, prot_name + '.psf'))
        pdb = PDBFile(os.path.join(data_dir, prot_name + '.pdb'))

        topology = psf.create_topology()

        forcefield = CharmmForcefield(topology)
        forcefield.set_param_files(
            os.path.join(data_dir, 'par_all36_prot.prm'),
            os.path.join(data_dir, 'toppar_water_ions_namd.str')
        )

        ensemble = forcefield.create_ensemble()
        ensemble.state.set_pbc_matrix(np.eye(3) * 100)
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