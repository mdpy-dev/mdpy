#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_log_writer.py
created time : 2021/10/29
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import pytest, os
import numpy as np
from mdpy.io import PSFParser, PDBParser
from mdpy.forcefield import CharmmForcefield
from mdpy.io import LogWriter
from mdpy.error import *
from mdpy.unit import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, "data/simulation/")
out_dir = os.path.join(cur_dir, "out")


class TestLogWriter:
    def setup(self):
        self.log_file = os.path.join(out_dir, "test_log_dumper.log")

    def teardown(self):
        pass

    def test_attributes(self):
        pass

    def test_exceptions(self):
        with pytest.raises(FileFormatError):
            LogWriter("test.lo", 10)

    def test_write(self):
        prot_name = "6PO6"
        psf = PSFParser(os.path.join(data_dir, prot_name + ".psf"))
        pdb = PDBParser(os.path.join(data_dir, prot_name + ".pdb"))

        topology = psf.topology

        forcefield = CharmmForcefield(
            topology, np.eye(3) * 100, long_range_solver="CUTOFF"
        )
        forcefield.set_parameter_files(
            os.path.join(data_dir, "par_all36_prot.prm"),
            os.path.join(data_dir, "toppar_water_ions_namd.str"),
        )

        ensemble = forcefield.create_ensemble()
        ensemble.tile_list.set_cutoff_radius(12)
        ensemble.state.set_positions(pdb.positions)
        ensemble.state.set_velocities_to_temperature(Quantity(300, kelvin))
        log_writer = LogWriter(
            self.log_file,
            Quantity(0.01, femtosecond),
            step=True,
            sim_time=True,
            volume=True,
            density=True,
            potential_energy=True,
            kinetic_energy=True,
            total_energy=True,
            temperature=True,
        )
        for i in range(10):
            log_writer.write(ensemble, i)
