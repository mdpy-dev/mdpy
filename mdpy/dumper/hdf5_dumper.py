#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : hdf5_dumper.py
created time : 2022/02/25
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

from mdpy import Simulation
from mdpy.dumper import Dumper
from mdpy.io import HDF5Writer

class HDF5Dumper(Dumper):
    def __init__(self, file_path: str, dump_frequency: int) -> None:
        super().__init__(file_path, dump_frequency, suffix='hdf5')
        self._writer = HDF5Writer(file_path, 'w')

    def dump(self, simulation: Simulation):
        if self._num_dumpped_frames == 0:
            self._writer.topology = simulation.ensemble.topology
            self._writer.pbc_matrix = simulation.ensemble.state.pbc_matrix
        self._writer.write(simulation.ensemble.state.positions)
        self._num_dumpped_frames += 1