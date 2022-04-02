#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : pdb_dumper.py
created time : 2021/10/19
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

from mdpy.dumper import Dumper
from mdpy.io import PDBWriter
from mdpy.simulation import Simulation

class PDBDumper(Dumper):
    def __init__(self, file_path: str, dump_frequency: int) -> None:
        super().__init__(file_path, dump_frequency, 'pdb')
        self._writer = PDBWriter(file_path, 'w')

    def dump(self, simulation: Simulation):
        if self._num_dumpped_frames == 0:
            self._writer.topology = simulation.ensemble.topology
            self._writer.pbc_matrix = simulation.ensemble.state.pbc_matrix
        self._writer.write(simulation.ensemble.state.positions)
        self._num_dumpped_frames += 1