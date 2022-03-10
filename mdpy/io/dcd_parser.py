#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : dcd_parser.py
created time : 2022/03/10
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import numpy as np
import MDAnalysis as mda
from .. import env
from ..error import *

class DCDParser:
    def __init__(self, file_path: str) -> None:
        # Initial reader and parser setting
        if not file_path.endswith('.dcd'):
            raise FileFormatError('The file should end with .dcd suffix')
        self._file_path = file_path
        self._reader = mda.coordinates.DCD.DCDReader(self._file_path)
        if self._reader.n_frames == 1:
            self._positions = self._reader.ts.positions.astype(env.NUMPY_FLOAT)
        else:
            self._positions = [ts.positions.astype(env.NUMPY_FLOAT) for ts in self._reader.trajectory]
            self._positions = np.stack(self._positions)
        self._pbc_matrix = self._reader.ts.triclinic_dimensions

    @property
    def positions(self) -> np.ndarray:
        return self._positions.copy()

    @property
    def pbc_matrix(self) -> np.ndarray:
        return self._pbc_matrix.copy()