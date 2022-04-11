#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : electrostatic_fdpe_constraint.py
created time : 2022/04/11
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
'''

import time
import numpy as np
import numba as nb
import scipy.sparse as sp
import cupy as cp
import cupyx.scipy.sparse as cps
from cupyx.scipy.sparse.linalg import spsolve, splu
from mdpy import env, SPATIAL_DIM
from mdpy.core import Ensemble
from mdpy.constraint import Constraint
from mdpy.utils import *
from mdpy.unit import *
from mdpy.error import *

from . import Constraint

@np.vectorize
def epsilon(x, y):
    return 0

parameters = {
    'relative_permittivity': epsilon,
    'num_grids': [100, 100, 100]
}

class ElectrostaticFDPEConstraint(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self._epsilon = parameters['relative_permittivity']
        self._num_grids = np.array(parameters['num_grids'], dtype=env.NUMPY_INT)
        self._num_points = self._num_grids + 1
        self._num_points_total = self._num_points.prod()

        self._map_sub_to_index = np.vectorize(self._map_sub_to_index, excluded='self')
        self._create_coefficient_matrix = nb.njit((
            env.NUMBA_INT[:], env.NUMBA_FLOAT[:]
        ))(self._create_coefficient_matrix_kernel)
        if env.platform == 'CPU':
            self._solve_fun = sp.linalg.spsolve
        elif env.platform == 'CUDA':
            self._solve_fun = spsolve

    def __repr__(self) -> str:
        return '<mdpy.constraint.ElectrostaticFDPEConstraint object>'

    __str__ = __repr__

    def _map_sub_to_index(self, i, j, k):
        return i + j * self._num_points[0] + k * self._num_points[0] * self._num_points[1]

    def bind_ensemble(self, ensemble: Ensemble):
        self._parent_ensemble = ensemble
        self._force_id = ensemble.constraints.index(self)

        pbc_diag = np.diagonal(self._parent_ensemble.state.pbc_matrix).astype(env.NUMPY_FLOAT)
        self._bin_widths = (pbc_diag / self._num_grids).astype(env.NUMPY_FLOAT)
        self._grid_size = np.zeros([SPATIAL_DIM, 2])
        self._grid_size[:, 0] = pbc_diag / -2
        self._grid_size[:, 1] = pbc_diag / 2
        self._charges = self._parent_ensemble.topology.charges
        if self._charges.sum() != 0:
            raise EnsemblePoorDefinedError(
                'mdpy.constraint.ElectrostaticFDPEConstraint is bound to a non-neutralized ensemble'
            )

        # Construct matrix
        index_array, data, row, col = self._create_coefficient_matrix(self._num_points, self._bin_widths)
        self._index_array = index_array.astype(env.NUMPY_INT)
        if env.platform == 'CPU':
            self._coefficient_matrix = sp.coo_matrix(
                (np.array(data), (np.array(row), np.array(col))),
                shape=(self._num_points_total, self._num_points_total)
            ).tocsr()
        elif env.platform == 'CUDA':
            self._coefficient_matrix = cps.coo_matrix(
                (cp.array(data), (cp.array(row), cp.array(col))),
                shape=(self._num_points_total, self._num_points_total)
            ).tocsr()

    @staticmethod
    def _create_coefficient_matrix_kernel(num_points, bin_widths):
        index_array = np.ones((num_points[0], num_points[1], num_points[2]))
        for i in range(num_points[0]):
            for j in range(num_points[1]):
                for k in range(num_points[2]):
                    index_array[i, j, k] = i + j*num_points[0] + k*num_points[0]*num_points[1]
        data, row, col = [], [], []
        hx2, hy2, hz2 = bin_widths**2
        data_template = [
            -2 * (hy2*hz2 + hx2*hz2 + hx2*hy2),
            hy2*hz2, hy2*hz2,
            hx2*hz2, hx2*hz2,
            hx2*hy2, hx2*hy2
        ]
        for i in range(num_points[0]):
            for j in range(num_points[1]):
                for k in range(num_points[2]):
                    data.extend(data_template)
                    row.extend([index_array[i, j, k]]*7)
                    col.extend([
                        index_array[i, j, k],
                        index_array[i-1, j, k],
                        index_array[i+1 if i!=num_points[0]-1 else 0, j, k],
                        index_array[i, j-1, k],
                        index_array[i, j+1 if j!=num_points[1]-1 else 0, k],
                        index_array[i, j, k-1],
                        index_array[i, j, k+1 if k!=num_points[2]-1 else 0]
                    ])
        return index_array, np.array(data), np.array(row), np.array(col)

    def update(self):
        self._check_bound_state()
        s = time.time()
        # Assign charge
        indexes = np.round(self._parent_ensemble.state.positions / self._bin_widths).astype(env.NUMPY_INT)
        indexes -= indexes.min(0)
        data, row, col = [], [], []
        factor = self._bin_widths.prod()**2
        for particle in range(self._parent_ensemble.topology.num_particles):
            data.append(self._charges[particle, 0]*factor)
            row.append(self._index_array[tuple(indexes[particle, :])])
            col.append(0)
        if env.platform == 'CPU':
            source = sp.coo_matrix(
                (np.array(data), (np.array(row), np.array(col))),
                shape=(self._num_points_total, 1)
            ).tocsr()
        elif env.platform == 'CUDA':
            source = cps.coo_matrix(
                (cp.array(data), (cp.array(row), cp.array(col))),
                shape=(self._num_points_total, 1)
            ).tocsr()

            self._solve_fun(self._coefficient_matrix, source)
        # self._solve_fun(self._coefficient_matrix, source)
        e = time.time()
        print('Run xxx for %s s' %(e-s))

        # Solve equation