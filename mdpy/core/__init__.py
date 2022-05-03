__author__ = "Zhenyu Wei"
__maintainer__ = "Zhenyu Wei"
__copyright__ = "(C)Copyright 2021-present, mdpy organization"
__license__ = "BSD"

import numpy as np
import cupy as cp
from mdpy import SPATIAL_DIM
from mdpy.environment import *

# Cell list searching constant
NUM_NEIGHBOR_CELLS = SPATIAL_DIM**SPATIAL_DIM
NEIGHBOR_CELL_TEMPLATE = np.zeros([NUM_NEIGHBOR_CELLS, SPATIAL_DIM], dtype=NUMPY_INT)
index = 0
for i in range(-1, 2):
    for j in range(-1, 2):
        for k in range(-1, 2):
            NEIGHBOR_CELL_TEMPLATE[index, :] = [i, j, k]
            index += 1
DEVICE_NEIGHBOR_CELL_TEMPLATE = cp.array(NEIGHBOR_CELL_TEMPLATE, CUPY_INT)

# Threshold value for num_excluded_particles and num_scaled_particles
MAX_NUM_EXCLUDED_PARTICLES = 15
MAX_NUM_SCALED_PARTICLES = 15

from mdpy.core.particle import Particle
from mdpy.core.topology import Topology
from mdpy.core.neighbor_list import NeighborList
from mdpy.core.state import State
from mdpy.core.trajectory import Trajectory
from mdpy.core.ensemble import Ensemble

__all__ = [
    'Particle',
    'Topology',
    'NeighborList',
    'State',
    'Trajectory',
    'Ensemble'
]