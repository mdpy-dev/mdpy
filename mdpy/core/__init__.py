__author__ = "Zhenyu Wei"
__maintainer__ = "Zhenyu Wei"
__copyright__ = "(C)Copyright 2021-present, mdpy organization"
__license__ = "BSD"

import numpy as np
from mdpy import SPATIAL_DIM, env

# Cell list searching constant
NUM_NEIGHBOR_CELLS = SPATIAL_DIM**SPATIAL_DIM
NEIGHBOR_CELL_TEMPLATE = np.zeros([NUM_NEIGHBOR_CELLS, SPATIAL_DIM], dtype=env.NUMPY_INT)
index = 0
for i in range(-1, 2):
    for j in range(-1, 2):
        for k in range(-1, 2):
            NEIGHBOR_CELL_TEMPLATE[index, :] = [i, j, k]
            index += 1
# Threshold value for num_bonded_particles and num_scaling_particles
MAX_NUM_BONDED_PARTICLES = 15
MAX_NUM_SCALING_PARTICLES = 15

from mdpy.core.particle import Particle
from mdpy.core.topology import Topology
from mdpy.core.cell_list import CellList
from mdpy.core.state import State
from mdpy.core.trajectory import Trajectory
from mdpy.core.ensemble import Ensemble

__all__ = [
    'Particle', 'Topology',
    'CellList', 'Grid',
    'State', 'Trajectory', 'Ensemble'
]