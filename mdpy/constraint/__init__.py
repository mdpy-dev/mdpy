__author__ = "Zhenyu Wei"
__maintainer__ = "Zhenyu Wei"
__email__ = "zhenyuwei99@gmail.com"
__copyright__ = "Copyright 2021-2021, Southeast University and Zhenyu Wei"
__license__ = "GPLv3"

from .constraint import Constraint

# Constants
import numpy as np
from .. import SPATIAL_DIM, env

NUM_NEIGHBOR_CELLS = 27
NEIGHBOR_CELL_TEMPLATE = np.zeros([NUM_NEIGHBOR_CELLS, SPATIAL_DIM], dtype=env.NUMPY_INT)
index = 0
for i in range(-1, 2):
    for j in range(-1, 2):
        for k in range(-1, 2):
            NEIGHBOR_CELL_TEMPLATE[index, :] = [i, j, k]
            index += 1

# Electrostatic
LONG_RANGE_SOLVER = ['PME', 'PPPM']
from .electrostatic_constraint import ElectrostaticConstraint

# Charmm
from .charmm_bond_constraint import CharmmBondConstraint
from .charmm_angle_constraint import CharmmAngleConstraint
from .charmm_dihedral_constraint import CharmmDihedralConstraint
from .charmm_improper_constraint import CharmmImproperConstraint
from .charmm_nonbonded_constraint import CharmmNonbondedConstraint

__all__ = [
    'ElectrostaticConstraint',
    'CharmmBondConstraint', 'CharmmAngleConstraint', 
    'CharmmDihedralConstraint', 'CharmmImproperConstraint', 
    'CharmmNonbondedConstraint'
]