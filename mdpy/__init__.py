__author__ = "Zhenyu Wei"
__maintainer__ = "Zhenyu Wei"
__email__ = "zhenyuwei99@gmail.com"
__copyright__ = "Copyright 2021-2021, Southeast University and Zhenyu Wei"
__license__ = "GPLv3"

# Constant
SPATIAL_DIM = 3

# Precision setting
import numpy as np
import numba as nb
NUMPY_FLOAT = np.float32
NUMBA_FLOAT= nb.float32
NUMPY_INT = np.int32
NUMBA_INT = nb.int32

# Import
import mdpy.unit as unit
import mdpy.math as math
import mdpy.core as core
import mdpy.file as file
import mdpy.constraint as constraint
import mdpy.forcefield as forcefield
from mdpy.ensemble import Ensemble
import mdpy.integrator as integrator
from mdpy.simulation import Simulation
import mdpy.dumper as dumper