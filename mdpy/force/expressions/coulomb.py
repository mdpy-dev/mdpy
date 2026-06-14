import math

from mdpy.unit import EPSILON0
from mdpy.force.nonbonded_transpiler import nonbonded_expression

# Coulomb constant 1/(4*pi*epsilon0) in mdpy internal units (file-local).
COULOMB_CONST = 1.0 / (4.0 * math.pi * float(EPSILON0.value))


@nonbonded_expression
def coulomb(pos1, pos2, charge1, charge2):
    r = distance(pos1, pos2)
    return COULOMB_CONST * charge1 * charge2 / r
