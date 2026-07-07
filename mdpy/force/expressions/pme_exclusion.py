import math

from mdpy.unit import EPSILON0
from mdpy.force.bonded_transpiler import bonded_expression
from mdpy.force.markers import param
from mdpy.force.expressions.geometry import distance

# Coulomb constant 1/(4*pi*epsilon0) in mdpy internal units (file-local).
COULOMB_CONST = 1.0 / (4.0 * math.pi * float(EPSILON0.value))


@bonded_expression(body=2)
def pme_exclusion_correction(pos1, pos2, charge1, charge2, alpha=param):
    r = distance(pos1, pos2)
    return -COULOMB_CONST * charge1 * charge2 * erf(alpha * r) / r
