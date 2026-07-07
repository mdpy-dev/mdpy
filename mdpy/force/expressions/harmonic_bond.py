from mdpy.force.bonded_transpiler import bonded_expression
from mdpy.force.expressions.geometry import distance


@bonded_expression(body=2)
def harmonic_bond(pos1, pos2, k=0.0, r0=0.0):
    r = distance(pos1, pos2)
    dr = r - r0
    return k * dr * dr
