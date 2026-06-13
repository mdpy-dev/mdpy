from mdpy.force.bonded_transpiler import bonded_expression
from mdpy.force.primitives import distance


@bonded_expression(body=2)
def harmonic_bond(p1, p2, k=0.0, r0=0.0):
    r = distance(p1, p2)
    dr = r - r0
    return k * dr * dr
