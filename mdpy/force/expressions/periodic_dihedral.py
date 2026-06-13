from mdpy.force.bonded_transpiler import bonded_expression
from mdpy.force.primitives import dihedral


@bonded_expression(body=4)
def periodic_dihedral(p1, p2, p3, p4, k=0.0, n=0.0, delta=0.0):
    phi = dihedral(p1, p2, p3, p4)
    return k * (1.0 + cos(n * phi - delta))
