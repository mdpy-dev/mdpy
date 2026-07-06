from mdpy.force.bonded_transpiler import bonded_expression
from mdpy.force.primitives import dihedral


@bonded_expression(body=4)
def periodic_dihedral(pos1, pos2, pos3, pos4, k=0.0, n=0.0, delta=0.0):
    phi = dihedral(pos1, pos2, pos3, pos4)
    return k * (1.0 + cos(n * phi - delta))
