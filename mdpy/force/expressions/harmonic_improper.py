from mdpy.force.bonded_transpiler import bonded_expression
from mdpy.force.primitives import dihedral


@bonded_expression(body=4)
def harmonic_improper(p1, p2, p3, p4, k=0.0, psi0=0.0):
    psi = dihedral(p1, p2, p3, p4)
    dp = psi - psi0
    return k * dp * dp
