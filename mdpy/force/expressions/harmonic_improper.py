from mdpy.force.bonded_transpiler import bonded_expression
from mdpy.force.expressions.geometry import dihedral


@bonded_expression(body=4)
def harmonic_improper(pos1, pos2, pos3, pos4, k=0.0, psi0=0.0):
    psi = dihedral(pos1, pos2, pos3, pos4)
    dp = psi - psi0
    return k * dp * dp
