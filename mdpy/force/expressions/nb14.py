from mdpy.force.bonded_transpiler import bonded_expression
from mdpy.force.primitives import param, distance


@bonded_expression(body=2)
def nb14_lj_coulomb(pos1, pos2, charge1, charge2, sigma=param, epsilon=param):
    r = distance(pos1, pos2)
    e_coul = 0.13893556595455 * charge1 * charge2 / r
    sr = sigma / r
    sr6 = sr * sr * sr * sr * sr * sr
    e_lj = 4.0 * epsilon * (sr6 * sr6 - sr6)
    return e_coul + e_lj
