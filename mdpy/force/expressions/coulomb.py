from mdpy.force.nonbonded_transpiler import nonbonded_expression


@nonbonded_expression
def coulomb(pos1, pos2, charge1, charge2):
    r = distance(pos1, pos2)
    return 0.13893556595455 * charge1 * charge2 / r
