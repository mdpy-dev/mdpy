from mdpy.force.nonbonded_transpiler import nonbonded_expression


@nonbonded_expression
def coulomb(r, charge1=0.0, charge2=0.0):
    return 0.13893556595455 * charge1 * charge2 / r
