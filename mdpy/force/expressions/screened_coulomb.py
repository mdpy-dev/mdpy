from mdpy.force.nonbonded_transpiler import nonbonded_expression
from mdpy.force.markers import scalar as scalar_marker


@nonbonded_expression
def screened_coulomb(pos1, pos2, charge1, charge2, alpha=scalar_marker):
    r = distance(pos1, pos2)
    return 0.13893556595455 * charge1 * charge2 * erfc(alpha * r) / r
