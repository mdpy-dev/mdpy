from mdpy.force.nonbonded_transpiler import nonbonded_expression
from mdpy.force.markers import param


@nonbonded_expression
def lennard_jones(pos1, pos2, sigma=param, epsilon=param):
    r = distance(pos1, pos2)
    sigma_ratio = sigma / r
    sigma_ratio_6 = sigma_ratio ** 6
    return 4.0 * epsilon * (sigma_ratio_6 * sigma_ratio_6 - sigma_ratio_6)
