from mdpy.force.nonbonded_transpiler import nonbonded_expression


@nonbonded_expression
def lennard_jones(r, sigma=0.0, epsilon=0.0):
    sigma_ratio = sigma / r
    sigma_ratio_6 = sigma_ratio ** 6
    return 4.0 * epsilon * (sigma_ratio_6 * sigma_ratio_6 - sigma_ratio_6)
