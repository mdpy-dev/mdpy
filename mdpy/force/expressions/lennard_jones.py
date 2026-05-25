from mdpy.force.nonbonded_force import nonbonded_expression, Parameter


@nonbonded_expression
def lennard_jones(r, atom_i, atom_j, sigma_half=Parameter(), sqrt_epsilon=Parameter()):
    sigma_ij = sigma_half[atom_i] + sigma_half[atom_j]
    epsilon_ij = sqrt_epsilon[atom_i] * sqrt_epsilon[atom_j]
    inv_r = 1.0 / r
    sr = sigma_ij * inv_r
    sr6 = sr**6
    sr12 = sr6 * sr6
    dsr = sr12 - sr6
    energy = 4.0 * epsilon_ij * (dsr)
    force_magnitude = -24.0 * epsilon_ij * (dsr + sr12) * inv_r
    return energy, force_magnitude
