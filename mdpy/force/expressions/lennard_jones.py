from mdpy.force.nonbonded_force import nonbonded_expression, PairParameter


@nonbonded_expression
def lennard_jones(r, atom_i, atom_j, sigma_ij_pair=PairParameter(), epsilon_ij_pair=PairParameter()):
    sr = sigma_ij_pair / r
    sr6 = sr**6
    sr12 = sr6 * sr6
    dsr = sr12 - sr6
    energy = 4.0 * epsilon_ij_pair * (dsr)
    force_magnitude = -24.0 * epsilon_ij_pair * (dsr + sr12) / r
    return energy, force_magnitude
