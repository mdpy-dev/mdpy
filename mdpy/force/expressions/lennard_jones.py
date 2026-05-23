from mdpy.force.nonbonded_force import nonbonded_expression, Parameter
from math import sqrt


@nonbonded_expression
def lennard_jones(r, atom_i, atom_j, sigma=Parameter(), epsilon=Parameter()):
    sigma_ij = 0.5 * (sigma[atom_i] + sigma[atom_j])
    epsilon_ij = sqrt(epsilon[atom_i] * epsilon[atom_j])
    sr = sigma_ij / r
    sr6 = sr ** 6
    sr12 = sr6 * sr6
    energy = 4.0 * epsilon_ij * (sr12 - sr6)
    force_magnitude = -24.0 * epsilon_ij * (2.0 * sr12 - sr6) / r
    return energy, force_magnitude
