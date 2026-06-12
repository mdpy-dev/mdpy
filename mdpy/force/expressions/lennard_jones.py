from mdpy.force.nonbonded_force import nonbonded_expression, PairParameter

from mdpy.force.nonbonded_transpiler import (
    nonbonded_expression as nonbonded_expression_ad,
)
from mdpy.force.markers import param


@nonbonded_expression
def lennard_jones(r, atom_i, atom_j, sigma_ij_pair=PairParameter(), epsilon_ij_pair=PairParameter()):
    sr = sigma_ij_pair / r
    sr6 = sr**6
    sr12 = sr6 * sr6
    dsr = sr12 - sr6
    energy = 4.0 * epsilon_ij_pair * (dsr)
    force_magnitude = -24.0 * epsilon_ij_pair * (dsr + sr12) / r
    return energy, force_magnitude


@nonbonded_expression_ad
def lennard_jones_ad(pos1, pos2, sigma=param, epsilon=param):
    r = distance(pos1, pos2)
    sr = sigma / r
    sr6 = sr * sr * sr * sr * sr * sr
    return 4.0 * epsilon * (sr6 * sr6 - sr6)
