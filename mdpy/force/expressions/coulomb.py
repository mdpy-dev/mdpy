from mdpy.force.nonbonded_force import nonbonded_expression, Parameter

from mdpy.force.nonbonded_transpiler import (
    nonbonded_expression as nonbonded_expression_ad,
)


@nonbonded_expression
def coulomb(r, atom_i, atom_j, charge=Parameter()):
    qq = charge[atom_i] * charge[atom_j]
    inv_r = 1.0 / r
    energy = 0.13893556595455 * qq * inv_r
    force_magnitude = -energy * inv_r
    return energy, force_magnitude


@nonbonded_expression_ad
def coulomb_ad(pos1, pos2, charge1, charge2):
    r = distance(pos1, pos2)
    return 0.13893556595455 * charge1 * charge2 / r
