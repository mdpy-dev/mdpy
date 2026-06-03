from mdpy.force.nonbonded_force import nonbonded_expression, Parameter, Scalar


@nonbonded_expression
def screened_coulomb(r, atom_i, atom_j, charge=Parameter(), alpha=Scalar()):
    qq = charge[atom_i] * charge[atom_j]
    alpha_r = alpha * r
    erfc_ar = erfc(alpha_r)
    gauss = exp(-alpha_r * alpha_r)
    inv_r = 1.0 / r
    energy = 0.13893556595455 * qq * erfc_ar * inv_r
    force_magnitude = -0.13893556595455 * qq * (
        erfc_ar * inv_r * inv_r
        + 2.0 * alpha * gauss * inv_r / 1.772453850905516
    )
    return energy, force_magnitude
