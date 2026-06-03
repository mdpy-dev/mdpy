from mdpy.force.nonbonded_force import nonbonded_expression, Parameter, Scalar


@nonbonded_expression
def screened_coulomb(r, atom_i, atom_j, charge=Parameter(), alpha=Scalar()):
    qq = charge[atom_i] * charge[atom_j]
    inv_r = 1.0 / r
    alpha_r = alpha * r
    z2 = alpha_r * alpha_r
    z4 = z2 * z2

    # Force correction: pmeCorrF(z2) = N(z2)/D(z2)
    # Denominator: FD4*z8 + FD3*z6 + FD2*z4 + FD1*z2 + FD0
    fd_a = 0.0011193462567257629232 * z4 + 0.11583842382862377919
    fd_b = 0.014866955030185295499 * z4 + 0.50736591960530292870
    fd_c = fd_a * z4 + 1.0
    fd_d = fd_b * z2 + fd_c
    inv_fd = 1.0 / fd_d

    # Numerator: FN6*z12 + FN5*z10 + ... + FN0
    fn_a = -1.7357322914161492954e-8 * z4 - 5.3401640219807709149e-5
    fn_b = 1.4703624142580877519e-6 * z4 + 1.0054721316683106153e-3
    fn_c = fn_a * z4 - 1.927831726488838059e-2
    fn_d = fn_b * z4 + 6.9670166153766424023e-2
    fn_e = fn_c * z4 - 0.75225204789749321333
    fn_f = fn_d * z2 + fn_e

    corr = fn_f * inv_fd

    # Force: dV/dr = -COULOMB * qq * (inv_r^2 + alpha^3 * r * corr)
    alpha3 = alpha * alpha * alpha
    force_magnitude = -0.13893556595455 * qq * (inv_r * inv_r + alpha3 * r * corr)

    # Energy: V = COULOMB * qq * erfc(alpha*r) / r = COULOMB * qq * (1 - erf(alpha*r)) / r
    erf_val = erf(alpha_r)
    energy = 0.13893556595455 * qq * (1.0 - erf_val) * inv_r

    return energy, force_magnitude
