from mdpy.force.bonded_expression import bonded_expression


@bonded_expression(body=2)
def harmonic_bond(r, k=0.0, r0=0.0):
    dr = r - r0
    return k * dr * dr, 2.0 * k * dr


@bonded_expression(body=3)
def charmm_angle(theta, r13, k=0.0, theta0=0.0, k_ub=0.0, r_ub=0.0):
    dt = theta - theta0
    e = k * dt * dt
    f_theta = 2.0 * k * dt
    dr = r13 - r_ub
    e_ub = k_ub * dr * dr
    f_r13 = 2.0 * k_ub * dr
    return e + e_ub, f_theta, f_r13


@bonded_expression(body=4)
def periodic_dihedral(phi, k=0.0, n=0.0, delta=0.0):
    return k * (1.0 + cos(n * phi - delta)), -k * n * sin(n * phi - delta)


@bonded_expression(body=4)
def harmonic_improper(psi, k=0.0, psi0=0.0):
    dp = psi - psi0
    return k * dp * dp, 2.0 * k * dp
