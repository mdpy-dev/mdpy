from mdpy.force.bonded_transpiler import bonded_expression
from mdpy.force.primitives import distance, angle


@bonded_expression(body=3)
def charmm_angle(pos1, pos2, pos3, k=0.0, theta0=0.0, k_ub=0.0, r_ub=0.0):
    theta = angle(pos1, pos2, pos3)
    dt = theta - theta0
    e_angle = k * dt * dt
    r13 = distance(pos1, pos3)
    dr13 = r13 - r_ub
    e_ub = k_ub * dr13 * dr13
    return e_angle + e_ub
