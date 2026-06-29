from mdpy.force.bonded_transpiler import bonded_expression
from mdpy.force.primitives import param, point, distance_to_point


@bonded_expression(body=1)
def position_restraint(p1, ref=point, k=param):
    r = distance_to_point(p1, ref)
    return k * r * r
