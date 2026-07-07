from mdpy.force.bonded_transpiler import bonded_expression
from mdpy.force.markers import param, point
from mdpy.force.expressions.geometry import distance_to_point


@bonded_expression(body=1)
def position_restraint(pos1, ref=point, k=param):
    r = distance_to_point(pos1, ref)
    return k * r * r
