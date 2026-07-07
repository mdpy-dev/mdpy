"""Geometry operations for force expressions.

These objects (``distance``, ``angle``, ``dihedral``, ``distance_to_point``)
are used inside @bonded_expression / @nonbonded_expression function bodies:

    @bonded_expression(body=2)
    def harmonic_bond(p1, p2, k=param, r0=param):
        return 0.5 * k * (distance(p1, p2) - r0)**2

They are **compile-time placeholders**, not real functions: calling them at
runtime raises. The bonded/nonbonded transpilers walk the expression
function's AST, recognize Call nodes targeting these names, and emit the
corresponding CUDA geometry code from HELPER_REGISTRY.
"""


class _CompileTimeGeometry:
    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return f'<geometry:{self._name}>'

    def __call__(self, *args):
        raise RuntimeError(
            f"{self._name}() is a compile-time geometry operation used by "
            f"@bonded_expression / @nonbonded_expression decorators."
        )


distance = _CompileTimeGeometry('distance')
angle = _CompileTimeGeometry('angle')
dihedral = _CompileTimeGeometry('dihedral')
distance_to_point = _CompileTimeGeometry('distance_to_point')
