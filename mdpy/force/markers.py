"""Parameter-classification markers for the force-expression transpiler.

These sentinel objects are used as default values in @bonded_expression /
@nonbonded_expression function signatures to tell the transpiler how to
classify each parameter:

    @bonded_expression(body=2)
    def harmonic_bond(p1, p2, k=param, r0=param):  # k, r0 are per-term params
        ...

The transpiler compares defaults by identity (``default is param``), so the
imported marker must be the exact object defined here. Never instantiated
directly by user code.
"""


class _Marker:
    __slots__ = ('_name',)

    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return self._name

    def __bool__(self):
        return True


param = _Marker("param")
scalar = _Marker("scalar")
point = _Marker("point")
