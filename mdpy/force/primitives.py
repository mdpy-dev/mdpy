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


class _HelperPrimitive:
    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return f'<helper:{self._name}>'

    def __call__(self, *args):
        raise RuntimeError(
            f"{self._name}() is a compile-time primitive used by "
            f"@bonded_expression / @nonbonded_expression decorators."
        )

distance = _HelperPrimitive('distance')
angle = _HelperPrimitive('angle')
dihedral = _HelperPrimitive('dihedral')
