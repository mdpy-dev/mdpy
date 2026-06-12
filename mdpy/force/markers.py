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
