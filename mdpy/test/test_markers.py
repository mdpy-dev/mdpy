import pytest
from mdpy.force.markers import param, scalar


class TestMarkers:
    def test_param_identity(self):
        from mdpy.force.markers import param as p2
        assert param is p2

    def test_scalar_identity(self):
        from mdpy.force.markers import scalar as s2
        assert scalar is s2

    def test_param_repr(self):
        assert repr(param) == "param"

    def test_scalar_repr(self):
        assert repr(scalar) == "scalar"

    def test_param_is_not_scalar(self):
        assert param is not scalar

    def test_param_bool(self):
        assert bool(param) is True

    def test_scalar_bool(self):
        assert bool(scalar) is True
