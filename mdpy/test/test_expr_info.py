import pytest
from mdpy.force.expr_info import classify_arguments, ExprInfo
from mdpy.force.markers import param, scalar


class TestClassifyArguments:
    def test_bonded_simple(self):
        def harmonic_bond(pos1, pos2, k=param, r0=param):
            pass
        info = classify_arguments(harmonic_bond, body=2)
        assert info.positions == ['pos1', 'pos2']
        assert info.body == 2
        assert info.params == ['k', 'r0']
        assert info.scalars == []
        assert info.per_particle == {}

    def test_angle_with_per_particle(self):
        def something(pos1, pos2, pos3, charge1, charge2, k=param, theta0=param):
            pass
        info = classify_arguments(something, body=3)
        assert info.positions == ['pos1', 'pos2', 'pos3']
        assert info.body == 3
        assert info.per_particle == {'charge1': 'charge', 'charge2': 'charge'}
        assert info.params == ['k', 'theta0']
        assert info.scalars == []

    def test_nonbonded_with_scalar(self):
        def screened(pos1, pos2, charge1, charge2, alpha=scalar):
            pass
        info = classify_arguments(screened, body=2)
        assert info.positions == ['pos1', 'pos2']
        assert info.per_particle == {'charge1': 'charge', 'charge2': 'charge'}
        assert info.params == []
        assert info.scalars == ['alpha']

    def test_dihedral(self):
        def periodic(pos1, pos2, pos3, pos4, k=param, n=param, delta=param):
            pass
        info = classify_arguments(periodic, body=4)
        assert info.positions == ['pos1', 'pos2', 'pos3', 'pos4']
        assert info.params == ['k', 'n', 'delta']
        assert info.per_particle == {}

    def test_name_stripping_charge(self):
        def f(pos1, pos2, charge1, charge2):
            pass
        info = classify_arguments(f, body=2)
        assert info.per_particle == {'charge1': 'charge', 'charge2': 'charge'}

    def test_name_stripping_with_underscore(self):
        def f(pos1, x0_1, k=param):
            pass
        info = classify_arguments(f, body=1)
        assert info.per_particle == {'x0_1': 'x0'}

    def test_no_params(self):
        def f(pos1, pos2, charge1, charge2):
            pass
        info = classify_arguments(f, body=2)
        assert info.params == []
        assert info.scalars == []

    def test_nb14_mixed(self):
        def nb14(pos1, pos2, charge1, charge2, sigma=param, epsilon=param, charge_scale=param):
            pass
        info = classify_arguments(nb14, body=2)
        assert info.per_particle == {'charge1': 'charge', 'charge2': 'charge'}
        assert info.params == ['sigma', 'epsilon', 'charge_scale']
        assert info.scalars == []

    def test_unique_particle_properties(self):
        def f(pos1, pos2, charge1, charge2):
            pass
        info = classify_arguments(f, body=2)
        assert info.particle_properties == {'charge'}
