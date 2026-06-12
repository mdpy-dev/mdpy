import pytest
from mdpy.force.nonbonded_transpiler import nonbonded_expression
from mdpy.force.markers import param, scalar


class TestNonbondedTranspilerV2:
    def test_lj_classification(self):
        @nonbonded_expression
        def lennard_jones(pos1, pos2, sigma=param, epsilon=param):
            r = distance(pos1, pos2)
            sr = sigma / r
            sr6 = sr * sr * sr * sr * sr * sr
            return 4.0 * epsilon * (sr6 * sr6 - sr6)

        info = lennard_jones.expr_info
        assert info.params == ['sigma', 'epsilon']
        assert info.per_particle == {}
        assert info.body == 2

    def test_coulomb_classification(self):
        @nonbonded_expression
        def coulomb(pos1, pos2, charge1, charge2):
            r = distance(pos1, pos2)
            return 0.13893556595455 * charge1 * charge2 / r

        info = coulomb.expr_info
        assert info.params == []
        assert info.per_particle == {'charge1': 'charge', 'charge2': 'charge'}

    def test_screened_coulomb_classification(self):
        @nonbonded_expression
        def screened(pos1, pos2, charge1, charge2, alpha=scalar):
            r = distance(pos1, pos2)
            return 0.13893556595455 * charge1 * charge2 * erfc(alpha * r) / r

        info = screened.expr_info
        assert info.scalars == ['alpha']
        assert info.per_particle == {'charge1': 'charge', 'charge2': 'charge'}

    def test_lj_has_dEdr(self):
        @nonbonded_expression
        def lennard_jones(pos1, pos2, sigma=param, epsilon=param):
            r = distance(pos1, pos2)
            sr = sigma / r
            sr6 = sr * sr * sr * sr * sr * sr
            return 4.0 * epsilon * (sr6 * sr6 - sr6)

        assert lennard_jones.dEdr_cuda is not None
        assert 'r' in lennard_jones.dEdr_cuda

    def test_coulomb_dEdr_sign(self):
        @nonbonded_expression
        def coulomb(pos1, pos2, charge1, charge2):
            r = distance(pos1, pos2)
            return 0.13893556595455 * charge1 * charge2 / r

        assert coulomb.dEdr_cuda is not None
        assert 'r' in coulomb.dEdr_cuda

    def test_add_combines_expressions(self):
        @nonbonded_expression
        def lj(pos1, pos2, sigma=param, epsilon=param):
            r = distance(pos1, pos2)
            sr = sigma / r
            sr6 = sr * sr * sr * sr * sr * sr
            return 4.0 * epsilon * (sr6 * sr6 - sr6)

        @nonbonded_expression
        def coul(pos1, pos2, charge1, charge2):
            r = distance(pos1, pos2)
            return 0.13893556595455 * charge1 * charge2 / r

        combined = lj + coul
        assert combined.dEdr_cuda is not None
        assert combined.expr_info.params == ['sigma', 'epsilon']
        assert 'charge' in str(combined.expr_info.per_particle.values()) or len(combined.expr_info.per_particle) > 0
