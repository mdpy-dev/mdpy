import pytest
from mdpy.force.ad_engine import TapeEntry, ForwardADEngine


class TestForwardADEngine:
    def test_simple_mul_chain(self):
        """_t1 = sigma * inv_dist (div by r), _t2 = _t1 * _t1, energy = _t2"""
        tape = [
            TapeEntry('r', 'distance', [], 'r'),
            TapeEntry('_t1', 'div', ['sigma', 'r'], '(sigma * inv_dist)'),
            TapeEntry('_t2', 'mul', ['_t1', '_t1'], '(_t1 * _t1)'),
        ]
        engine = ForwardADEngine()
        lines, derivs = engine.differentiate(tape)

        assert derivs['_t1'] == '_d__t1'
        assert derivs['_t2'] == '_d__t2'
        assert 'inv_dist * inv_dist' in lines[0]
        assert 'sigma' in lines[0]
        assert '2.0f' in lines[1]
        assert '_d__t1' in lines[1]

    def test_constant_has_zero_derivative(self):
        """_t5 = 4.0f * epsilon -> derivative is 0, no line emitted"""
        tape = [
            TapeEntry('r', 'distance', [], 'r'),
            TapeEntry('_t5', 'mul', ['4.0f', 'epsilon'], '(4.0f * epsilon)'),
        ]
        engine = ForwardADEngine()
        lines, derivs = engine.differentiate(tape)
        assert len(lines) == 0
        assert derivs.get('_t5', '0.0f') == '0.0f'

    def test_sub_rule(self):
        """_t7 = _t6 - _t4 -> d_t7 = _d_t6 - _d_t4"""
        tape = [
            TapeEntry('r', 'distance', [], 'r'),
            TapeEntry('_t3', 'mul', ['sigma', 'r'], '(sigma * r)'),
            TapeEntry('_t4', 'mul', ['_t3', '_t3'], '(_t3 * _t3)'),
            TapeEntry('_t6', 'mul', ['_t4', '_t4'], '(_t4 * _t4)'),
            TapeEntry('_t7', 'sub', ['_t6', '_t4'], '(_t6 - _t4)'),
        ]
        engine = ForwardADEngine()
        lines, derivs = engine.differentiate(tape)
        t7_line = [l for l in lines if '_d__t7' in l][0]
        assert '_d__t6' in t7_line
        assert '_d__t4' in t7_line
        assert '-' in t7_line

    def test_erfc_rule(self):
        """erfc(alpha_r) -> -2/sqrt(pi) * exp(-alpha_r^2) * d_alpha_r"""
        tape = [
            TapeEntry('r', 'distance', [], 'r'),
            TapeEntry('_ar', 'mul', ['alpha', 'r'], '(alpha * r)'),
            TapeEntry('_er', 'erfc', ['_ar'], 'erfcf(_ar)'),
        ]
        engine = ForwardADEngine()
        lines, derivs = engine.differentiate(tape)
        er_line = [l for l in lines if '_d__er' in l][0]
        assert 'expf' in er_line
        assert '1.1283791670955126f' in er_line


class TestDecomposeExponent:
    def test_exact_match(self):
        """Target exponent exists directly in available powers."""
        from mdpy.force.ad_engine import _decompose_exponent
        available = {1: '_t1', 2: '_t2', 3: '_t3', 6: '_t4'}
        result = _decompose_exponent(3, available)
        assert result == ['_t3']

    def test_two_way_decomposition(self):
        """5 = 3 + 2, needs two variables."""
        from mdpy.force.ad_engine import _decompose_exponent
        available = {1: '_t1', 2: '_t2', 3: '_t3', 6: '_t4'}
        result = _decompose_exponent(5, available)
        assert set(result) == {'_t3', '_t2'}
        assert len(result) == 2

    def test_three_way_decomposition(self):
        """11 = 6 + 3 + 2, needs three variables."""
        from mdpy.force.ad_engine import _decompose_exponent
        available = {1: '_t1', 2: '_t2', 3: '_t3', 6: '_t4', 12: '_t6'}
        result = _decompose_exponent(11, available)
        assert set(result) == {'_t4', '_t3', '_t2'}
        assert len(result) == 3

    def test_impossible_decomposition_returns_none(self):
        """If decomposition is impossible, return None."""
        from mdpy.force.ad_engine import _decompose_exponent
        available = {2: '_t2', 4: '_t4'}
        result = _decompose_exponent(3, available)
        assert result is None

    def test_target_zero_returns_empty(self):
        """x^0 = 1, no variables needed."""
        from mdpy.force.ad_engine import _decompose_exponent
        available = {1: '_t1', 2: '_t2'}
        result = _decompose_exponent(0, available)
        assert result == []
