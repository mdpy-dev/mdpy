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


class TestPowerChainOptimization:
    def test_lj_power_chain_skips_intermediate_gradients(self):
        """LJ tape: _t1=sr, _t2=sr^2, _t3=sr^3, _t4=sr^6.
        Power rule should eliminate _d__t2 and _d__t3."""
        tape = [
            TapeEntry('r', 'distance', [], 'r'),
            TapeEntry('_t1', 'div', ['sigma', 'r'], '(sigma * inv_dist)'),
            TapeEntry('_t2', 'mul', ['_t1', '_t1'], '(_t1 * _t1)'),
            TapeEntry('_t3', 'mul', ['_t1', '_t2'], '(_t1 * _t2)'),
            TapeEntry('_t4', 'mul', ['_t3', '_t3'], '(_t3 * _t3)'),
            TapeEntry('_t5', 'mul', ['4.0f', 'epsilon'], '(4.0f * epsilon)'),
            TapeEntry('_t6', 'mul', ['_t4', '_t4'], '(_t4 * _t4)'),
            TapeEntry('_t7', 'sub', ['_t6', '_t4'], '(_t6 - _t4)'),
            TapeEntry('_t8', 'mul', ['_t5', '_t7'], '(_t5 * _t7)'),
        ]
        engine = ForwardADEngine()
        lines, derivs = engine.differentiate(tape)

        all_text = '\n'.join(lines)

        # d_t1 must exist
        assert '_d__t1' in all_text
        assert derivs['_t1'] == '_d__t1'

        # d_t2 should NOT exist (dead code, eliminated by power rule)
        assert '_d__t2' not in all_text

        # d_t3 should NOT exist (dead code, eliminated by power rule)
        assert '_d__t3' not in all_text

        # d_t4 must exist, using power rule (6.0f * _t3 * _t2 * _d__t1)
        d_t4_line = [l for l in lines if '_d__t4' in l and '=' in l][0]
        assert '6.0f' in d_t4_line
        assert '_t3' in d_t4_line
        assert '_t2' in d_t4_line
        assert '_d__t1' in d_t4_line
        # Must NOT reference _d__t3 (that's the whole point)
        assert '_d__t3' not in d_t4_line

        # d_t6 must exist, using power rule (12.0f * _t4 * _t3 * _t2 * _d__t1)
        d_t6_line = [l for l in lines if '_d__t6' in l and '=' in l][0]
        assert '12.0f' in d_t6_line
        assert '_t4' in d_t6_line
        assert '_d__t1' in d_t6_line
        assert '_d__t4' not in d_t6_line

        # d_t7 = d_t6 - d_t4 (subtraction rule, unchanged)
        d_t7_line = [l for l in lines if '_d__t7' in l and '=' in l][0]
        assert '_d__t6' in d_t7_line
        assert '_d__t4' in d_t7_line

        # d_t8 = _t5 * d_t7 (mul with one zero-derivative operand)
        assert derivs['_t8'] is not None
        d_t8_line = [l for l in lines if '_d__t8' in l and '=' in l][0]
        assert '_d__t7' in d_t8_line

    def test_non_power_mul_keeps_product_rule(self):
        """mul of variables with different bases should still use product rule."""
        tape = [
            TapeEntry('r', 'distance', [], 'r'),
            TapeEntry('_a', 'div', ['sigma', 'r'], '(sigma * inv_dist)'),
            TapeEntry('_b', 'div', ['epsilon', 'r'], '(epsilon * inv_dist)'),
            TapeEntry('_c', 'mul', ['_a', '_b'], '(_a * _b)'),
        ]
        engine = ForwardADEngine()
        lines, derivs = engine.differentiate(tape)

        c_line = [l for l in lines if '_d__c' in l and '=' in l][0]
        # Product rule: _d__a * _b + _a * _d__b
        assert '_d__a' in c_line
        assert '_d__b' in c_line

    def test_simple_square_keeps_squaring_rule(self):
        """t = a * a (simple square) should still use 2*a*da, not power rule."""
        tape = [
            TapeEntry('r', 'distance', [], 'r'),
            TapeEntry('_t1', 'div', ['sigma', 'r'], '(sigma * inv_dist)'),
            TapeEntry('_t2', 'mul', ['_t1', '_t1'], '(_t1 * _t1)'),
        ]
        engine = ForwardADEngine()
        lines, derivs = engine.differentiate(tape)

        t2_line = [l for l in lines if '_d__t2' in l and '=' in l][0]
        # Squaring: 2.0f * _t1 * _d__t1 (not power rule with decomposition)
        assert '2.0f' in t2_line
        assert '_t1' in t2_line
        assert '_d__t1' in t2_line
