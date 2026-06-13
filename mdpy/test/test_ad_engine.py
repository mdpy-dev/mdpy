import math
import pytest
from mdpy.force.ad_engine import TapeEntry, ScalarADEngine, ForwardADEngine


class TestScalarAD:
    def setup_method(self):
        self.engine = ScalarADEngine()

    def test_constant(self):
        tape = [
            TapeEntry('x', 'const', [], '3.0f'),
            TapeEntry('y', 'mul', ['x', 'x'], '(x * x)'),
        ]
        grads = self.engine.differentiate(tape, 'y')
        assert grads['x'] == '(x + x)'

    def test_x_squared_gradient(self):
        tape = [
            TapeEntry('x', 'const', [], 'x_val'),
            TapeEntry('y', 'pow', ['x', 'n'], 'powf(x, n)'),
        ]
        grads = self.engine.differentiate(tape, 'y', seed_grad='1.0f')
        assert grads['x'] == 'n * powf(x, n-1)'

    def test_add_gradient(self):
        tape = [
            TapeEntry('a', 'const', [], 'a_val'),
            TapeEntry('b', 'const', [], 'b_val'),
            TapeEntry('c', 'add', ['a', 'b'], '(a + b)'),
        ]
        grads = self.engine.differentiate(tape, 'c')
        assert grads['a'] == '1.0f'
        assert grads['b'] == '1.0f'

    def test_mul_gradient(self):
        tape = [
            TapeEntry('a', 'const', [], 'a_val'),
            TapeEntry('b', 'const', [], 'b_val'),
            TapeEntry('c', 'mul', ['a', 'b'], '(a * b)'),
        ]
        grads = self.engine.differentiate(tape, 'c')
        assert grads['a'] == 'b'
        assert grads['b'] == 'a'

    def test_chain_rule(self):
        tape = [
            TapeEntry('r', 'const', [], 'r_val'),
            TapeEntry('dr', 'sub', ['r', 'r0'], '(r - r0)'),
            TapeEntry('dr2', 'mul', ['dr', 'dr'], '(dr * dr)'),
            TapeEntry('energy', 'mul', ['k', 'dr2'], '(k * dr2)'),
        ]
        grads = self.engine.differentiate(tape, 'energy')
        assert 'k' in grads
        assert 'dr' in grads
        assert 'r' in grads

    def test_sin_gradient(self):
        tape = [
            TapeEntry('phi', 'const', [], 'phi_val'),
            TapeEntry('s', 'sin', ['phi'], 'sinf(phi)'),
        ]
        grads = self.engine.differentiate(tape, 's')
        assert grads['phi'] == 'cosf(phi)'

    def test_cos_gradient(self):
        tape = [
            TapeEntry('phi', 'const', [], 'phi_val'),
            TapeEntry('c', 'cos', ['phi'], 'cosf(phi)'),
        ]
        grads = self.engine.differentiate(tape, 'c')
        assert grads['phi'] == '-sinf(phi)'

    def test_exp_gradient(self):
        tape = [
            TapeEntry('x', 'const', [], 'x_val'),
            TapeEntry('y', 'exp', ['x'], 'expf(x)'),
        ]
        grads = self.engine.differentiate(tape, 'y')
        assert grads['x'] == 'expf(x)'

    def test_log_gradient(self):
        tape = [
            TapeEntry('x', 'const', [], 'x_val'),
            TapeEntry('y', 'log', ['x'], 'logf(x)'),
        ]
        grads = self.engine.differentiate(tape, 'y')
        assert grads['x'] == '1.0f / x'

    def test_erf_gradient(self):
        tape = [
            TapeEntry('x', 'const', [], 'x_val'),
            TapeEntry('y', 'erf', ['x'], 'erff(x)'),
        ]
        grads = self.engine.differentiate(tape, 'y')
        assert 'expf' in grads['x']
        assert 'x' in grads['x']

    def test_div_gradient(self):
        tape = [
            TapeEntry('a', 'const', [], 'a_val'),
            TapeEntry('b', 'const', [], 'b_val'),
            TapeEntry('c', 'div', ['a', 'b'], '(a / b)'),
        ]
        grads = self.engine.differentiate(tape, 'c')
        assert grads['a'] == '1.0f / b'
        assert 'a' in grads['b']

    def test_sub_gradient(self):
        tape = [
            TapeEntry('a', 'const', [], 'a_val'),
            TapeEntry('b', 'const', [], 'b_val'),
            TapeEntry('c', 'sub', ['a', 'b'], '(a - b)'),
        ]
        grads = self.engine.differentiate(tape, 'c')
        assert grads['a'] == '1.0f'
        assert grads['b'] == '-1.0f'

    def test_sqrt_gradient(self):
        tape = [
            TapeEntry('x', 'const', [], 'x_val'),
            TapeEntry('y', 'sqrt', ['x'], 'sqrtf(x)'),
        ]
        grads = self.engine.differentiate(tape, 'y')
        assert grads['x'] == '0.5f / sqrtf(x)'

    def test_helper_not_differentiated(self):
        tape = [
            TapeEntry('r', 'distance', ['pos1', 'pos2'], 'r_val'),
            TapeEntry('energy', 'mul', ['k', 'r'], '(k * r)'),
        ]
        grads = self.engine.differentiate(tape, 'energy')
        assert 'r' in grads
        assert tape[0].d_output is not None


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
