import math
import pytest
from mdpy.force.ad_engine import TapeEntry, ScalarADEngine


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
