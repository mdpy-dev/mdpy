import math


class TapeEntry:
    __slots__ = ('var_name', 'operation', 'operands', 'cuda_value', 'd_output')

    def __init__(self, var_name, operation, operands, cuda_value):
        self.var_name = var_name
        self.operation = operation
        self.operands = operands
        self.cuda_value = cuda_value
        self.d_output = None


_HELPER_OPERATIONS = frozenset({'distance', 'angle', 'dihedral'})


def _simplify(expr):
    while expr.startswith('1.0f * '):
        expr = expr[len('1.0f * '):]
    if expr.startswith('(-') and expr.endswith(')'):
        depth = 0
        simple = True
        for i, ch in enumerate(expr):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0 and i < len(expr) - 1:
                    simple = False
                    break
        if simple:
            expr = expr[1:-1]
    return expr


def _accumulate(existing, new_expr):
    if existing is None or existing == '0.0f':
        return new_expr
    if new_expr == '0.0f':
        return existing
    return f'({existing} + {new_expr})'


def _diff_add(d_out, entry):
    return [(entry.operands[0], d_out), (entry.operands[1], d_out)]


def _diff_sub(d_out, entry):
    return [(entry.operands[0], d_out), (entry.operands[1], f'-{d_out}')]


def _diff_mul(d_out, entry):
    a, b = entry.operands
    return [(a, f'{d_out} * {b}'), (b, f'{d_out} * {a}')]


def _diff_div(d_out, entry):
    a, b = entry.operands
    return [(a, f'{d_out} / {b}'), (b, f'-{d_out} * {a} / ({b} * {b})')]


def _diff_pow(d_out, entry):
    base, exp = entry.operands
    return [(base, f'{d_out} * {exp} * powf({base}, {exp}-1)')]


def _diff_sqrt(d_out, entry):
    a = entry.operands[0]
    return [(a, f'{d_out} * 0.5f / sqrtf({a})')]


def _diff_sin(d_out, entry):
    a = entry.operands[0]
    return [(a, f'{d_out} * cosf({a})')]


def _diff_cos(d_out, entry):
    a = entry.operands[0]
    return [(a, f'{d_out} * (-sinf({a}))')]


def _diff_exp(d_out, entry):
    a = entry.operands[0]
    return [(a, f'{d_out} * expf({a})')]


def _diff_log(d_out, entry):
    a = entry.operands[0]
    return [(a, f'{d_out} / {a}')]


def _diff_erf(d_out, entry):
    a = entry.operands[0]
    c = 2.0 / math.sqrt(math.pi)
    return [(a, f'{d_out} * {c}f * expf(-({a}) * ({a}))')]


def _diff_erfc(d_out, entry):
    a = entry.operands[0]
    c = 2.0 / math.sqrt(math.pi)
    return [(a, f'{d_out} * (-{c}f * expf(-({a}) * ({a})))')]


_DIFF_RULES = {
    'add': _diff_add,
    'sub': _diff_sub,
    'mul': _diff_mul,
    'div': _diff_div,
    'pow': _diff_pow,
    'sqrt': _diff_sqrt,
    'sin': _diff_sin,
    'cos': _diff_cos,
    'exp': _diff_exp,
    'log': _diff_log,
    'erf': _diff_erf,
    'erfc': _diff_erfc,
}


class ScalarADEngine:
    def differentiate(self, tape, seed_var, seed_grad='1.0f'):
        gradients = {seed_var: seed_grad}
        for entry in reversed(tape):
            if entry.var_name not in gradients:
                continue
            d_out = gradients[entry.var_name]
            if entry.operation in _HELPER_OPERATIONS:
                entry.d_output = d_out
                continue
            if entry.operation not in _DIFF_RULES:
                continue
            for operand, d_expr in _DIFF_RULES[entry.operation](d_out, entry):
                gradients[operand] = _accumulate(gradients.get(operand), _simplify(d_expr))
        return {k: _simplify(v) for k, v in gradients.items()}
