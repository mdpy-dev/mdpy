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


_TWO_OVER_SQRT_PI = 1.1283791670955126


def _fwd_add(operands, d_operands):
    parts = [d for d in d_operands if d != '0.0f']
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return f'({parts[0]} + {parts[1]})'


def _fwd_sub(operands, d_operands):
    a, b = operands
    da, db = d_operands
    if da == '0.0f' and db == '0.0f':
        return None
    if db == '0.0f':
        return da
    if da == '0.0f':
        return f'(-{db})'
    return f'({da} - {db})'


def _fwd_mul(operands, d_operands):
    a, b = operands
    da, db = d_operands
    if a == b:
        if da == '0.0f':
            return None
        return f'(2.0f * {a} * {da})'
    if da == '0.0f' and db == '0.0f':
        return None
    if da == '0.0f':
        return f'({a} * {db})'
    if db == '0.0f':
        return f'({da} * {b})'
    return f'({da} * {b} + {a} * {db})'


def _fwd_div(operands, d_operands):
    a, b = operands
    da, db = d_operands
    if b == 'r':
        if da == '0.0f':
            return f'(-{a} * inv_dist * inv_dist)'
        return f'({da} * inv_dist - {a} * inv_dist * inv_dist)'
    if db == '0.0f':
        return f'({da} / {b})'
    return f'(({da} * {b} - {a} * {db}) / ({b} * {b}))'


def _fwd_erfc(operands, d_operands):
    a = operands[0]
    da = d_operands[0]
    if da == '0.0f':
        return None
    return f'(-{_TWO_OVER_SQRT_PI}f * expf(-({a}) * ({a})) * {da})'


def _fwd_erf(operands, d_operands):
    a = operands[0]
    da = d_operands[0]
    if da == '0.0f':
        return None
    return f'({_TWO_OVER_SQRT_PI}f * expf(-({a}) * ({a})) * {da})'


def _fwd_pow(operands, d_operands):
    base, exp = operands
    da = d_operands[0]
    if da == '0.0f':
        return None
    return f'({exp} * powf({base}, {exp} - 1) * {da})'


def _fwd_exp(operands, d_operands):
    a = operands[0]
    da = d_operands[0]
    if da == '0.0f':
        return None
    return f'(expf({a}) * {da})'


def _fwd_sqrt(operands, d_operands):
    a = operands[0]
    da = d_operands[0]
    if da == '0.0f':
        return None
    return f'(0.5f * {da} / sqrtf({a}))'


def _fwd_log(operands, d_operands):
    a = operands[0]
    da = d_operands[0]
    if da == '0.0f':
        return None
    return f'({da} / {a})'


def _fwd_sin(operands, d_operands):
    a = operands[0]
    da = d_operands[0]
    if da == '0.0f':
        return None
    return f'(cosf({a}) * {da})'


def _fwd_cos(operands, d_operands):
    a = operands[0]
    da = d_operands[0]
    if da == '0.0f':
        return None
    return f'((-sinf({a})) * {da})'


_FWD_RULES = {
    'add': _fwd_add,
    'sub': _fwd_sub,
    'mul': _fwd_mul,
    'div': _fwd_div,
    'pow': _fwd_pow,
    'erfc': _fwd_erfc,
    'erf': _fwd_erf,
    'exp': _fwd_exp,
    'sqrt': _fwd_sqrt,
    'log': _fwd_log,
    'sin': _fwd_sin,
    'cos': _fwd_cos,
}


class ForwardADEngine:
    """Forward-mode AD for scalar-to-scalar functions.

    Computes d(each_tape_var)/d(seed_var) by propagating derivatives
    forward through the tape. Each derivative is a named CUDA variable,
    preventing the expression blowup that plagues reverse-mode string AD.
    """

    def differentiate(self, tape, seed_vars=None):
        """Process tape in forward order, computing derivatives.

        Args:
            tape: list of TapeEntry in forward evaluation order.
            seed_vars: dict mapping variable names to their derivative
                       expressions w.r.t. the independent variable.
                       Default: {'r': '1.0f'}

        Returns:
            (grad_lines, derivs) where grad_lines is a list of CUDA
            code strings (e.g. 'float _d__t4 = (2.0f * _t3 * _d__t3);')
            and derivs maps each tape variable name to its derivative
            variable name (or '0.0f' if derivative is zero).
        """
        if seed_vars is None:
            seed_vars = {'r': '1.0f'}

        derivs = dict(seed_vars)
        grad_lines = []

        for entry in tape:
            if entry.operation in _HELPER_OPERATIONS:
                derivs[entry.var_name] = '1.0f'
                continue

            rule = _FWD_RULES.get(entry.operation)
            if rule is None:
                continue

            operands = entry.operands
            d_operands = [derivs.get(op, '0.0f') for op in operands]

            deriv_expr = rule(operands, d_operands)

            if deriv_expr is not None:
                name = f'_d_{entry.var_name}'
                grad_lines.append(f'float {name} = {deriv_expr};')
                derivs[entry.var_name] = name

        return grad_lines, derivs
