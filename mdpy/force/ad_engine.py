class TapeEntry:
    __slots__ = ('var_name', 'operation', 'operands', 'cuda_value', 'd_output')

    def __init__(self, var_name, operation, operands, cuda_value):
        self.var_name = var_name
        self.operation = operation
        self.operands = operands
        self.cuda_value = cuda_value
        self.d_output = None


_HELPER_OPERATIONS = frozenset({'distance', 'angle', 'dihedral'})


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


def _decompose_exponent(target_exp, available_powers):
    """Decompose base^target_exp into a product of known tape variables.

    Given a set of available powers {exponent: var_name}, find a subset
    of exponents that sum to target_exp. Returns a list of variable names
    whose product equals base^target_exp, or None if impossible.

    Uses greedy decomposition: sort exponents descending, greedily subtract.

    Args:
        target_exp: The desired exponent (e.g., 5 for base^5).
        available_powers: dict mapping exponent -> variable name.
            e.g., {1: '_t1', 2: '_t2', 3: '_t3', 6: '_t4'}

    Returns:
        List of variable names (e.g., ['_t3', '_t2']) or None.
    """
    if target_exp == 0:
        return []
    if target_exp < 0:
        return None

    result = []
    remaining = target_exp
    for exp in sorted(available_powers.keys(), reverse=True):
        while remaining >= exp and exp > 0:
            result.append(available_powers[exp])
            remaining -= exp

    if remaining == 0:
        return result
    return None


class ForwardADEngine:
    """Forward-mode AD for scalar-to-scalar functions.

    Computes d(each_tape_var)/d(seed_var) by propagating derivatives
    forward through the tape. Each derivative is a named CUDA variable,
    preventing the expression blowup that plagues reverse-mode string AD.
    """

    def differentiate(self, tape, seed_vars=None, prefix='_d_'):
        """Process tape in forward order, computing derivatives.

        For power chains (e.g., x -> x^2 -> x^3 -> x^6 -> x^12), uses the
        power rule d(x^n)/dr = n * x^(n-1) * dx instead of the product rule.
        This produces fewer intermediate gradient variables, reducing register
        pressure in the generated CUDA kernel.

        Intermediate power-chain nodes whose derivatives are never referenced
        by a downstream operation are not emitted at all (lazy emission).

        Args:
            tape: list of TapeEntry in forward evaluation order.
            seed_vars: dict mapping variable names to their derivative
                       expressions w.r.t. the independent variable.
                       If None, all helper operations are auto-seeded
                       to '1.0f' (useful for single-helper expressions).
                       If provided explicitly, only the specified helpers
                       are seeded — other helpers get derivative '0.0f'.
            prefix: prefix for generated derivative variable names.
                    Default '_d_' for backward compatibility.

        Returns:
            (grad_lines, derivs) where grad_lines is a list of CUDA
            code strings (e.g. 'float _d__t4 = (6.0f * _t3 * _t2 * _d__t1);')
            and derivs maps each tape variable name to its derivative
            variable name (or '0.0f' if derivative is zero).
        """
        auto_seed_helpers = seed_vars is None
        if seed_vars is None:
            seed_vars = {}

        derivs = dict(seed_vars)
        grad_lines = []

        # Lazy emission: power-chain gradient expressions are stored here
        # and only appended to grad_lines when referenced by a downstream
        # operation. Unreferenced intermediates are never emitted.
        pending_grads = {}

        # Power chain tracking
        # power_map: var_name -> (base_var, exponent)
        # power_vars: (base_var, exponent) -> var_name  (reverse lookup)
        power_map = {}
        power_vars = {}

        def flush_if_pending(name):
            """Emit a pending gradient line if it hasn't been emitted yet."""
            if name in pending_grads:
                expr = pending_grads.pop(name)
                grad_lines.append(f'float {name} = {expr};')

        for entry in tape:
            if entry.operation in _HELPER_OPERATIONS:
                if auto_seed_helpers and entry.var_name not in derivs:
                    derivs[entry.var_name] = '1.0f'
                continue

            rule = _FWD_RULES.get(entry.operation)
            if rule is None:
                continue

            operands = entry.operands
            d_operands = [derivs.get(op, '0.0f') for op in operands]

            # --- Power chain detection ---
            # Must happen BEFORE flushing d_operands, because a power-chain
            # mul uses d_base (not d_operands). Flushing d_operands here
            # would prematurely emit gradients that the power rule bypasses.
            if entry.operation == 'mul':
                a, b = operands
                pa = power_map.get(a)
                pb = power_map.get(b)

                detected_power = None

                if a == b:
                    # Squaring: t = a * a
                    if pa is not None:
                        detected_power = (pa[0], pa[1] * 2)
                    elif derivs.get(a, '0.0f') != '0.0f':
                        detected_power = (a, 2)
                elif pa is not None and pb is not None and pa[0] == pb[0]:
                    # Same base: a^p1 * b^p2 = base^(p1+p2)
                    detected_power = (pa[0], pa[1] + pb[1])
                elif pa is not None and pa[0] == b:
                    # a^p * a = a^(p+1)
                    detected_power = (b, pa[1] + 1)
                elif pb is not None and pb[0] == a:
                    # a * a^p = a^(p+1)
                    detected_power = (a, pb[1] + 1)

                if detected_power is not None:
                    base, exp = detected_power
                    power_map[entry.var_name] = detected_power
                    power_vars[detected_power] = entry.var_name

                    # Try power rule for gradient
                    d_base = derivs.get(base, '0.0f')
                    flush_if_pending(d_base)

                    if d_base != '0.0f' and exp > 1:
                        # Find base^(exp-1) variables via decomposition
                        target = exp - 1
                        if target == 0:
                            deriv_expr = f'({exp}.0f * {d_base})'
                            name = f'{prefix}{entry.var_name}'
                            pending_grads[name] = deriv_expr
                            derivs[entry.var_name] = name
                            continue

                        available = {e: v for (b2, e), v in power_vars.items()
                                     if b2 == base}
                        decomp = _decompose_exponent(target, available)

                        if decomp is not None and len(decomp) > 0:
                            factors = ' * '.join(decomp)
                            deriv_expr = f'({exp}.0f * {factors} * {d_base})'
                            name = f'{prefix}{entry.var_name}'
                            pending_grads[name] = deriv_expr
                            derivs[entry.var_name] = name
                            continue

                    # Power detected but can't use power rule (decomp failed
                    # or d_base is zero) — fall through to standard rule

            # --- Standard differentiation (fallback) ---
            # Flush pending gradients that this operation references
            for d_op in d_operands:
                flush_if_pending(d_op)

            deriv_expr = rule(operands, d_operands)

            if deriv_expr is not None:
                name = f'{prefix}{entry.var_name}'
                grad_lines.append(f'float {name} = {deriv_expr};')
                derivs[entry.var_name] = name

            # Track this variable as a potential power base (power 1)
            # for future power chain detection
            if entry.var_name not in power_map:
                d_self = derivs.get(entry.var_name, '0.0f')
                if d_self != '0.0f':
                    power_map[entry.var_name] = (entry.var_name, 1)
                    power_vars[(entry.var_name, 1)] = entry.var_name

        # Flush the final result's pending gradient if any.
        # Intermediate pending gradients that were never referenced are
        # dead code and intentionally not emitted.
        if tape:
            last_deriv = derivs.get(tape[-1].var_name)
            if last_deriv is not None:
                flush_if_pending(last_deriv)

        return grad_lines, derivs
