import re

_REMAP_INDICES_KERNEL = r"""
extern "C" __global__
void remap_indices_kernel(
    const int* __restrict__ d_remap,
    int* __restrict__ d_indices,
    int num_indices,
    const int* __restrict__ d_rebuild_flag
) {
    if (d_rebuild_flag[0] == 0) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_indices) return;
    d_indices[i] = d_remap[d_indices[i]];
}
"""

_MATH_FUNCTIONS = {
    'sqrt': 'sqrtf', 'sin': 'sinf', 'cos': 'cosf',
    'exp': 'expf', 'log': 'logf', 'abs': 'fabsf',
    'erf': 'erff', 'erfc': 'erfcf',
}


def _numeric_literal(value):
    """Return a CUDA float literal string for a numeric value, or None.

    Used by the transpilers to inline module-level named constants (resolved
    from a decorated function's ``__globals__``) as CUDA literals. Booleans are
    excluded so module-level boolean flags are not inlined as numbers. Accepts
    Python scalars and numpy scalars.
    """
    import numpy as np
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return repr(float(value)) + 'f'
    if isinstance(value, (float, np.floating)):
        s = repr(float(value))
        if "." not in s and "e" not in s and "E" not in s:
            s += ".0"
        return s + "f"
    return None


class ExprInfo:
    def __init__(self, positions, per_particle, params, scalars, body):
        self.positions = positions
        self.per_particle = per_particle
        self.params = params
        self.scalars = scalars
        self.body = body


def _strip_trailing_digits(name):
    m = re.match(r'^(.*?)(\d+)$', name)
    if not m:
        return name
    base = m.group(1)
    if base.endswith('_'):
        base = base[:-1]
    return base
