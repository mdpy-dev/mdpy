import re

_REMAP_INDICES_KERNEL = r"""
extern "C" __global__
void remap_indices_kernel(
    const int* __restrict__ d_remap,
    int* __restrict__ d_indices,
    int num_indices
) {
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
