import inspect
import re
from mdpy.force.markers import param as _param_marker, scalar as _scalar_marker


class ExprInfo:
    def __init__(self, positions, per_particle, params, scalars, body):
        self.positions = positions
        self.per_particle = per_particle
        self.params = params
        self.scalars = scalars
        self.body = body

    @property
    def particle_properties(self):
        return set(self.per_particle.values())


def _strip_trailing_digits(name):
    m = re.match(r'^(.*?)(\d+)$', name)
    if not m:
        return name
    base = m.group(1)
    if base.endswith('_'):
        base = base[:-1]
    return base


def classify_arguments(func, body):
    sig = inspect.signature(func)
    positions = []
    per_particle = {}
    params = []
    scalars = []

    for i, (name, p) in enumerate(sig.parameters.items()):
        if i < body and re.match(r'^pos\d+$', name):
            positions.append(name)
        elif p.default is _param_marker:
            params.append(name)
        elif p.default is _scalar_marker:
            scalars.append(name)
        elif p.default is inspect.Parameter.empty:
            prop_name = _strip_trailing_digits(name)
            per_particle[name] = prop_name

    return ExprInfo(positions, per_particle, params, scalars, body)
