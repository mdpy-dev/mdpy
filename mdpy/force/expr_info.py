import re


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
