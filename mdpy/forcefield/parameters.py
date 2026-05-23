import numpy as np
from mdpy import env


class ParameterTable:

    def __init__(self):
        self.per_type = {}
        self.per_atom = {}
        self.per_term = {}

    def add_per_type(self, name, values):
        self.per_type[name] = np.asarray(values, dtype=env.NUMPY_FLOAT)

    def add_per_atom(self, name, values):
        self.per_atom[name] = np.asarray(values, dtype=env.NUMPY_FLOAT)

    def add_per_term(self, name, values):
        self.per_term[name] = np.asarray(values, dtype=env.NUMPY_FLOAT)

    def get_per_term(self, name):
        return self.per_term[name]

    def expand_to_per_atom(self, name, particle_types):
        if name in self.per_atom:
            return self.per_atom[name]
        if name not in self.per_type:
            raise KeyError(f'Parameter "{name}" not found in per_type or per_atom')
        per_type_values = self.per_type[name]
        return per_type_values[particle_types]
