import numpy as np
from mdpy import env


class ParameterTable:
    """Three-tier parameter storage.

    Attributes
    ----------
    type_parameters : dict[str, ndarray]
        Maps parameter name to an array of shape (num_types,).
        Indexed by particle type index. Example entry:
        ``self.type_parameters['sigma']`` → ``sigma[type_index]``.

    particle_parameters : dict[str, ndarray]
        Maps parameter name to an array of shape (num_particles,).
        Indexed by particle index. Example entry:
        ``self.particle_parameters['charge']`` → ``charge[particle_index]``.

    term_parameters : dict[str, ndarray]
        Maps term type name to a 2D array of shape (num_terms, k).
        Indexed by bonded term index. Example entry:
        ``self.term_parameters['bond']`` → ``bond_parameters[term_index, :]``.
    """

    def __init__(self):
        self.type_parameters = {}
        self.particle_parameters = {}
        self.term_parameters = {}

    def add_type_parameter(self, name, values):
        """Store a parameter indexed by particle type.

        name   : str — parameter name, e.g. ``'sigma'``.
        values : array of shape (num_types,) — one value per type.
        """
        self.type_parameters[name] = np.asarray(values, dtype=env.NUMPY_FLOAT)

    def add_particle_parameter(self, name, values):
        """Store a parameter indexed by individual particle.

        name   : str — parameter name, e.g. ``'charge'``.
        values : array of shape (num_particles,) — one value per particle.
        """
        self.particle_parameters[name] = np.asarray(values, dtype=env.NUMPY_FLOAT)

    def add_term_parameter(self, name, values):
        """Store a parameter indexed by bonded term instance.

        name   : str — term type name, e.g. ``'bond'``.
        values : 2D array of shape (num_terms, parameters_per_term).
                 Each row holds parameters for one bonded interaction
                 (e.g. [force_constant, equilibrium_distance]).
        """
        self.term_parameters[name] = np.asarray(values, dtype=env.NUMPY_FLOAT)

    def get_term_parameter(self, name):
        """Retrieve a term parameter array by term type name."""
        return self.term_parameters[name]

    def expand_to_particle(self, name, particle_types):
        """Return a per-particle array for *name*.

        If *name* already exists in ``particle_parameters``, return it
        directly (e.g. charges are natively per-particle). Otherwise
        expand from ``type_parameters`` using the ``particle_types``
        integer index array.

        This is the bridge that lets ``NonbondedForce`` obtain per-particle
        sigma / epsilon arrays from the type-level LJ table.
        """
        if name in self.particle_parameters:
            return self.particle_parameters[name]
        if name not in self.type_parameters:
            raise KeyError(
                f'Parameter "{name}" not found in type_parameters '
                f'or particle_parameters'
            )
        type_values = self.type_parameters[name]
        return type_values[particle_types]
