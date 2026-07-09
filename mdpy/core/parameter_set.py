import numpy as np
from mdpy import precision


class ParameterSet:
    """System-specific force field parameter set.

    Stores type-indexed parameters, per-particle parameters, bonded term
    parameters, and type-pair parameters — all resolved for a specific
    topology and atom type set.

    Attributes
    ----------
    type_parameters : dict[str, ndarray]
        Maps parameter name to an array of shape (num_types,).
        Indexed by particle type index.
    particle_parameters : dict[str, ndarray]
        Maps parameter name to an array of shape (num_particles,).
        Indexed by particle index.
    term_parameters : dict[str, ndarray]
        Maps term type name to a 2D array of shape (num_terms, k).
        Indexed by bonded term index.
    type_pair_parameters : dict[str, ndarray]
        Maps parameter name to a 1D array of stride-2 pairs of shape
        (num_types * num_types * 2,). Indexed as [type_i * num_types + type_j].
    particle_type_indices : ndarray of int (num_particles,)
        Per-particle type index, PDB order. Set during resolve.
    type_name_to_index : dict[str, int]
        Maps alphabetically-sorted type name to integer index.
    num_types : int
        Number of unique atom types in this system.
    """

    def __init__(self):
        self.type_parameters = {}
        self.particle_parameters = {}
        self.term_parameters = {}
        self.type_pair_parameters = {}
        self.particle_type_indices = None
        self.type_name_to_index = {}
        self.num_types = 0

    def add_type_parameter(self, name, values):
        """Store a parameter indexed by particle type.

        name   : str — parameter name, e.g. ``'sigma'``.
        values : array of shape (num_types,) — one value per type.
        """
        self.type_parameters[name] = np.asarray(values, dtype=precision.FLOAT)

    def add_particle_parameter(self, name, values):
        """Store a parameter indexed by individual particle.

        name   : str — parameter name, e.g. ``'mass'``.
        values : array of shape (num_particles,) — one value per particle.
        """
        self.particle_parameters[name] = np.asarray(values, dtype=precision.FLOAT)

    def add_term_parameter(self, name, values):
        """Store a parameter indexed by bonded term instance.

        name   : str — term type name, e.g. ``'bond'``.
        values : 2D array of shape (num_terms, parameters_per_term).
                 Each row holds parameters for one bonded interaction
                 (e.g. [force_constant, equilibrium_distance]).
        """
        self.term_parameters[name] = np.asarray(values, dtype=precision.FLOAT)

    def add_type_pair_parameter(self, name, values):
        """Store a parameter indexed by type pair (flattened n_type × n_type matrix).

        name   : str — parameter name, e.g. ``'sigma_ij'``.
        values : 1D array of shape (num_types * num_types,).
                 Indexed as ``values[type_i * num_types + type_j]``.
        """
        self.type_pair_parameters[name] = np.asarray(values, dtype=precision.FLOAT)

    def get_term_parameter(self, name):
        """Retrieve a term parameter array by term type name."""
        return self.term_parameters[name]
