from __future__ import annotations

import numpy as np
from mdpy import env


class Topology:

    __slots__ = [
        'num_particles',
        'bond_indices', 'num_bonds',
        'angle_indices', 'num_angles',
        'dihedral_indices', 'num_dihedrals',
        'improper_indices', 'num_impropers',
        'exclusion_offset', 'exclusion_neighbors', 'exclusion_scale',
        'masses', 'charges', 'particle_types', 'molecule_ids',
        'particle_names', 'type_names', 'chain_ids', 'molecule_types',
        '_legacy_particles', '_legacy_bonds', '_legacy_angles',
        '_legacy_dihedrals', '_legacy_impropers', '_is_joined',
        '_legacy_bonded_particles', '_legacy_scaling_particles',
    ]

    def __init__(self, builder: Builder | None = None):
        if builder is None:
            self._init_legacy()
            return
        self.num_particles = builder._num_particles
        self.masses = builder._masses.copy()
        self.charges = builder._charges.copy()
        self.particle_types = builder._particle_types.copy()
        self.molecule_ids = builder._molecule_ids.copy()
        self.particle_names = list(builder._particle_names)
        self.type_names = list(builder._type_names)
        self.chain_ids = list(builder._chain_ids)
        self.molecule_types = list(builder._molecule_types)

        if builder._bonds:
            self.bond_indices = np.array(
                [b[:2] for b in builder._bonds], dtype=env.NUMPY_INT
            )
        else:
            self.bond_indices = np.empty((0, 2), dtype=env.NUMPY_INT)
        self.num_bonds = self.bond_indices.shape[0]

        if builder._angles:
            self.angle_indices = np.array(
                [a[:3] for a in builder._angles], dtype=env.NUMPY_INT
            )
        else:
            self.angle_indices = np.empty((0, 3), dtype=env.NUMPY_INT)
        self.num_angles = self.angle_indices.shape[0]

        if builder._dihedrals:
            self.dihedral_indices = np.array(
                [d[:4] for d in builder._dihedrals], dtype=env.NUMPY_INT
            )
        else:
            self.dihedral_indices = np.empty((0, 4), dtype=env.NUMPY_INT)
        self.num_dihedrals = self.dihedral_indices.shape[0]

        if builder._impropers:
            self.improper_indices = np.array(
                [im[:4] for im in builder._impropers], dtype=env.NUMPY_INT
            )
        else:
            self.improper_indices = np.empty((0, 4), dtype=env.NUMPY_INT)
        self.num_impropers = self.improper_indices.shape[0]

        if builder._exclusion_offset is not None:
            self.exclusion_offset = builder._exclusion_offset.copy()
            self.exclusion_neighbors = builder._exclusion_neighbors.copy()
            self.exclusion_scale = builder._exclusion_scale.copy()
        else:
            self.exclusion_offset = np.zeros(
                self.num_particles + 1, dtype=env.NUMPY_INT
            )
            self.exclusion_neighbors = np.empty(0, dtype=env.NUMPY_INT)
            self.exclusion_scale = np.empty(0, dtype=env.NUMPY_FLOAT)

    def _init_legacy(self):
        self.num_particles = 0
        self.masses = np.empty(0, dtype=env.NUMPY_FLOAT)
        self.charges = np.empty(0, dtype=env.NUMPY_FLOAT)
        self.particle_types = np.empty(0, dtype=env.NUMPY_INT)
        self.molecule_ids = np.empty(0, dtype=env.NUMPY_INT)
        self.particle_names = []
        self.type_names = []
        self.chain_ids = []
        self.molecule_types = []
        self.bond_indices = np.empty((0, 2), dtype=env.NUMPY_INT)
        self.num_bonds = 0
        self.angle_indices = np.empty((0, 3), dtype=env.NUMPY_INT)
        self.num_angles = 0
        self.dihedral_indices = np.empty((0, 4), dtype=env.NUMPY_INT)
        self.num_dihedrals = 0
        self.improper_indices = np.empty((0, 4), dtype=env.NUMPY_INT)
        self.num_impropers = 0
        self.exclusion_offset = np.zeros(1, dtype=env.NUMPY_INT)
        self.exclusion_neighbors = np.empty(0, dtype=env.NUMPY_INT)
        self.exclusion_scale = np.empty(0, dtype=env.NUMPY_FLOAT)
        self._legacy_particles = []
        self._legacy_bonds = []
        self._legacy_angles = []
        self._legacy_dihedrals = []
        self._legacy_impropers = []
        self._is_joined = False

    def get_excluded_neighbors(self, particle_index: int) -> tuple:
        start = self.exclusion_offset[particle_index]
        end = self.exclusion_offset[particle_index + 1]
        return (
            self.exclusion_neighbors[start:end],
            self.exclusion_scale[start:end],
        )

    @property
    def particles(self):
        return getattr(self, '_legacy_particles', [])

    @property
    def bonds(self):
        return getattr(self, '_legacy_bonds', [])

    @property
    def angles(self):
        return getattr(self, '_legacy_angles', [])

    @property
    def dihedrals(self):
        return getattr(self, '_legacy_dihedrals', [])

    @property
    def impropers(self):
        return getattr(self, '_legacy_impropers', [])

    @property
    def is_joined(self):
        return getattr(self, '_is_joined', True)

    @property
    def bonded_particles(self):
        return getattr(self, '_legacy_bonded_particles', np.empty((0, 0), dtype=env.NUMPY_INT))

    @property
    def scaling_particles(self):
        return getattr(self, '_legacy_scaling_particles', np.empty((0, 0), dtype=env.NUMPY_INT))

    def join(self):
        self._is_joined = True

    def remap_bonded_indices(self, pdb_to_sorted):
        for field in ('bond_indices', 'angle_indices', 'dihedral_indices', 'improper_indices'):
            indices = getattr(self, field, None)
            if indices is not None and len(indices) > 0:
                setattr(self, field, pdb_to_sorted[indices])

        self.exclusion_offset = None
        self.exclusion_neighbors = None
        self.exclusion_scale = None

    def sorted_particle_types(self, pdb_to_sorted):
        return self.particle_types[pdb_to_sorted]

    def build_exclusion_map(self, scale_14=1.0):
        num_particles = self.num_particles
        exclusion_dict = {i: {} for i in range(num_particles)}

        def _add(pair_i, pair_j, scale):
            if pair_j < pair_i:
                pair_i, pair_j = pair_j, pair_i
            if pair_j not in exclusion_dict[pair_i]:
                exclusion_dict[pair_i][pair_j] = scale
            else:
                exclusion_dict[pair_i][pair_j] = min(
                    exclusion_dict[pair_i][pair_j], scale
                )

        if self.bond_indices.shape[0] > 0:
            for bond in self.bond_indices:
                _add(int(bond[0]), int(bond[1]), 0.0)

        if self.angle_indices.shape[0] > 0:
            for angle in self.angle_indices:
                _add(int(angle[0]), int(angle[2]), 0.0)

        if self.dihedral_indices.shape[0] > 0:
            for dihedral in self.dihedral_indices:
                _add(int(dihedral[0]), int(dihedral[3]), scale_14)

        if self.improper_indices.shape[0] > 0:
            for improper in self.improper_indices:
                _add(int(improper[0]), int(improper[3]), 0.0)

        sorted_pairs = []
        for particle_index in range(num_particles):
            neighbors = sorted(exclusion_dict[particle_index].keys())
            for neighbor in neighbors:
                sorted_pairs.append(
                    (particle_index, neighbor, exclusion_dict[particle_index][neighbor])
                )

        offset = np.zeros(num_particles + 1, dtype=env.NUMPY_INT)
        neighbors_array = np.empty(len(sorted_pairs), dtype=env.NUMPY_INT)
        scale_array = np.empty(len(sorted_pairs), dtype=env.NUMPY_FLOAT)

        pair_index = 0
        for particle_index in range(num_particles):
            offset[particle_index] = pair_index
            while (
                pair_index < len(sorted_pairs)
                and sorted_pairs[pair_index][0] == particle_index
            ):
                neighbors_array[pair_index] = sorted_pairs[pair_index][1]
                scale_array[pair_index] = sorted_pairs[pair_index][2]
                pair_index += 1
        offset[num_particles] = pair_index

        self.exclusion_offset = offset
        self.exclusion_neighbors = neighbors_array
        self.exclusion_scale = scale_array

    def split(self):
        self._is_joined = False

    def add_particles(self, particles):
        pass

    def add_bond(self, bond):
        pass

    def add_angle(self, angle):
        pass

    def add_dihedral(self, dihedral, scaling_factor=1):
        pass

    def add_improper(self, improper):
        pass

    def __repr__(self) -> str:
        return (
            '<mdpy.core.Topology: %d particles, %d bonds, %d angles, '
            '%d dihedrals, %d impropers>'
            % (
                self.num_particles, self.num_bonds, self.num_angles,
                self.num_dihedrals, self.num_impropers,
            )
        )


class Builder:

    def __init__(self):
        self._num_particles = 0
        self._masses = None
        self._charges = None
        self._particle_types = None
        self._molecule_ids = None
        self._particle_names: list[str] = []
        self._type_names: list[str] = []
        self._chain_ids: list[str] = []
        self._molecule_types: list[str] = []
        self._bonds: list[list] = []
        self._angles: list[list] = []
        self._dihedrals: list[list] = []
        self._impropers: list[list] = []
        self._exclusion_offset = None
        self._exclusion_neighbors = None
        self._exclusion_scale = None

    def set_particles(
        self,
        masses: np.ndarray,
        charges: np.ndarray,
        particle_types: np.ndarray,
        molecule_ids: np.ndarray | None = None,
        particle_names: list[str] | None = None,
        type_names: list[str] | None = None,
        chain_ids: list[str] | None = None,
        molecule_types: list[str] | None = None,
    ) -> Builder:
        self._num_particles = len(masses)
        self._masses = np.asarray(masses, dtype=env.NUMPY_FLOAT)
        self._charges = np.asarray(charges, dtype=env.NUMPY_FLOAT)
        self._particle_types = np.asarray(particle_types, dtype=env.NUMPY_INT)
        if molecule_ids is not None:
            self._molecule_ids = np.asarray(molecule_ids, dtype=env.NUMPY_INT)
        else:
            self._molecule_ids = np.zeros(self._num_particles, dtype=env.NUMPY_INT)
        self._particle_names = particle_names or [''] * self._num_particles
        self._type_names = type_names or [''] * self._num_particles
        self._chain_ids = chain_ids or [''] * self._num_particles
        self._molecule_types = molecule_types or [''] * self._num_particles
        return self

    def add_bond(self, i: int, j: int, k: float, r0: float) -> Builder:
        self._bonds.append([i, j, k, r0])
        return self

    def add_angle(
        self, i: int, j: int, k: int, force_constant: float,
        equilibrium_angle: float, k_ub: float = 0.0, r_ub: float = 0.0,
    ) -> Builder:
        self._angles.append([i, j, k, force_constant, equilibrium_angle, k_ub, r_ub])
        return self

    def add_dihedral(
        self, i: int, j: int, k: int, l: int,
        force_constant: float, periodicity: float, phase: float,
    ) -> Builder:
        self._dihedrals.append([i, j, k, l, force_constant, periodicity, phase])
        return self

    def add_improper(
        self, i: int, j: int, k: int, l: int,
        force_constant: float, equilibrium_angle: float,
    ) -> Builder:
        self._impropers.append([i, j, k, l, force_constant, equilibrium_angle])
        return self

    def add_bond_indices(
        self, indices: np.ndarray, parameters: np.ndarray,
    ) -> Builder:
        for row in range(indices.shape[0]):
            self._bonds.append(
                [indices[row, 0], indices[row, 1],
                 parameters[row, 0], parameters[row, 1]]
            )
        return self

    def add_angle_indices(
        self, indices: np.ndarray, parameters: np.ndarray,
    ) -> Builder:
        for row in range(indices.shape[0]):
            self._angles.append(
                [indices[row, 0], indices[row, 1], indices[row, 2],
                 parameters[row, 0], parameters[row, 1],
                 parameters[row, 2], parameters[row, 3]]
            )
        return self

    def add_dihedral_indices(
        self, indices: np.ndarray, parameters: np.ndarray,
    ) -> Builder:
        for row in range(indices.shape[0]):
            self._dihedrals.append(
                [indices[row, 0], indices[row, 1],
                 indices[row, 2], indices[row, 3],
                 parameters[row, 0], parameters[row, 1], parameters[row, 2]]
            )
        return self

    def add_improper_indices(
        self, indices: np.ndarray, parameters: np.ndarray,
    ) -> Builder:
        for row in range(indices.shape[0]):
            self._impropers.append(
                [indices[row, 0], indices[row, 1],
                 indices[row, 2], indices[row, 3],
                 parameters[row, 0], parameters[row, 1]]
            )
        return self

    def build_exclusion_map(self, scale_14: float = 1.0) -> Builder:
        num_particles = self._num_particles
        exclusion_dict: dict[int, dict[int, float]] = {
            i: {} for i in range(num_particles)
        }

        def _add(pair_i: int, pair_j: int, scale: float):
            if pair_j < pair_i:
                pair_i, pair_j = pair_j, pair_i
            if pair_j not in exclusion_dict[pair_i]:
                exclusion_dict[pair_i][pair_j] = scale
            else:
                exclusion_dict[pair_i][pair_j] = min(
                    exclusion_dict[pair_i][pair_j], scale
                )

        for bond in self._bonds:
            _add(bond[0], bond[1], 0.0)

        for angle in self._angles:
            _add(angle[0], angle[2], 0.0)

        for dihedral in self._dihedrals:
            _add(dihedral[0], dihedral[3], scale_14)

        for improper in self._impropers:
            _add(improper[0], improper[3], 0.0)

        sorted_pairs = []
        for particle_index in range(num_particles):
            neighbors = sorted(exclusion_dict[particle_index].keys())
            for neighbor in neighbors:
                sorted_pairs.append(
                    (particle_index, neighbor, exclusion_dict[particle_index][neighbor])
                )

        offset = np.zeros(num_particles + 1, dtype=env.NUMPY_INT)
        neighbors_array = np.empty(len(sorted_pairs), dtype=env.NUMPY_INT)
        scale_array = np.empty(len(sorted_pairs), dtype=env.NUMPY_FLOAT)

        pair_index = 0
        for particle_index in range(num_particles):
            offset[particle_index] = pair_index
            while (
                pair_index < len(sorted_pairs)
                and sorted_pairs[pair_index][0] == particle_index
            ):
                neighbors_array[pair_index] = sorted_pairs[pair_index][1]
                scale_array[pair_index] = sorted_pairs[pair_index][2]
                pair_index += 1
        offset[num_particles] = pair_index

        self._exclusion_offset = offset
        self._exclusion_neighbors = neighbors_array
        self._exclusion_scale = scale_array
        return self

    def build(self) -> tuple:
        if self._masses is None:
            raise ValueError('set_particles() must be called before build()')
        topology = Topology(self)
        term_params = {}
        if self._bonds:
            bond_data = np.array(self._bonds, dtype=env.NUMPY_FLOAT)
            term_params['bond'] = bond_data[:, 2:].astype(env.NUMPY_FLOAT)
        if self._angles:
            angle_data = np.array(self._angles, dtype=env.NUMPY_FLOAT)
            term_params['angle'] = angle_data[:, 3:].astype(env.NUMPY_FLOAT)
        if self._dihedrals:
            dihed_data = np.array(self._dihedrals, dtype=env.NUMPY_FLOAT)
            term_params['dihedral'] = dihed_data[:, 4:].astype(env.NUMPY_FLOAT)
        if self._impropers:
            improd_data = np.array(self._impropers, dtype=env.NUMPY_FLOAT)
            term_params['improper'] = improd_data[:, 4:].astype(env.NUMPY_FLOAT)
        return topology, term_params
