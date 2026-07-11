import numpy as np


def build_particle_molecule_ids(bond_indices, num_particles):
    """Compute per-particle molecule IDs from bond-graph connected components.

    Uses union-find (disjoint set union) with path compression. Two atoms
    share the same molecule ID iff they are connected by bonds, directly
    or transitively.

    Args:
        bond_indices: (num_bonds, 2) array of atom index pairs.
        num_particles: Total number of particles.

    Returns:
        np.int32 array of shape (num_particles,) with sequential molecule
        IDs starting from 0.
    """
    parent = list(range(num_particles))

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for i in range(bond_indices.shape[0]):
        a, b = int(bond_indices[i, 0]), int(bond_indices[i, 1])
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    mol_id_map = {}
    next_id = 0
    result = np.empty(num_particles, dtype=np.int32)
    for i in range(num_particles):
        root = find(i)
        if root not in mol_id_map:
            mol_id_map[root] = next_id
            next_id += 1
        result[i] = mol_id_map[root]

    return result
