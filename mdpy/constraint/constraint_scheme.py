import numpy as np
from .settle import SettleConstraint
from .lincs import LincsConstraint

_WATER_RESIDUE_NAMES = frozenset({
    'TIP3', 'TIP3P', 'TIP4', 'TIP4P', 'TIP5', 'TIP5P',
    'SPC', 'SPCE', 'SPC/E', 'SOL', 'WAT', 'HOH',
})


def _identify_water_molecules(topology, particle_masses, particle_molecule_ids, particle_molecule_types):
    water_triplets = []
    water_bond_set = set()
    if topology.num_bonds == 0:
        return water_triplets, water_bond_set
    bond_indices = topology.bond_indices
    use_mol_types = particle_molecule_types and particle_molecule_types[0] != ''

    oxygen_hydrogen_bonds = {}
    for b in range(bond_indices.shape[0]):
        i, j = int(bond_indices[b, 0]), int(bond_indices[b, 1])
        if particle_molecule_ids[i] != particle_molecule_ids[j]:
            continue
        if use_mol_types:
            if particle_molecule_types[i] not in _WATER_RESIDUE_NAMES:
                continue
        mi, mj = particle_masses[i], particle_masses[j]
        if (mi > 14.5 and mj < 5.0):
            oxygen, hydrogen = i, j
        elif (mj > 14.5 and mi < 5.0):
            oxygen, hydrogen = j, i
        else:
            continue
        oxygen_hydrogen_bonds.setdefault(oxygen, []).append(hydrogen)

    for oxygen, hydrogens in oxygen_hydrogen_bonds.items():
        if len(hydrogens) == 2:
            h1, h2 = hydrogens[0], hydrogens[1]
            water_triplets.append((oxygen, h1, h2))
            water_bond_set.add((min(oxygen, h1), max(oxygen, h1)))
            water_bond_set.add((min(oxygen, h2), max(oxygen, h2)))
            water_bond_set.add((min(h1, h2), max(h1, h2)))

    return water_triplets, water_bond_set


def _identify_constrained_bonds(topology, scheme, water_bond_set, masses):
    bond_indices = topology.bond_indices
    constrained = []
    for b in range(bond_indices.shape[0]):
        i, j = int(bond_indices[b, 0]), int(bond_indices[b, 1])
        key = (min(i, j), max(i, j))
        if key in water_bond_set:
            continue
        if scheme == 'h-bonds':
            if masses[i] < 5.0 or masses[j] < 5.0:
                constrained.append((i, j))
        elif scheme == 'all-bonds':
            constrained.append((i, j))
    return constrained


def _build_bond_length_map(topology, parameter_table):
    """Build {(min(i,j), max(i,j)) -> r0} for all bonds in O(N)."""
    bond_indices = topology.bond_indices
    bond_params = parameter_table.get_term_parameter('bond')
    length_map = {}
    if bond_params is None:
        return length_map
    for b in range(bond_indices.shape[0]):
        bi, bj = int(bond_indices[b, 0]), int(bond_indices[b, 1])
        key = (min(bi, bj), max(bi, bj))
        length_map[key] = float(bond_params[b, 1])
    return length_map


def create_constraints(topology, parameter_set, scheme='h-bonds',
                       *, particle_masses, particle_molecule_ids, particle_molecule_types):
    constraints = []
    if scheme == 'none':
        return constraints

    water_triplets, water_bond_set = _identify_water_molecules(
        topology, particle_masses, particle_molecule_ids, particle_molecule_types)
    length_map = _build_bond_length_map(topology, parameter_set)

    if water_triplets:
        ow, h1 = water_triplets[0][0], water_triplets[0][1]
        dOH = length_map.get((min(ow, h1), max(ow, h1)), 1.5)
        dHH_sq = 2.0 * dOH * dOH * (1.0 - np.cos(np.radians(104.45)))
        dHH = np.sqrt(dHH_sq)
        settle = SettleConstraint(water_triplets, particle_masses, dOH, dHH)
        constraints.append(settle)

    constrained_bonds = _identify_constrained_bonds(
        topology, scheme, water_bond_set, particle_masses)
    if constrained_bonds:
        target_lengths = [
            length_map.get((min(i, j), max(i, j)), 1.5)
            for (i, j) in constrained_bonds
        ]
        lincs = LincsConstraint(constrained_bonds, target_lengths,
                                particle_masses, expansion_order=4,
                                num_iterations=1)
        constraints.append(lincs)

    return constraints
