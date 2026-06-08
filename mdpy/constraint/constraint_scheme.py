import numpy as np
from .settle import SettleConstraint
from .lincs import LincsConstraint


def _identify_water_molecules(topology):
    water_triplets = []
    water_bond_set = set()
    if topology.num_bonds == 0:
        return water_triplets, water_bond_set
    bond_indices = topology.bond_indices
    masses = topology.masses
    mol_ids = topology.molecule_ids
    if mol_ids is None:
        return water_triplets, water_bond_set

    oxygen_hydrogen_bonds = {}
    for b in range(bond_indices.shape[0]):
        i, j = int(bond_indices[b, 0]), int(bond_indices[b, 1])
        if mol_ids[i] != mol_ids[j]:
            continue
        mi, mj = masses[i], masses[j]
        if (mi > 10.0 and mj < 5.0):
            oxygen, hydrogen = i, j
        elif (mj > 10.0 and mi < 5.0):
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


def _identify_constrained_bonds(topology, scheme, water_bond_set):
    bond_indices = topology.bond_indices
    masses = topology.masses
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


def _find_bond_target_length(topology, parameter_table, atom_i, atom_j):
    bond_indices = topology.bond_indices
    bond_params = parameter_table.get_term_parameter('bond')
    if bond_params is None:
        return 1.5
    for b in range(bond_indices.shape[0]):
        bi, bj = int(bond_indices[b, 0]), int(bond_indices[b, 1])
        if (bi == atom_i and bj == atom_j) or (bi == atom_j and bj == atom_i):
            return float(bond_params[b, 1])
    return 1.5


def create_constraints(topology, parameter_table, scheme='h-bonds'):
    constraints = []
    if scheme == 'none':
        return constraints

    water_triplets, water_bond_set = _identify_water_molecules(topology)

    if water_triplets:
        ow = water_triplets[0][0]
        h1 = water_triplets[0][1]
        dOH = _find_bond_target_length(topology, parameter_table, ow, h1)
        dHH_sq = 2.0 * dOH * dOH * (1.0 - np.cos(np.radians(104.45)))
        dHH = np.sqrt(dHH_sq)
        settle = SettleConstraint(water_triplets, topology.masses, dOH, dHH)
        constraints.append(settle)

    constrained_bonds = _identify_constrained_bonds(topology, scheme, water_bond_set)
    if constrained_bonds:
        target_lengths = []
        for (i, j) in constrained_bonds:
            r0 = _find_bond_target_length(topology, parameter_table, i, j)
            target_lengths.append(r0)
        lincs = LincsConstraint(constrained_bonds, target_lengths,
                                topology.masses, expansion_order=4,
                                num_iterations=1)
        constraints.append(lincs)

    return constraints
