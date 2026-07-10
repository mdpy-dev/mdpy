from __future__ import annotations

import numpy as np
import cupy as cp

from mdpy.unit import KB, NA, default_energy_unit, kelvin

BOLTZMANN = float(KB.convert_to(default_energy_unit / kelvin).value)

BAR_TO_INTERNAL_PRESSURE = float(NA.value) * 1e-32


def build_molecule_csr(particle_molecule_ids):
    """Build CSR format from per-particle molecule IDs.

    Groups atom indices by molecule, producing a flat atom-index array and
    CSR start offsets suitable for GPU kernel consumption.

    Args:
        particle_molecule_ids: array-like of int, one molecule ID per particle
            (PDB order).

    Returns:
        molecule_atoms: np.int32 array of atom indices, grouped by molecule
        molecule_start_index: np.int32 array of size (num_molecules + 1)
    """
    mol_ids = np.asarray(particle_molecule_ids, dtype=np.int32)
    sort_order = np.argsort(mol_ids, kind='stable').astype(np.int32)
    sorted_mol_ids = mol_ids[sort_order]

    boundaries = [0]
    for i in range(1, len(sorted_mol_ids)):
        if sorted_mol_ids[i] != sorted_mol_ids[i - 1]:
            boundaries.append(i)
    boundaries.append(len(sorted_mol_ids))

    return sort_order, np.array(boundaries, dtype=np.int32)
