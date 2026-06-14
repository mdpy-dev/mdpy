from __future__ import annotations

import numpy as np

from mdpy import env
from mdpy.force.bonded_force import BondedForce
from mdpy.force.nonbonded_force import NonbondedForce
from mdpy.force.force_group import ForceGroup
from mdpy.force.expressions.harmonic_bond import harmonic_bond
from mdpy.force.expressions.charmm_angle import charmm_angle
from mdpy.force.expressions.periodic_dihedral import periodic_dihedral
from mdpy.force.expressions.harmonic_improper import harmonic_improper
from mdpy.force.expressions.nb14 import nb14_lj_coulomb
from mdpy.force.expressions.lennard_jones import lennard_jones
from mdpy.force.pme_reciprocal_force import PMEReciprocalForce
from mdpy.force.expressions.coulomb import coulomb


CHARMM_14_CHARGE_SCALE = 1.0


def _create_bond_force(topology, parameter_table):
    force = BondedForce(harmonic_bond)
    force.name = 'bond'
    if topology.num_bonds > 0:
        bond_params = parameter_table.get_term_parameter('bond')
        for idx in range(topology.num_bonds):
            i, j = topology.bond_indices[idx]
            k_val, r0 = bond_params[idx]
            force.add([int(i), int(j)], k=float(k_val), r0=float(r0))
    return force


def _create_angle_force(topology, parameter_table):
    force = BondedForce(charmm_angle)
    force.name = 'angle'
    if topology.num_angles > 0:
        angle_params = parameter_table.get_term_parameter('angle')
        for idx in range(topology.num_angles):
            i, j, k_atom = topology.angle_indices[idx]
            k_val, theta0, k_ub, r_ub = angle_params[idx]
            force.add(
                [int(i), int(j), int(k_atom)],
                k=float(k_val), theta0=float(theta0),
                k_ub=float(k_ub), r_ub=float(r_ub),
            )
    return force


def _create_dihedral_force(topology, parameter_table):
    force = BondedForce(periodic_dihedral)
    force.name = 'dihedral'
    if topology.num_dihedrals > 0:
        dihedral_params = parameter_table.get_term_parameter('dihedral')
        for idx in range(topology.num_dihedrals):
            i, j, k_atom, l = topology.dihedral_indices[idx]
            k_val, n_val, delta = dihedral_params[idx]
            force.add(
                [int(i), int(j), int(k_atom), int(l)],
                k=float(k_val), n=float(n_val), delta=float(delta),
            )
    return force


def _create_improper_force(topology, parameter_table):
    force = BondedForce(harmonic_improper)
    force.name = 'improper'
    if topology.num_impropers > 0:
        improper_params = parameter_table.get_term_parameter('improper')
        for idx in range(topology.num_impropers):
            i, j, k_atom, l = topology.improper_indices[idx]
            k_val, psi0 = improper_params[idx]
            force.add(
                [int(i), int(j), int(k_atom), int(l)],
                k=float(k_val), psi0=float(psi0),
            )
    return force


def _create_nb14_force(topology, parameter_table):
    if topology.num_dihedrals == 0 or 'lj_pair_14' not in parameter_table.type_pair_parameters:
        return None
    force = BondedForce(nb14_lj_coulomb)
    force.name = 'nb14'
    charges = parameter_table.particle_parameters.get('charge')
    if charges is not None:
        force.set_parameter('charge', charges)
    lj_pair_14 = parameter_table.type_pair_parameters['lj_pair_14']
    n_types = int(np.sqrt(len(lj_pair_14) // 2))
    particle_types = topology.particle_types
    seen_pairs = set()
    for idx in range(topology.num_dihedrals):
        a, b, c, d = topology.dihedral_indices[idx]
        pair = (min(int(a), int(d)), max(int(a), int(d)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        i_atom, j_atom = pair
        type_i = int(particle_types[i_atom])
        type_j = int(particle_types[j_atom])
        pair_idx = type_i * n_types + type_j
        sigma = float(lj_pair_14[pair_idx * 2])
        epsilon = float(lj_pair_14[pair_idx * 2 + 1])
        force.add(
            [i_atom, j_atom],
            sigma=sigma, epsilon=epsilon,
            charge_scale=CHARMM_14_CHARGE_SCALE,
        )
    return force


def _create_nonbonded_force(topology, parameter_table, cutoff):
    lj = NonbondedForce(lennard_jones, cutoff)
    lj_pair = parameter_table.type_pair_parameters['lj_pair']
    sigma_matrix = lj_pair[0::2].astype(env.NUMPY_FLOAT)
    epsilon_matrix = lj_pair[1::2].astype(env.NUMPY_FLOAT)
    lj.set_pair_parameter('sigma', sigma_matrix)
    lj.set_pair_parameter('epsilon', epsilon_matrix)

    coulomb_force = NonbondedForce(coulomb, cutoff)

    group = lj + coulomb_force
    group.name = 'nonbonded'
    return group


def create_bonded_group(topology, parameter_table):
    """Create a ForceGroup containing all CHARMM bonded force terms.

    Returns: ForceGroup with bond + angle + dihedral + improper + nb14 forces.
    """
    sub_forces = [
        _create_bond_force(topology, parameter_table),
        _create_angle_force(topology, parameter_table),
        _create_dihedral_force(topology, parameter_table),
        _create_improper_force(topology, parameter_table),
    ]
    group = ForceGroup(sub_forces)
    group.name = 'bonded'
    return group


def create_charmm_forces(topology, parameter_table, pbc_matrix, cutoff=12.0):
    """Create all CHARMM force terms for a PME simulation.

    Args:
        topology: molecular topology from PSF parser.
        parameter_table: parameter table from CHARMM toppar parser.
        pbc_matrix: 3x3 PBC matrix (needed for PME).
        cutoff: nonbonded cutoff in Angstroms.

    Returns:
        dict with keys:
            'bonded': ForceGroup (bond + angle + dihedral + improper + nb14)
            'nonbonded': NonbondedForce (LJ + screened Coulomb)
            'pme': PMEReciprocalForce
    """
    from mdpy.force.expressions.screened_coulomb import screened_coulomb

    bonded = create_bonded_group(topology, parameter_table)
    nb14 = _create_nb14_force(topology, parameter_table)
    if nb14 is not None:
        bonded = bonded + nb14

    nb = NonbondedForce(lennard_jones + screened_coulomb, cutoff)
    lj_pair = parameter_table.type_pair_parameters['lj_pair']
    nb.set_pair_parameter('sigma', lj_pair[0::2].astype(env.NUMPY_FLOAT))
    nb.set_pair_parameter('epsilon', lj_pair[1::2].astype(env.NUMPY_FLOAT))
    nb.name = 'nonbonded'

    pme = PMEReciprocalForce(cutoff)
    pme.bind(topology, parameter_table, pbc_matrix=pbc_matrix)
    nb.set_scalar('alpha', pme.alpha)

    return {
        'bonded': bonded,
        'nonbonded': nb,
        'pme': pme,
    }
