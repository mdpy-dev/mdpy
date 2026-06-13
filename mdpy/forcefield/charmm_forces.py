from __future__ import annotations

import numpy as np

from mdpy import env
from mdpy.force.bonded_force import BondedForce
from mdpy.force.nonbonded_force import NonbondedForce
from mdpy.force.bonded_transpiler import bonded_expression
from mdpy.force.expressions.nb14 import nb14_lj_coulomb
from mdpy.force.expressions.lennard_jones import lennard_jones
from mdpy.force.expressions.coulomb import coulomb


CHARMM_14_CHARGE_SCALE = 1.0


@bonded_expression(body=2)
def charmm_bond(p1, p2, k=0.0, r0=0.0):
    r = distance(p1, p2)
    dr = r - r0
    return k * dr * dr


@bonded_expression(body=3)
def charmm_angle(p1, p2, p3, k=0.0, theta0=0.0, k_ub=0.0, r_ub=0.0):
    theta = angle(p1, p2, p3)
    dt = theta - theta0
    e_angle = k * dt * dt
    r13 = distance(p1, p3)
    dr13 = r13 - r_ub
    e_ub = k_ub * dr13 * dr13
    return e_angle + e_ub


@bonded_expression(body=4)
def charmm_dihedral(p1, p2, p3, p4, k=0.0, n=0.0, delta=0.0):
    phi = dihedral(p1, p2, p3, p4)
    return k * (1.0 + cos(n * phi - delta))


@bonded_expression(body=4)
def charmm_improper(p1, p2, p3, p4, k=0.0, psi0=0.0):
    psi = dihedral(p1, p2, p3, p4)
    dp = psi - psi0
    return k * dp * dp


def _create_bond_force(topology, parameter_table):
    force = BondedForce(charmm_bond)
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
    force = BondedForce(charmm_dihedral)
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
    force = BondedForce(charmm_improper)
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
    force = BondedForce(nb14_lj_coulomb)
    force.name = 'nb14'
    if topology.num_dihedrals > 0:
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


def create_charmm_forces(topology, parameter_table, number_atoms, cutoff=12.0):
    bond = _create_bond_force(topology, parameter_table)
    angle = _create_angle_force(topology, parameter_table)
    dihed = _create_dihedral_force(topology, parameter_table)
    improper = _create_improper_force(topology, parameter_table)
    nb14 = _create_nb14_force(topology, parameter_table)

    bonded_group = bond + angle + dihed + improper + nb14
    bonded_group.name = 'bonded'

    nonbonded = _create_nonbonded_force(topology, parameter_table, cutoff)

    return {
        'bonded': bonded_group,
        'nonbonded': nonbonded,
    }
