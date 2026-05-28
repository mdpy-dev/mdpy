"""Generate brute-force reference data for mdpy validation on 1M9Z.

Pure numpy O(N^2) computation of neighbor pairs, exclusion/scaling masks,
bonded forces, and nonbonded forces. Stored as .npz for test comparison.

Usage:
    conda run -n md_analysis python mdpy/test/generate_bruteforce_reference.py
"""

import os
import sys
import time
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

PSF_PATH = os.path.join(DATA_DIR, '1M9Z.psf')
PDB_PATH = os.path.join(DATA_DIR, '1M9Z_minimized.pdb')
PRM_PATH = os.path.join(DATA_DIR, 'par_all36_prot.prm')
STR_PATH = os.path.join(DATA_DIR, 'toppar_water_ions.str')
OUTPUT_PATH = os.path.join(DATA_DIR, 'bruteforce_reference_1M9Z.npz')

BOX_SIZE = 108.0
CUTOFF = 12.0
COULOMB_CONSTANT = 0.13893556595455


def _minimum_image_vector(pos_i, pos_j, box_size):
    delta = pos_j - pos_i
    delta -= box_size * np.round(delta / box_size)
    return delta


def _setup_system():
    from mdpy.io.psf_parser import PSFParser
    from mdpy.io.pdb_parser import PDBParser
    from mdpy.io.charmm_toppar_parser import CharmmTopparParser
    from mdpy.io.charmm_toppar_parser import create_parameter_table

    psf = PSFParser(PSF_PATH)
    pdb = PDBParser(PDB_PATH)
    toppar = CharmmTopparParser(PRM_PATH, STR_PATH)
    topology = psf.topology
    parameter_table = create_parameter_table(topology, toppar)
    pbc_matrix = np.eye(3, dtype=np.float64) * BOX_SIZE
    pbc_inv = np.linalg.inv(pbc_matrix)
    raw_positions = pdb.positions.astype(np.float64)
    frac = raw_positions @ pbc_inv
    frac -= np.floor(frac)
    wrapped = frac @ pbc_matrix
    return topology, parameter_table, wrapped, pbc_matrix, pbc_inv


def compute_neighbor_pairs(positions, box_size, cutoff):
    N = len(positions)
    cutoff_sq = cutoff * cutoff
    pairs_i = []
    pairs_j = []
    distances = []
    print(f"  Computing O(N^2) neighbor pairs for N={N}, cutoff={cutoff} A...")
    t0 = time.time()
    for i in range(N):
        if i % 5000 == 0 and i > 0:
            elapsed = time.time() - t0
            pct = i / N * 100
            eta = elapsed / i * (N - i)
            print(f"    {i}/{N} ({pct:.1f}%) - {elapsed:.1f}s elapsed, ETA {eta:.0f}s")
        delta = positions[i + 1:] - positions[i]
        delta -= box_size * np.round(delta / box_size)
        dist_sq = np.einsum('ij,ij->i', delta, delta)
        mask = dist_sq < cutoff_sq
        indices = np.where(mask)[0]
        if len(indices) > 0:
            pairs_i.append(np.full(len(indices), i, dtype=np.int64))
            pairs_j.append((indices + i + 1).astype(np.int64))
            distances.append(np.sqrt(dist_sq[indices]))
    if pairs_i:
        pairs_i = np.concatenate(pairs_i)
        pairs_j = np.concatenate(pairs_j)
        distances = np.concatenate(distances)
    else:
        pairs_i = np.empty(0, dtype=np.int64)
        pairs_j = np.empty(0, dtype=np.int64)
        distances = np.empty(0, dtype=np.float64)
    elapsed = time.time() - t0
    print(f"    Done: {len(pairs_i)} pairs found in {elapsed:.1f}s")
    return pairs_i, pairs_j, distances


def compute_exclusion_scaling_flags(topology, pairs_i, pairs_j):
    print("  Computing exclusion/scaling flags...")
    num_pairs = len(pairs_i)
    flags = np.zeros(num_pairs, dtype=np.int8)

    exclusion_offset = topology.exclusion_offset
    exclusion_neighbors = topology.exclusion_neighbors
    exclusion_scale = topology.exclusion_scale

    pair_set = {}
    for idx in range(num_pairs):
        a, b = int(pairs_i[idx]), int(pairs_j[idx])
        if a > b:
            a, b = b, a
        pair_set[(a, b)] = idx

    for atom_i in range(topology.num_particles):
        start = exclusion_offset[atom_i]
        end = exclusion_offset[atom_i + 1]
        for k in range(start, end):
            atom_j = int(exclusion_neighbors[k])
            a, b = min(atom_i, atom_j), max(atom_i, atom_j)
            key = (a, b)
            if key in pair_set:
                scale = exclusion_scale[k]
                idx = pair_set[key]
                if scale == 0.0:
                    flags[idx] = 1
                else:
                    flags[idx] = 2

    n_excluded = np.sum(flags == 1)
    n_scaled = np.sum(flags == 2)
    n_normal = np.sum(flags == 0)
    print(f"    excluded={n_excluded}, scaled={n_scaled}, normal={n_normal}")
    return flags


def compute_bond_forces(positions, bond_indices, bond_params, box_size):
    forces = np.zeros_like(positions, dtype=np.float64)
    energy = 0.0
    for b in range(len(bond_indices)):
        i, j = bond_indices[b]
        k, r0 = bond_params[b]
        delta = _minimum_image_vector(positions[i], positions[j], box_size)
        r = np.linalg.norm(delta)
        if r < 1e-12:
            continue
        dr = r - r0
        e = k * dr * dr
        f_mag = 2.0 * k * dr / r
        f_vec = f_mag * delta
        forces[i] += f_vec
        forces[j] -= f_vec
        energy += e
    return forces, energy


def compute_angle_forces(positions, angle_indices, angle_params, box_size):
    forces = np.zeros_like(positions, dtype=np.float64)
    energy = 0.0
    for a in range(len(angle_indices)):
        i, j, k = angle_indices[a]
        fc, theta0, k_ub, r_ub = angle_params[a]
        r_ji = _minimum_image_vector(positions[j], positions[i], box_size)
        r_jk = _minimum_image_vector(positions[j], positions[k], box_size)
        l_ji = np.linalg.norm(r_ji)
        l_jk = np.linalg.norm(r_jk)
        if l_ji < 1e-12 or l_jk < 1e-12:
            continue
        cos_theta = np.dot(r_ji, r_jk) / (l_ji * l_jk)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        dtheta = theta - theta0
        e_angle = fc * dtheta * dtheta
        f_theta = 2.0 * fc * dtheta
        neg_dEdtheta = -f_theta
        n_vec = np.cross(r_ji, r_jk)
        c1 = np.cross(r_ji, n_vec)
        lc1 = np.linalg.norm(c1)
        if lc1 > 1e-12:
            fv1 = neg_dEdtheta / (lc1 * l_ji) * c1
            forces[i] += fv1
            forces[j] -= fv1
        c3 = np.cross(-r_jk, n_vec)
        lc3 = np.linalg.norm(c3)
        if lc3 > 1e-12:
            fv3 = neg_dEdtheta / (lc3 * l_jk) * c3
            forces[k] += fv3
            forces[j] -= fv3
        r_ik = _minimum_image_vector(positions[i], positions[k], box_size)
        l_ik = np.linalg.norm(r_ik)
        e_ub = 0.0
        if k_ub > 0 and l_ik >= 1e-12:
            dr13 = l_ik - r_ub
            e_ub = k_ub * dr13 * dr13
            f_ub = 2.0 * k_ub * dr13 / l_ik
            f13 = f_ub * r_ik
            forces[i] += f13
            forces[k] -= f13
        energy += e_angle + e_ub
    return forces, energy


def compute_dihedral_forces(positions, dihedral_indices, dihedral_params, box_size):
    forces = np.zeros_like(positions, dtype=np.float64)
    energy = 0.0
    for d in range(len(dihedral_indices)):
        i, j, k, l = dihedral_indices[d]
        fc, n, delta = dihedral_params[d]
        r_ba = _minimum_image_vector(positions[j], positions[i], box_size)
        r_cb = _minimum_image_vector(positions[k], positions[j], box_size)
        r_dc = _minimum_image_vector(positions[l], positions[k], box_size)
        l_ba = np.linalg.norm(r_ba)
        l_cb = np.linalg.norm(r_cb)
        l_dc = np.linalg.norm(r_dc)
        if l_ba < 1e-12 or l_cb < 1e-12 or l_dc < 1e-12:
            continue
        n1 = np.cross(r_ba, r_cb)
        n2 = np.cross(r_cb, r_dc)
        dn = np.dot(n1, n2)
        drn = np.dot(r_ba, n2)
        phi = np.arctan2(l_cb * drn, dn)
        n1s = np.dot(n1, n1)
        n2s = np.dot(n2, n2)
        if n1s < 1e-12 or n2s < 1e-12:
            continue
        e = fc * (1.0 + np.cos(n * phi - delta))
        fv = -(-fc * n * np.sin(n * phi - delta))
        fa_mag = fv * l_cb / n1s
        fd_mag = fv * l_cb / n2s
        f_a = -fa_mag * n1
        f_d = fd_mag * n2
        voc = 0.5 * r_cb
        loc = l_cb * 0.5
        ils = 1.0 / (loc * loc)
        t1 = np.cross(voc, f_d)
        t2 = 0.5 * np.cross(r_dc, f_d)
        t3 = 0.5 * np.cross(-r_ba, f_a)
        st = -(t1 + t2 + t3)
        f_c = np.cross(st, voc) * ils
        f_b = -(f_a + f_c + f_d)
        forces[i] += f_a
        forces[j] += f_b
        forces[k] += f_c
        forces[l] += f_d
        energy += e
    return forces, energy


def compute_improper_forces(positions, improper_indices, improper_params, box_size):
    forces = np.zeros_like(positions, dtype=np.float64)
    energy = 0.0
    for im in range(len(improper_indices)):
        i, j, k, l = improper_indices[im]
        fc, psi0 = improper_params[im]
        r_ba = _minimum_image_vector(positions[j], positions[i], box_size)
        r_cb = _minimum_image_vector(positions[k], positions[j], box_size)
        r_dc = _minimum_image_vector(positions[l], positions[k], box_size)
        l_ba = np.linalg.norm(r_ba)
        l_cb = np.linalg.norm(r_cb)
        l_dc = np.linalg.norm(r_dc)
        if l_ba < 1e-12 or l_cb < 1e-12 or l_dc < 1e-12:
            continue
        n1 = np.cross(r_ba, r_cb)
        n2 = np.cross(r_cb, r_dc)
        dn = np.dot(n1, n2)
        drn = np.dot(r_ba, n2)
        psi = np.arctan2(l_cb * drn, dn)
        n1s = np.dot(n1, n1)
        n2s = np.dot(n2, n2)
        if n1s < 1e-12 or n2s < 1e-12:
            continue
        dpsi = psi - psi0
        e = fc * dpsi * dpsi
        fv = -(2.0 * fc * dpsi)
        fa_mag = fv * l_cb / n1s
        fd_mag = fv * l_cb / n2s
        f_a = -fa_mag * n1
        f_d = fd_mag * n2
        voc = 0.5 * r_cb
        loc = l_cb * 0.5
        ils = 1.0 / (loc * loc)
        t1 = np.cross(voc, f_d)
        t2 = 0.5 * np.cross(r_dc, f_d)
        t3 = 0.5 * np.cross(-r_ba, f_a)
        st = -(t1 + t2 + t3)
        f_c = np.cross(st, voc) * ils
        f_b = -(f_a + f_c + f_d)
        forces[i] += f_a
        forces[j] += f_b
        forces[k] += f_c
        forces[l] += f_d
        energy += e
    return forces, energy


def compute_nonbonded_forces(
    positions, pairs_i, pairs_j, exclusion_flags,
    charges, charges_14, sigma_ij, epsilon_ij, sigma_ij_14, epsilon_ij_14,
    particle_types, box_size, n_types,
):
    N = len(positions)
    forces = np.zeros((N, 3), dtype=np.float64)
    energy = 0.0
    num_pairs = len(pairs_i)
    print(f"  Computing O(N^2) nonbonded forces for {num_pairs} pairs...")
    t0 = time.time()
    for idx in range(num_pairs):
        if idx % 2000000 == 0 and idx > 0:
            elapsed = time.time() - t0
            pct = idx / num_pairs * 100
            eta = elapsed / idx * (num_pairs - idx)
            print(f"    {idx}/{num_pairs} ({pct:.1f}%) - {elapsed:.1f}s, ETA {eta:.0f}s")
        i = pairs_i[idx]
        j = pairs_j[idx]
        flag = exclusion_flags[idx]
        if flag == 1:
            continue
        delta = _minimum_image_vector(positions[i], positions[j], box_size)
        r = np.linalg.norm(delta)
        if r < 1e-12:
            continue
        ti = particle_types[i]
        tj = particle_types[j]
        if flag == 2:
            qi = charges_14[i]
            qj = charges_14[j]
            sig = sigma_ij_14[ti * n_types + tj]
            eps = epsilon_ij_14[ti * n_types + tj]
        else:
            qi = charges[i]
            qj = charges[j]
            sig = sigma_ij[ti * n_types + tj]
            eps = epsilon_ij[ti * n_types + tj]
        sr = sig / r
        sr6 = sr ** 6
        sr12 = sr6 * sr6
        e_lj = 4.0 * eps * (sr12 - sr6)
        e_coul = COULOMB_CONSTANT * qi * qj / r
        f_lj = -24.0 * eps * (2.0 * sr12 - sr6) / r
        f_coul = -COULOMB_CONSTANT * qi * qj / (r * r)
        f_total = (f_lj + f_coul) / r * delta
        forces[i] += f_total
        forces[j] -= f_total
        energy += e_lj + e_coul
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")
    return forces, energy


def main():
    topology, parameter_table, positions, pbc_matrix, pbc_inv = _setup_system()
    N = topology.num_particles
    print(f"System: {N} atoms, box={BOX_SIZE} A, cutoff={CUTOFF} A")

    print("\n[1/5] Neighbor pairs...")
    pairs_i, pairs_j, pair_distances = compute_neighbor_pairs(positions, BOX_SIZE, CUTOFF)

    print("\n[2/5] Exclusion/scaling flags...")
    flags = compute_exclusion_scaling_flags(topology, pairs_i, pairs_j)

    print("\n[3/5] Bonded forces...")
    bond_params = parameter_table.get_term_parameter('bond')
    bond_forces, bond_energy = compute_bond_forces(
        positions, topology.bond_indices, bond_params, BOX_SIZE,
    )
    print(f"  bond energy: {bond_energy:.6f}")

    angle_params = parameter_table.get_term_parameter('angle')
    angle_forces, angle_energy = compute_angle_forces(
        positions, topology.angle_indices, angle_params, BOX_SIZE,
    )
    print(f"  angle energy: {angle_energy:.6f}")

    dihedral_params = parameter_table.get_term_parameter('dihedral')
    dihedral_forces, dihedral_energy = compute_dihedral_forces(
        positions, topology.dihedral_indices, dihedral_params, BOX_SIZE,
    )
    print(f"  dihedral energy: {dihedral_energy:.6f}")

    improper_params = parameter_table.get_term_parameter('improper')
    improper_forces, improper_energy = compute_improper_forces(
        positions, topology.improper_indices, improper_params, BOX_SIZE,
    )
    print(f"  improper energy: {improper_energy:.6f}")

    bonded_forces = bond_forces + angle_forces + dihedral_forces + improper_forces
    bonded_energy_total = bond_energy + angle_energy + dihedral_energy + improper_energy

    print("\n[4/5] Nonbonded forces...")
    charges = parameter_table.particle_parameters['charge'].astype(np.float64)
    charges_14 = parameter_table.particle_parameters.get('charge_14', charges).astype(np.float64)
    sigma_ij = parameter_table.type_pair_parameters['sigma_ij'].astype(np.float64)
    epsilon_ij = parameter_table.type_pair_parameters['epsilon_ij'].astype(np.float64)
    sigma_ij_14 = parameter_table.type_pair_parameters.get('sigma_ij_14', sigma_ij).astype(np.float64)
    epsilon_ij_14 = parameter_table.type_pair_parameters.get('epsilon_ij_14', epsilon_ij).astype(np.float64)
    n_types = int(np.sqrt(len(sigma_ij)))
    particle_types = topology.particle_types

    nonbonded_forces, nonbonded_energy = compute_nonbonded_forces(
        positions, pairs_i, pairs_j, flags,
        charges, charges_14, sigma_ij, epsilon_ij, sigma_ij_14, epsilon_ij_14,
        particle_types, BOX_SIZE, n_types,
    )
    print(f"  nonbonded energy: {nonbonded_energy:.6f}")

    print("\n[5/5] Saving reference data...")
    total_forces = bonded_forces + nonbonded_forces
    total_energy = bonded_energy_total + nonbonded_energy
    print(f"  total energy: {total_energy:.6f}")

    np.savez(
        OUTPUT_PATH,
        positions=positions.astype(np.float64),
        neighbor_pairs_i=pairs_i,
        neighbor_pairs_j=pairs_j,
        neighbor_distances=pair_distances.astype(np.float64),
        pair_exclusion_flags=flags,
        bond_forces=bond_forces.astype(np.float64),
        bond_energy=np.float64(bond_energy),
        angle_forces=angle_forces.astype(np.float64),
        angle_energy=np.float64(angle_energy),
        dihedral_forces=dihedral_forces.astype(np.float64),
        dihedral_energy=np.float64(dihedral_energy),
        improper_forces=improper_forces.astype(np.float64),
        improper_energy=np.float64(improper_energy),
        bonded_forces=bonded_forces.astype(np.float64),
        bonded_energy=np.float64(bonded_energy_total),
        nonbonded_forces=nonbonded_forces.astype(np.float64),
        nonbonded_energy=np.float64(nonbonded_energy),
        total_forces=total_forces.astype(np.float64),
        total_energy=np.float64(total_energy),
    )
    print(f"  Saved: {OUTPUT_PATH}")
    print("Done!")


if __name__ == '__main__':
    main()
