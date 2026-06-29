#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : charmm_param_parser.py
created time : 2021/10/08
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import itertools
import numpy as np
from mdpy import env
from mdpy.error import FileFormatError
from mdpy.unit import *
from mdpy.core.parameter_table import ParameterTable

RMIN_TO_SIGMA_FACTOR = env.NUMPY_FLOAT(2 ** (-1 / 6))
USED_BLOCK_LABELS = ["ATOMS", "BONDS", "ANGLES", "DIHEDRALS", "IMPROPER", "NONBONDED", "NBFIX"]
UNUSED_BLOCK_LABELS = ["CMAP", "HBOND", "END"]
BLOCK_LABELS = USED_BLOCK_LABELS + UNUSED_BLOCK_LABELS


class CharmmTopparParser:
    def __init__(self, *file_path_list) -> None:
        # Read input
        self._file_path_list = file_path_list
        # Set attributes
        self._parameters = {
            "atom": [],
            "mass": {},
            "charge": {},
            "bond": {},
            "angle": {},
            "nonbonded": {},
            "dihedral": {},
            "improper": {},
        }
        # Parse file
        for file_path in self._file_path_list:
            if file_path.startswith("toppar") or file_path.endswith("str"):
                self.parse_toppar_file(file_path)
            elif file_path.startswith("par") or file_path.endswith("prm"):
                self.parse_par_file(file_path)
            elif file_path.startswith("top") or file_path.endswith("rtf"):
                self.parse_top_file(file_path)
            else:
                raise FileFormatError(
                    "Keyword: top, par, or toppar do not appear in %s, unsupported by CharmmTopparParser."
                    % file_path.split("/")[-1]
                )

    @property
    def parameters(self) -> dict:
        return self._parameters.copy()

    def parse_par_file(self, file_path):
        """Data info:
        - BONDS: V(bond) = Kb(b - b0)**2;
            - Kb: kcal/mole/A**2
            - b0: A
        - ANGLES: V(angle) = Ktheta(Theta - Theta0)**2;
            - Ktheta: kcal/mole/rad**2
            - Theta0: degrees
        DIHEDRALS: V(dihedral) = Kchi(1 + cos(n(chi) - delta))
            - Kchi: kcal/mole
            - n: multiplicity
            - delta: degrees
        IMPROPER: V(improper) = Kpsi(psi - psi0)**2;
            - Kpsi: kcal/mole/rad**2
            - psi0: degrees
        NONBONDED: V(Lennard-Jones) = Eps,i,j[(Rmin,i,j/ri,j)**12 - 2(Rmin,i,j/ri,j)**6]
            - epsilon: kcal/mole, Eps,i,j = sqrt(eps,i * eps,j)
            - Rmin/2: A, Rmin,i,j = Rmin/2,i + Rmin/2,j
        """
        with open(file_path, "r") as f:
            info = f.read().split("\n")
        info_dict =         self._fine_par_info(info)
        self._parse_par_mass_block(info_dict["ATOMS"])
        self._parse_par_bond_block(info_dict.get("BONDS", []))
        self._parse_par_angle_block(info_dict.get("ANGLES", []))
        self._parse_par_dihedral_block(info_dict.get("DIHEDRALS", []))
        self._parse_par_improper_block(info_dict.get("IMPROPER", []))
        self._parse_par_nonbonded_block(info_dict.get("NONBONDED", []))
        if "NBFIX" in info_dict:
            self._parse_par_nbfix_block(info_dict["NBFIX"])

    @staticmethod
    def _fine_par_info(info):
        joined = []
        i = 0
        while i < len(info):
            line = info[i].rstrip()
            while line.endswith('-'):
                line = line[:-1]
                i += 1
                if i < len(info):
                    line += ' ' + info[i].strip()
                else:
                    break
            joined.append(line)
            i += 1
        info = joined

        new_info = []
        start_index = 0
        for cur_index, cur_info in enumerate(info):
            for block_label in BLOCK_LABELS:
                if cur_info.startswith(block_label):
                    new_info.append(info[start_index:cur_index])
                    start_index = cur_index
                    break
        new_info.append(info[start_index:])
        new_info = [i for i in new_info if i != []]  # Avoid no parameter block
        info_dict = {}
        for info in new_info:
            head = info[0].lstrip()
            for block_label in USED_BLOCK_LABELS:
                if head.startswith(block_label):
                    info_dict[block_label] = info
        for key, val in info_dict.items():
            remove_list = [i for i in val if i.lstrip().startswith("!") or i == ""]
            [val.remove(i) for i in remove_list]
            info_dict[key] = [i.strip().split("!")[0].split() for i in val][1:]
        return info_dict

    def _embed_x_element(self, pair):
        if not "X" in pair:
            return [pair]
        else:
            res = []
            pair = [
                [i] if i != "X" else self._parameters["atom"] for i in pair.split("-")
            ]
            pairs = itertools.product(*pair)
            for pair in pairs:
                res.append("-".join(pair))
            return res

    def _parse_par_mass_block(self, infos):
        for info in infos:
            self._parameters["atom"].append(info[2])
            self._parameters["mass"][info[2]] = (
                Quantity(float(info[3]), dalton).convert_to(default_mass_unit).value
            )

    def _parse_par_bond_block(self, infos):
        for info in infos:
            res = [
                Quantity(float(info[2]), kilocalorie_permol / angstrom**2)
                .convert_to(default_energy_unit / default_length_unit**2)
                .value,
                Quantity(float(info[3]), angstrom)
                .convert_to(default_length_unit)
                .value,
            ]
            target_keys = self._embed_x_element("%s-%s" % (info[0], info[1]))
            target_keys.extend(self._embed_x_element("%s-%s" % (info[1], info[0])))
            for key in target_keys:
                if not key in self._parameters["bond"].keys():
                    self._parameters["bond"][key] = res

    def _parse_par_angle_block(self, infos):
        for info in infos:
            if len(info) == 5:
                res = [
                    Quantity(float(info[3]), kilocalorie_permol)
                    .convert_to(default_energy_unit)
                    .value,
                    np.deg2rad(Quantity(float(info[4])).value),
                    Quantity(0, kilocalorie_permol)
                    .convert_to(default_energy_unit)
                    .value,
                    Quantity(0, angstrom).convert_to(default_length_unit).value,
                ]
                target_keys = self._embed_x_element(
                    "%s-%s-%s" % (info[0], info[1], info[2])
                )
                target_keys.extend(
                    self._embed_x_element("%s-%s-%s" % (info[2], info[1], info[0]))
                )
                for key in target_keys:
                    if not key in self._parameters["angle"].keys():
                        self._parameters["angle"][key] = res
            elif len(info) == 7:
                res = [
                    Quantity(float(info[3]), kilocalorie_permol)
                    .convert_to(default_energy_unit)
                    .value,
                    np.deg2rad(Quantity(float(info[4])).value),
                    Quantity(float(info[5]), kilocalorie_permol)
                    .convert_to(default_energy_unit)
                    .value,
                    Quantity(float(info[6]), angstrom)
                    .convert_to(default_length_unit)
                    .value,
                ]
                target_keys = self._embed_x_element(
                    "%s-%s-%s" % (info[0], info[1], info[2])
                )
                target_keys.extend(
                    self._embed_x_element("%s-%s-%s" % (info[2], info[1], info[0]))
                )
                for key in target_keys:
                    if not key in self._parameters["angle"].keys():
                        self._parameters["angle"][key] = res

    def _parse_par_dihedral_block(self, infos):
        x_include_pairs = []
        for info in infos:
            if "X" in "-".join(info[:4]):
                x_include_pairs.append(info)
            else:
                res = [
                    Quantity(float(info[4]), kilocalorie_permol)
                    .convert_to(default_energy_unit)
                    .value,
                    Quantity(float(info[5])).value,
                    np.deg2rad(Quantity(float(info[6])).value),
                ]
                target_keys = [
                    "%s-%s-%s-%s" % (info[0], info[1], info[2], info[3]),
                    "%s-%s-%s-%s" % (info[3], info[2], info[1], info[0]),
                ]
                for key in target_keys:
                    # if not key in self._parameters['dihedral'].keys():
                    if not key in self._parameters["dihedral"].keys():
                        self._parameters["dihedral"][key] = [res]
                    else:
                        self._parameters["dihedral"][key].append(res)
        for info in x_include_pairs:
            res = [
                Quantity(float(info[4]), kilocalorie_permol)
                .convert_to(default_energy_unit)
                .value,
                Quantity(float(info[5])).value,
                np.deg2rad(Quantity(float(info[6])).value),
            ]
            target_keys = self._embed_x_element(
                "%s-%s-%s-%s" % (info[0], info[1], info[2], info[3])
            )
            target_keys.extend(
                self._embed_x_element(
                    "%s-%s-%s-%s" % (info[3], info[2], info[1], info[0])
                )
            )
            for key in target_keys:
                if not key in self._parameters["dihedral"].keys():
                    self._parameters["dihedral"][key] = [res]

    def _parse_par_improper_block(self, infos):
        for info in infos:
            res = [
                Quantity(float(info[4]), kilocalorie_permol)
                .convert_to(default_energy_unit)
                .value,
                np.deg2rad(Quantity(float(info[6])).value),
            ]
            target_keys = []
            for target_key in itertools.permutations(info[:4]):
                target_keys.extend(self._embed_x_element("%s-%s-%s-%s" % (target_key)))
            for key in target_keys:
                if not key in self._parameters["improper"].keys():
                    self._parameters["improper"][key] = res

    def _parse_par_nonbonded_block(self, infos):
        for info in infos:
            if len(info) == 4:
                self._parameters["nonbonded"][info[0]] = [
                    -Quantity(float(info[2]), kilocalorie_permol)
                    .convert_to(default_energy_unit)
                    .value,
                    Quantity(float(info[3]), angstrom)
                    .convert_to(default_length_unit)
                    .value
                    * 2
                    * RMIN_TO_SIGMA_FACTOR,
                ]
            else:
                self._parameters["nonbonded"][info[0]] = [
                    -Quantity(float(info[2]), kilocalorie_permol)
                    .convert_to(default_energy_unit)
                    .value,
                    Quantity(float(info[3]), angstrom)
                    .convert_to(default_length_unit)
                    .value
                    * 2
                    * RMIN_TO_SIGMA_FACTOR,
                    -Quantity(float(info[5]), kilocalorie_permol)
                    .convert_to(default_energy_unit)
                    .value,
                    Quantity(float(info[6]), angstrom)
                    .convert_to(default_length_unit)
                    .value
                    * 2
                    * RMIN_TO_SIGMA_FACTOR,
                ]

    def _parse_par_nbfix_block(self, infos):
        for info in infos:
            if len(info) >= 4:
                type_a = info[0]
                type_b = info[1]
                rmin_val = Quantity(float(info[3]), angstrom).convert_to(default_length_unit).value
                eps_val = -Quantity(float(info[2]), kilocalorie_permol).convert_to(default_energy_unit).value
                rmin_14 = rmin_val
                eps_14 = eps_val
                if len(info) >= 6:
                    rmin_14 = Quantity(float(info[5]), angstrom).convert_to(default_length_unit).value
                    eps_14 = -Quantity(float(info[4]), kilocalorie_permol).convert_to(default_energy_unit).value
                sigma_val = rmin_val * RMIN_TO_SIGMA_FACTOR
                sigma_14_val = rmin_14 * RMIN_TO_SIGMA_FACTOR
                key_fwd = "%s-%s" % (type_a, type_b)
                key_rev = "%s-%s" % (type_b, type_a)
                if "nbfix" not in self._parameters:
                    self._parameters["nbfix"] = {}
                self._parameters["nbfix"][key_fwd] = [eps_val, sigma_val, eps_14, sigma_14_val]
                self._parameters["nbfix"][key_rev] = [eps_val, sigma_val, eps_14, sigma_14_val]

    def parse_top_file(self, file_path):
        with open(file_path, "r") as f:
            info = f.read().split("\n")
        info_dict = self._fine_top_info(info)
        self._parse_top_charge_block(info_dict)

    @staticmethod
    def _fine_top_info(info):
        new_info = []
        start_index = 0
        for cur_index, cur_info in enumerate(info):
            if cur_info.startswith("RESI") or cur_info.startswith("PRES"):
                new_info.append(info[start_index:cur_index])
                start_index = cur_index
        new_info.append(info[start_index:])
        new_info = new_info[1:]
        info_dict = {}
        for info in new_info:
            key = info[0].split()[1]
            remove_list = [i for i in info if not i.startswith("ATOM")]
            [info.remove(i) for i in remove_list]
            info_dict[key] = [i.strip().split("!")[0].split() for i in info]
        return info_dict

    def _parse_top_charge_block(self, info_dict):
        for key, val in info_dict.items():
            for line in val:
                if key != line[2]:
                    self._parameters["charge"]["%s-%s" % (key, line[2])] = (
                        Quantity(float(line[3]), e)
                        .convert_to(default_charge_unit)
                        .value
                    )
                else:  # group name is the same as atom name: ion
                    self._parameters["charge"]["%s" % key] = (
                        Quantity(float(line[3]), e)
                        .convert_to(default_charge_unit)
                        .value
                    )

    def parse_toppar_file(self, file_path):
        with open(file_path, "r") as f:
            info = f.read().split("\n")
        top_info_dict, par_info_dict = self._fine_toppar_info(info)
        # Top data
        self._parse_top_charge_block(top_info_dict)
        # Par data
        self._parse_par_mass_block(par_info_dict["ATOMS"])
        self._parse_par_bond_block(par_info_dict["BONDS"])
        self._parse_par_angle_block(par_info_dict["ANGLES"])
        self._parse_par_dihedral_block(par_info_dict["DIHEDRALS"])
        self._parse_par_improper_block(par_info_dict["IMPROPER"])
        self._parse_par_nonbonded_block(par_info_dict["NONBONDED"])
        if "NBFIX" in par_info_dict:
            self._parse_par_nbfix_block(par_info_dict["NBFIX"])

    def _fine_toppar_info(self, info):
        for i, j in enumerate(info):
            if j.startswith("END"):
                split_index = i
                break
        top_info, par_info = info[: split_index + 1], info[split_index:]

        return self._fine_top_info(top_info), self._fine_par_info(par_info)

    def type_parameters(self, unique_type_names):
        """Return (sigma_array, epsilon_array, sigma_14_array, epsilon_14_array)
        for the given sorted list of type names.
        """
        num_types = len(unique_type_names)
        sigma = np.zeros(num_types, dtype=env.NUMPY_FLOAT)
        epsilon = np.zeros(num_types, dtype=env.NUMPY_FLOAT)
        sigma_14 = np.zeros(num_types, dtype=env.NUMPY_FLOAT)
        epsilon_14 = np.zeros(num_types, dtype=env.NUMPY_FLOAT)
        nonbonded = self._parameters.get("nonbonded", {})
        for type_index, type_name in enumerate(unique_type_names):
            entry = nonbonded.get(type_name)
            if entry is not None:
                epsilon[type_index] = entry[0]
                sigma[type_index] = entry[1]
                if len(entry) == 4:
                    epsilon_14[type_index] = entry[2]
                    sigma_14[type_index] = entry[3]
                else:
                    epsilon_14[type_index] = entry[0]
                    sigma_14[type_index] = entry[1]
        return sigma, epsilon, sigma_14, epsilon_14


def create_parameter_table(topology, toppar_parser):
    """Assemble a ParameterTable from Topology and CHARMM parameters.

    Parameters
    ----------
    topology : Topology
        Must have particle_types, type_names, charges, and bonded
        indices (bond_indices, angle_indices, etc.).
    toppar_parser : CharmmTopparParser
        Parsed CHARMM parameter data.

    Returns
    -------
    ParameterTable
        With type_parameters (sigma, epsilon, sigma_14, epsilon_14),
        particle_parameters (charge, charge_14), and
        term_parameters (bond, angle, dihedral, improper).
    """
    parameters = toppar_parser.parameters

    type_names_sorted = sorted(set(topology.type_names))
    type_name_to_index = {name: idx for idx, name in enumerate(type_names_sorted)}
    num_types = len(type_names_sorted)

    sigma_array = np.zeros(num_types, dtype=env.NUMPY_FLOAT)
    epsilon_array = np.zeros(num_types, dtype=env.NUMPY_FLOAT)
    sigma_14_array = np.zeros(num_types, dtype=env.NUMPY_FLOAT)
    epsilon_14_array = np.zeros(num_types, dtype=env.NUMPY_FLOAT)

    nonbonded = parameters.get("nonbonded", {})
    for type_name, type_index in type_name_to_index.items():
        entry = nonbonded.get(type_name)
        if entry is not None:
            epsilon_array[type_index] = entry[0]
            sigma_array[type_index] = entry[1]
            if len(entry) == 4:
                epsilon_14_array[type_index] = entry[2]
                sigma_14_array[type_index] = entry[3]
            else:
                epsilon_14_array[type_index] = entry[0]
                sigma_14_array[type_index] = entry[1]

    table = ParameterTable()
    table.add_type_parameter("sigma", sigma_array)
    table.add_type_parameter("epsilon", epsilon_array)
    table.add_type_parameter("sigma_14", sigma_14_array)
    table.add_type_parameter("epsilon_14", epsilon_14_array)
    table.add_particle_parameter("charge", topology.charges.copy())
    table.add_particle_parameter("charge_14", topology.charges.copy())

    def _build_pair_matrix(sigma_arr, epsilon_arr, n):
        sigma_half = 0.5 * sigma_arr
        sqrt_eps = np.sqrt(np.maximum(epsilon_arr, 0.0))
        sigma_ij = sigma_half[:, None] + sigma_half[None, :]
        epsilon_ij = sqrt_eps[:, None] * sqrt_eps[None, :]
        return sigma_ij.ravel().astype(env.NUMPY_FLOAT), epsilon_ij.ravel().astype(env.NUMPY_FLOAT)

    sigma_ij, epsilon_ij = _build_pair_matrix(sigma_array, epsilon_array, num_types)
    sigma_ij_14, epsilon_ij_14 = _build_pair_matrix(sigma_14_array, epsilon_14_array, num_types)

    n_pair = num_types * num_types
    lj_pair = np.empty(n_pair * 2, dtype=env.NUMPY_FLOAT)
    lj_pair[0::2] = sigma_ij
    lj_pair[1::2] = epsilon_ij
    lj_pair_14 = np.empty(n_pair * 2, dtype=env.NUMPY_FLOAT)
    lj_pair_14[0::2] = sigma_ij_14
    lj_pair_14[1::2] = epsilon_ij_14

    nbfix_data = parameters.get("nbfix", {})
    for key, entry in nbfix_data.items():
        parts = key.split("-")
        if len(parts) == 2:
            ti = type_name_to_index.get(parts[0])
            tj = type_name_to_index.get(parts[1])
            if ti is not None and tj is not None:
                eps_val, sig_val = entry[0], entry[1]
                eps_14_val, sig_14_val = entry[2], entry[3]
                idx = ti * num_types + tj
                idx_rev = tj * num_types + ti
                lj_pair[idx * 2] = sig_val
                lj_pair[idx * 2 + 1] = eps_val
                lj_pair[idx_rev * 2] = sig_val
                lj_pair[idx_rev * 2 + 1] = eps_val
                lj_pair_14[idx * 2] = sig_14_val
                lj_pair_14[idx * 2 + 1] = eps_14_val
                lj_pair_14[idx_rev * 2] = sig_14_val
                lj_pair_14[idx_rev * 2 + 1] = eps_14_val

    table.add_type_pair_parameter("lj_pair", lj_pair)
    table.add_type_pair_parameter("lj_pair_14", lj_pair_14)

    table.add_term_parameter(
        "bond",
        _resolve_bonds(topology, parameters.get("bond", {})),
    )
    table.add_term_parameter(
        "angle",
        _resolve_angles(topology, parameters.get("angle", {})),
    )
    table.add_term_parameter(
        "dihedral",
        _resolve_dihedrals(topology, parameters.get("dihedral", {})),
    )
    table.add_term_parameter(
        "improper",
        _resolve_impropers(topology, parameters.get("improper", {})),
    )

    return table


def _resolve_bonds(topology, bonded_parameters):
    num_bonds = topology.num_bonds
    result = np.zeros((num_bonds, 2), dtype=env.NUMPY_FLOAT)
    for idx in range(num_bonds):
        i, j = topology.bond_indices[idx]
        type_name_i = topology.type_names[i]
        type_name_j = topology.type_names[j]
        key_forward = "%s-%s" % (type_name_i, type_name_j)
        key_reverse = "%s-%s" % (type_name_j, type_name_i)
        params = bonded_parameters.get(key_forward) or bonded_parameters.get(
            key_reverse
        )
        if params is not None:
            result[idx] = params
    return result


def _resolve_angles(topology, angle_parameters):
    num_angles = topology.num_angles
    result = np.zeros((num_angles, 4), dtype=env.NUMPY_FLOAT)
    for idx in range(num_angles):
        i, j, k = topology.angle_indices[idx]
        type_name_i = topology.type_names[i]
        type_name_j = topology.type_names[j]
        type_name_k = topology.type_names[k]
        key_forward = "%s-%s-%s" % (type_name_i, type_name_j, type_name_k)
        key_reverse = "%s-%s-%s" % (type_name_k, type_name_j, type_name_i)
        params = angle_parameters.get(key_forward) or angle_parameters.get(key_reverse)
        if params is not None:
            result[idx] = params
    return result


def _resolve_dihedrals(topology, dihedral_parameters):
    num_dihedrals = topology.num_dihedrals
    result = np.zeros((num_dihedrals, 3), dtype=env.NUMPY_FLOAT)
    for idx in range(num_dihedrals):
        i, j, k, l = topology.dihedral_indices[idx]
        type_name_i = topology.type_names[i]
        type_name_j = topology.type_names[j]
        type_name_k = topology.type_names[k]
        type_name_l = topology.type_names[l]
        key_forward = "%s-%s-%s-%s" % (
            type_name_i,
            type_name_j,
            type_name_k,
            type_name_l,
        )
        key_reverse = "%s-%s-%s-%s" % (
            type_name_l,
            type_name_k,
            type_name_j,
            type_name_i,
        )
        term_list = dihedral_parameters.get(key_forward) or dihedral_parameters.get(
            key_reverse
        )
        if term_list is not None:
            result[idx] = term_list[0]
    return result


def _resolve_impropers(topology, improper_parameters):
    num_impropers = topology.num_impropers
    result = np.zeros((num_impropers, 2), dtype=env.NUMPY_FLOAT)
    for idx in range(num_impropers):
        i, j, k, l = topology.improper_indices[idx]
        type_name_i = topology.type_names[i]
        type_name_j = topology.type_names[j]
        type_name_k = topology.type_names[k]
        type_name_l = topology.type_names[l]
        key_forward = "%s-%s-%s-%s" % (
            type_name_i,
            type_name_j,
            type_name_k,
            type_name_l,
        )
        key_reverse = "%s-%s-%s-%s" % (
            type_name_l,
            type_name_k,
            type_name_j,
            type_name_i,
        )
        params = improper_parameters.get(key_forward) or improper_parameters.get(
            key_reverse
        )
        if params is not None:
            result[idx] = params
    return result
