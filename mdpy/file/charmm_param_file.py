#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : charmm_param_file.py
created time : 2021/10/08
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import itertools
import numpy as np
from .. import env
from ..error import *
from ..unit import *

RMIN_TO_SIGMA_FACTOR = env.NUMPY_FLOAT(2**(-1/6))
USED_BLOCK_LABELS = ['ATOMS', 'BONDS', 'ANGLES', 'DIHEDRALS', 'IMPROPER', 'NONBONDED']
UNUSED_BLOCK_LABELS = ['CMAP', 'NBFIX', 'HBOND', 'END']
BLOCK_LABELS = USED_BLOCK_LABELS + UNUSED_BLOCK_LABELS

class CharmmParamFile:
    def __init__(self, *file_path_list) -> None:
        # Read input
        self._file_path_list = file_path_list
        # Set attributes
        self._params = {
            'atom': [], 'mass': {}, 'charge': {},
            'bond': {}, 'angle':{},
            'nonbonded': {}, 'dihedral': {}, 'improper': {}
        }
        # Parse file
        for file_path in self._file_path_list:
            if file_path.startswith('toppar') or file_path.endswith('str'):
                self.parse_toppar_file(file_path)
            elif file_path.startswith('par') or file_path.endswith('prm'):
                self.parse_par_file(file_path)
            elif file_path.startswith('top') or file_path.endswith('rtf'):
                self.parse_top_file(file_path)
            else:
                raise FileFormatError(
                    'Keyword: top, par, or toppar do not appear in %s, unsupported by CharmmParamFile.'
                    %file_path.split('/')[-1]
                )

    @property
    def params(self):
        return self._params

    def parse_par_file(self, file_path):
        ''' Data info:
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
        '''
        with open(file_path, 'r') as f:
            info = f.read().split('\n')
        info_dict = self._fine_par_info(info)
        self._parse_par_mass_block(info_dict['ATOMS'])
        self._parse_par_bond_block(info_dict['BONDS'])
        self._parse_par_angle_block(info_dict['ANGLES'])
        self._parse_par_dihedral_block(info_dict['DIHEDRALS'])
        self._parse_par_improper_block(info_dict['IMPROPER'])
        self._parse_par_nonbonded_block(info_dict['NONBONDED'])

    @staticmethod
    def _fine_par_info(info):
        new_info = []
        start_index = 0
        for cur_index, cur_info in enumerate(info):
            for block_label in BLOCK_LABELS:
                if cur_info.startswith(block_label):
                    new_info.append(info[start_index:cur_index])
                    start_index = cur_index
                    break
        new_info.append(info[start_index:])
        new_info = [i for i in new_info if i != []] # Avoid no parameter block
        info_dict = {}
        for info in new_info:
            head = info[0].lstrip()
            for block_label in USED_BLOCK_LABELS:
                if head.startswith(block_label):
                    info_dict[block_label] = info
        for key, val in info_dict.items():
            remove_list = [i for i in val if i.lstrip().startswith('!') or i=='']
            [val.remove(i) for i in remove_list]
            info_dict[key] = [i.strip().split('!')[0].split() for i in val][1:]
        return info_dict

    def _embed_x_element(self, pair):
        if not 'X' in pair:
            return [pair]
        else:
            res = []
            pair = [[i] if i != 'X' else self._params['atom'] for i in pair.split('-')]
            pairs = itertools.product(*pair)
            for pair in pairs:
                res.append('-'.join(pair))
            return res

    def _parse_par_mass_block(self, infos):
        for info in infos:
            self._params['atom'].append(info[2])
            self._params['mass'][info[2]] = Quantity(float(info[3]), dalton).convert_to(default_mass_unit).value

    def _parse_par_bond_block(self, infos):
        for info in infos:
            res = [
                Quantity(float(info[2]), kilocalorie_permol / angstrom**2).convert_to(default_energy_unit / default_length_unit**2).value, 
                Quantity(float(info[3]), angstrom).convert_to(default_length_unit).value
            ]
            target_keys = self._embed_x_element('%s-%s' %(info[0], info[1]))
            target_keys.extend(self._embed_x_element('%s-%s' %(info[1], info[0])))
            for key in target_keys:
                if not key in self._params['bond'].keys():
                    self._params['bond'][key] = res

    def _parse_par_angle_block(self, infos):
        for info in infos:
            if len(info) == 5:
                res = [
                    Quantity(float(info[3]), kilocalorie_permol).convert_to(default_energy_unit).value, 
                    np.deg2rad(Quantity(float(info[4])).value),
                    Quantity(0, kilocalorie_permol).convert_to(default_energy_unit).value, 
                    Quantity(0, angstrom).convert_to(default_length_unit).value
                ]
                target_keys = self._embed_x_element('%s-%s-%s' %(info[0], info[1], info[2]))
                target_keys.extend(self._embed_x_element('%s-%s-%s' %(info[2], info[1], info[0])))
                for key in target_keys:
                    if not key in self._params['angle'].keys():
                        self._params['angle'][key] = res
            elif len(info) == 7:
                res = [
                    Quantity(float(info[3]), kilocalorie_permol).convert_to(default_energy_unit).value, 
                    np.deg2rad(Quantity(float(info[4])).value),
                    Quantity(float(info[5]), kilocalorie_permol).convert_to(default_energy_unit).value, 
                    Quantity(float(info[6]), angstrom).convert_to(default_length_unit).value
                ]
                target_keys = self._embed_x_element('%s-%s-%s' %(info[0], info[1], info[2]))
                target_keys.extend(self._embed_x_element('%s-%s-%s' %(info[2], info[1], info[0])))
                for key in target_keys:
                    if not key in self._params['angle'].keys():
                        self._params['angle'][key] = res

    def _parse_par_dihedral_block(self, infos):
        x_include_pairs = []
        for info in infos:
            if 'X' in '-'.join(info[:4]):
                x_include_pairs.append(info)
            else:
                res = [
                    Quantity(float(info[4]), kilocalorie_permol).convert_to(default_energy_unit).value, 
                    Quantity(float(info[5])).value,
                    np.deg2rad(Quantity(float(info[6])).value)
                ]
                target_keys = [
                    '%s-%s-%s-%s' %(info[0], info[1], info[2], info[3]),
                    '%s-%s-%s-%s' %(info[3], info[2], info[1], info[0])
                ]
                for key in target_keys:
                    # if not key in self._params['dihedral'].keys():
                    if not key in self._params['dihedral'].keys():
                        self._params['dihedral'][key] = [res]
                    else:
                        self._params['dihedral'][key].append(res)
        for info in x_include_pairs:
            res = [
                Quantity(float(info[4]), kilocalorie_permol).convert_to(default_energy_unit).value, 
                Quantity(float(info[5])).value,
                np.deg2rad(Quantity(float(info[6])).value)
            ]
            target_keys = self._embed_x_element(
                '%s-%s-%s-%s' %(info[0], info[1], info[2], info[3])
            )
            target_keys.extend(self._embed_x_element(
                '%s-%s-%s-%s' %(info[3], info[2], info[1], info[0])
            ))
            for key in target_keys:
                if not key in self._params['dihedral'].keys():
                    self._params['dihedral'][key] = [res]

    def _parse_par_improper_block(self, infos):
        for info in infos:
            res = [
                Quantity(float(info[4]), kilocalorie_permol).convert_to(default_energy_unit).value, 
                np.deg2rad(Quantity(float(info[6])).value)
            ]
            target_keys = []
            for target_key in itertools.permutations(info[:4]):
                target_keys.extend(self._embed_x_element(
                    '%s-%s-%s-%s' %(target_key)
                ))
            for key in target_keys:
                if not key in self._params['improper'].keys():
                    self._params['improper'][key] = res

    def _parse_par_nonbonded_block(self, infos):
        for info in infos[1:]:
            if len(info) == 4:
                self._params['nonbonded'][info[0]] = [
                    - Quantity(float(info[2]), kilocalorie_permol).convert_to(default_energy_unit).value,
                    Quantity(float(info[3]), angstrom).convert_to(default_length_unit).value * 2 * RMIN_TO_SIGMA_FACTOR
                ]
            else:
                self._params['nonbonded'][info[0]] = [
                    - Quantity(float(info[2]), kilocalorie_permol).convert_to(default_energy_unit).value,
                    Quantity(float(info[3]), angstrom).convert_to(default_length_unit).value * 2 * RMIN_TO_SIGMA_FACTOR,
                    - Quantity(float(info[5]), kilocalorie_permol).convert_to(default_energy_unit).value,
                    Quantity(float(info[6]), angstrom).convert_to(default_length_unit).value * 2 * RMIN_TO_SIGMA_FACTOR
                ]     

    def parse_top_file(self, file_path):
        with open(file_path, 'r') as f:
            info = f.read().split('\n')
        info_dict = self._fine_top_info(info)
        self._parse_top_charge_block(info_dict)

    @staticmethod
    def _fine_top_info(info):
        new_info = []
        start_index = 0
        for cur_index, cur_info in enumerate(info):
            if cur_info.startswith('RESI') or cur_info.startswith('PRES'):
                new_info.append(info[start_index:cur_index])
                start_index = cur_index
        new_info.append(info[start_index:])
        new_info = new_info[1:]
        info_dict = {}
        for info in new_info:
            key = info[0].split()[1]
            remove_list = [i for i in info if not i.startswith('ATOM')]
            [info.remove(i) for i in remove_list]
            info_dict[key] = [i.strip().split('!')[0].split() for i in info]
        return info_dict

    def _parse_top_charge_block(self, info_dict):
        for key, val in info_dict.items():
            for line in val:
                if key != line[2]:
                    self._params['charge']['%s-%s' %(key, line[2])] = Quantity(float(line[3]), e).convert_to(default_charge_unit).value
                else: # group name is the same as atom name: ion
                    self._params['charge']['%s' %key] = Quantity(float(line[3]), e).convert_to(default_charge_unit).value

    def parse_toppar_file(self, file_path):
        with open(file_path, 'r') as f:
            info = f.read().split('\n')
        top_info_dict, par_info_dict = self._fine_toppar_info(info)
        # Top data
        self._parse_top_charge_block(top_info_dict)
        # Par data
        self._parse_par_mass_block(par_info_dict['ATOMS'])
        self._parse_par_bond_block(par_info_dict['BONDS'])
        self._parse_par_angle_block(par_info_dict['ANGLES'])
        self._parse_par_dihedral_block(par_info_dict['DIHEDRALS'])
        self._parse_par_improper_block(par_info_dict['IMPROPER'])
        self._parse_par_nonbonded_block(par_info_dict['NONBONDED'])


    def _fine_toppar_info(self, info):
        for i, j in enumerate(info):
            if j.startswith('END'):
                split_index = i
                break
        top_info, par_info = info[:split_index+1], info[split_index:]

        return self._fine_top_info(top_info), self._fine_par_info(par_info)