#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : charmm_file.py
created time : 2021/10/08
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

from ..error import *
from ..unit import *

USED_BLOCK_LABELS = ['ATOMS', 'BONDS', 'ANGLES', 'DIHEDRALS', 'IMPROPER', 'NONBONDED']
UNUSED_BLOCK_LABELS = ['CMAP', 'NBFIX', 'HBOND', 'END']
BLOCK_LABELS = USED_BLOCK_LABELS + UNUSED_BLOCK_LABELS

class CharmmParamFile:
    def __init__(self, *file_path_list) -> None:
        # Read input
        self._file_path_list = file_path_list
        # Set attributes
        self._param = {
            'mass': {}, 'charge': {},
            'bond': {}, 'angle':{}, 'nonbonded': {},
            'dihedral': {}, 'improper': {}
        }
        # Parse file
        for file_path in self._file_path_list:
            if 'toppar' in file_path or file_path.endswith('str'):
                self.parse_toppar_file(file_path)
            elif 'par' in file_path or file_path.endswith('prm'):
                self.parse_par_file(file_path)
            elif 'top' in file_path or file_path.endswith('rtf'):
                self.parse_top_file(file_path)
            else:
                raise FileFormatError(
                    'Keyword: top, par, or toppar do not appear in %s, unsupported by CharmmParamFile.'
                    %file_path.split('/')[-1]
                )

    @property
    def param(self):
        return self._param

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

    def _parse_par_mass_block(self, infos):
        for info in infos:
            self._param['mass'][info[2]] = Quantity(float(info[3]), dalton).convert_to(default_mass_unit)

    def _parse_par_bond_block(self, infos):
        for info in infos:
            res = [
                Quantity(float(info[2]), kilocalorie_permol / angstrom**2).convert_to(default_energy_unit / default_length_unit**2), 
                Quantity(float(info[3]), angstrom).convert_to(default_length_unit)
            ]
            self._param['bond']['%s-%s' %(info[0], info[1])] = res
            self._param['bond']['%s-%s' %(info[1], info[0])] = res

    def _parse_par_angle_block(self, infos):
        for info in infos:
            res = [
                Quantity(float(info[3]), kilocalorie_permol).convert_to(default_energy_unit), 
                Quantity(float(info[4]))
            ]
            self._param['angle']['%s-%s-%s' %(info[0], info[1], info[2])] = res
            self._param['angle']['%s-%s-%s' %(info[2], info[1], info[0])] = res

    def _parse_par_dihedral_block(self, infos):
        for info in infos:
            res = [
                Quantity(float(info[4]), kilocalorie_permol).convert_to(default_energy_unit), 
                Quantity(float(info[5])),
                Quantity(float(info[6]))
            ]
            self._param['dihedral']['%s-%s-%s-%s' %(info[0], info[1], info[2], info[3])] = res
            self._param['dihedral']['%s-%s-%s-%s' %(info[3], info[2], info[1], info[0])] = res

    def _parse_par_improper_block(self, infos):
        for info in infos:
            res = [
                Quantity(float(info[4]), kilocalorie_permol).convert_to(default_energy_unit), 
                Quantity(float(info[6]))
            ]
            self._param['improper']['%s-%s-%s-%s' %(info[0], info[1], info[2], info[3])] = res

    def _parse_par_nonbonded_block(self, infos):
        for info in infos[1:]:
            self._param['nonbonded'][info[0]] = [
                Quantity(float(info[2]), kilocalorie_permol).convert_to(default_energy_unit),
                Quantity(float(info[3]), angstrom).convert_to(default_length_unit)
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
                    self._param['charge']['%s-%s' %(key, line[2])] = Quantity(float(line[3]), e).convert_to(default_charge_unit)
                else: # group name is the same as atom name: ion
                    self._param['charge']['%s' %key] = Quantity(float(line[3]), e).convert_to(default_charge_unit)

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