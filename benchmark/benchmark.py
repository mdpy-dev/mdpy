#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : nonbonded_constraint.py
created time : 2021/11/03
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import os, sys
import cupy.cuda.nvtx as nvtx

cur_dir = os.path.dirname(os.path.abspath(__file__))
toppar_dir = os.path.join(cur_dir, '../data/charmm')
data_dir = os.path.join(cur_dir, 'data')
out_dir = os.path.join(cur_dir, 'out')
sys.path.append(os.path.join(cur_dir, '..'))
import mdpy as md
from mdpy.unit import *
md.env.set_platform('CUDA')

nvtx.RangePush('Create Topology')
prot_name = '1M9Z'
psf = md.file.PSFFile(os.path.join(data_dir, prot_name + '.psf'))
pdb = md.file.PDBFile(os.path.join(data_dir, prot_name + '.pdb'))

topology = psf.create_topology()
nvtx.RangePop()

nvtx.RangePush('Create Ensemble')
params = md.file.CharmmParamFile(
    os.path.join(toppar_dir, 'par_all36_prot.prm'),
    os.path.join(toppar_dir, 'toppar_water_ions_namd.str')
).params
ensemble = md.Ensemble(topology)

nvtx.RangePush('ElectrostaticConstraint IO')
ele_constraint = md.constraint.ElectrostaticConstraint()
ensemble.add_constraints(ele_constraint)
nvtx.RangePop()

nvtx.RangePush('CharmmNonbondedConstraint IO')
nonbonded_constraint = md.constraint.CharmmNonbondedConstraint(params['nonbonded'], 12)
ensemble.add_constraints(nonbonded_constraint)
nvtx.RangePop()

nvtx.RangePush('CharmmBondConstraint IO')
bond_constraint = md.constraint.CharmmBondConstraint(params['bond'])
ensemble.add_constraints(bond_constraint)
nvtx.RangePop()

nvtx.RangePush('CharmmAngleConstraint IO')
angle_constraint = md.constraint.CharmmAngleConstraint(params['angle'])
ensemble.add_constraints(angle_constraint)
nvtx.RangePop()

nvtx.RangePush('CharmmDihedralConstraint IO')
dihedral_constraint = md.constraint.CharmmDihedralConstraint(params['dihedral'])
ensemble.add_constraints(dihedral_constraint)
nvtx.RangePop()

nvtx.RangePush('CharmmImproperConstraint IO')
improper_constraint = md.constraint.CharmmImproperConstraint(params['improper'])
ensemble.add_constraints(improper_constraint)
nvtx.RangePop()

nvtx.RangePop()

nvtx.RangePush('State Initialization')
ensemble.state.set_pbc_matrix(pdb.pbc_matrix)
nvtx.RangePush('Positions')
ensemble.state.set_positions(pdb.positions)
nvtx.RangePop()
nvtx.RangePush('Velocities')
ensemble.state.set_velocities_to_temperature(Quantity(273, kelvin))
nvtx.RangePop()
nvtx.RangePop()

print('Start calculation')

for constraint in ensemble.constraints:
    job_name = '%s' %constraint
    job_name = (job_name.split()[0]).split('.')[-1]
    nvtx.RangePush(job_name)
    constraint.update()
    print(Quantity(constraint.potential_energy, default_energy_unit).convert_to(kilocalorie_permol).value)
    nvtx.RangePop()

# constraint = [i for i in ensemble._constraints if isinstance(i, md.constraint.CharmmNonbondedConstraint)][0]
