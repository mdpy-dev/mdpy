#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : benchmark.py
created time : 2021/11/03
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import os, sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
toppar_dir = os.path.join(cur_dir, '../data/charmm')
data_dir = os.path.join(cur_dir, 'data')
out_dir = os.path.join(cur_dir, 'out')
sys.path.append(os.path.join(cur_dir, '..'))
import numpy as np
import mdpy as md
from mdpy.unit import *

prot_name = '6PO6'
psf = md.file.PSFFile(os.path.join(data_dir, prot_name + '.psf'))
pdb = md.file.PDBFile(os.path.join(data_dir, prot_name + '.pdb'))

topology = psf.create_topology()

forcefield = md.forcefield.CharmmForcefield(topology)
forcefield.set_param_files(
    os.path.join(toppar_dir, 'par_all36_prot.prm'),
    os.path.join(toppar_dir, 'toppar_water_ions_namd.str')
)

ensemble = forcefield.create_ensemble()
ensemble.state.set_pbc_matrix(np.diag(np.ones(3)*100.))
ensemble.state.set_positions(pdb.positions)
ensemble.state.set_velocities_to_temperature(Quantity(273, kelvin))

constraint = [i for i in ensemble._constraints if isinstance(i, md.constraint.CharmmNonbondedConstraint)][0]

constraint.update()

# simulation.ensemble.update_energy()
# simulation.minimize_energy(0.01, max_iterations=250)
# for constraint in simulation.ensemble.constraints:
#     energy = Quantity(constraint.potential_energy, default_energy_unit).convert_to(kilocalorie_permol).value
#     print(constraint, energy)
# simulation.sample(sim_step)