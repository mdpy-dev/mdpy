#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
file : charmm.py
created time : 2021/10/20
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
'''

import os, sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
toppar_dir = os.path.join(cur_dir, '../../data/charmm')
str_dir = os.path.join(cur_dir, 'str')
out_dir = os.path.join(cur_dir, 'out')
sys.path.append(os.path.join(cur_dir, '../..'))
import numpy as np
import mdpy as md
from mdpy.unit import *

# Simulation paramters
md.env.set_platform('CUDA')
prot_name = '6PO6_ionized'
dump_interval = 50
sim_step = 100000

# Read input
psf = md.file.PSFFile(os.path.join(str_dir, prot_name + '.psf'))
pdb = md.file.PDBFile(os.path.join(str_dir, prot_name + '.pdb'))
topology = psf.create_topology()
# Create ensemble
forcefield = md.forcefield.CharmmForcefield(topology)
forcefield.set_param_files(
    os.path.join(toppar_dir, 'par_all36_prot.prm'),
    os.path.join(toppar_dir, 'toppar_water_ions_namd.str')
)
ensemble = forcefield.create_ensemble()
ensemble.state.set_pbc_matrix(pdb.pbc_matrix)
# ensemble.state.set_pbc_matrix(np.diag(np.ones(3)*30))
ensemble.state.set_positions(pdb.positions)
ensemble.state.set_velocities_to_temperature(Quantity(300, kelvin))

# Set simlation
integrator = md.integrator.VerletIntegrator(Quantity(0.05, femtosecond))
# integrator = md.integrator.LangevinIntegrator(Quantity(0.05, femtosecond), Quantity(300, kelvin), Quantity(1, 1 / picosecond))
pdb_dumper = md.dumper.PDBDumper(os.path.join(out_dir, prot_name + '.pdb'), dump_interval)
log_dumper = md.dumper.LogDumper(os.path.join(out_dir, prot_name + '.log'), dump_interval,
    step=True, sim_time=True, sim_speed=True, cpu_time=True, rest_time=True, total_step=sim_step, 
    volume=True, density=True,
    potential_energy=True, kinetic_energy=True, total_energy=True, temperature=True
)

simulation = md.Simulation(ensemble, integrator)
simulation.add_dumpers(pdb_dumper, log_dumper)

simulation.ensemble.update()
energy = Quantity(simulation.ensemble.potential_energy, default_energy_unit).convert_to(kilocalorie_permol).value
for constraint in simulation.ensemble.constraints:
    energy = Quantity(constraint.potential_energy, default_energy_unit).convert_to(kilocalorie_permol).value
    print(constraint, energy)

simulation.minimize_energy(0.001, max_iterations=250)

for constraint in simulation.ensemble.constraints:
    energy = Quantity(constraint.potential_energy, default_energy_unit).convert_to(kilocalorie_permol).value
    print(constraint, energy)

simulation.sample(sim_step)
