"""Integration tests: NPT ensemble simulation with barostat on 6PO6 molecular system."""

import os
import numpy as np
import pytest

from mdpy import precision
from mdpy.core.state import State
from mdpy.io.psf_parser import PSFParser
from mdpy.io.pdb_parser import PDBParser
from mdpy.io.charmm_toppar_parser import CharmmTopparParser
from mdpy.force.factories.charmm import create_charmm_forces
from mdpy.system import System
from mdpy.integrator.verlet import VerletIntegrator
from mdpy.barostat.berendsen import BerendsenBarostat
from mdpy.barostat.monte_carlo import MonteCarloBarostat
from mdpy.utils.velocity import generate_velocity_from_temperature

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_PSF_PATH = os.path.join(DATA_DIR, "6PO6.psf")
_PDB_PATH = os.path.join(DATA_DIR, "6PO6.pdb")
_PRM_PATH = os.path.join(DATA_DIR, "par_all36_prot.prm")
_STR_PATH = os.path.join(DATA_DIR, "toppar_water_ions.str")


def _setup_6po6_npt():
    psf = PSFParser(_PSF_PATH)
    pdb = PDBParser(_PDB_PATH)
    toppar = CharmmTopparParser(_PRM_PATH, _STR_PATH)
    topology = psf.topology
    parameter_set = toppar.resolve_parameter_set(topology, psf.particle_type_names)

    pbc_matrix = np.eye(3, dtype=np.float64) * 40.0

    state = State(topology.num_particles)
    state.set_positions(pdb.positions)
    state.set_particle_charges(psf.particle_charges)
    state.set_particle_masses(psf.particle_masses)
    state.set_particle_type_indices(parameter_set.particle_type_indices)
    state.set_pbc(pbc_matrix)

    system = System(topology, state)

    forces = create_charmm_forces(topology, parameter_set, pbc_matrix, cutoff=12.0)
    for f in forces["bonded"]:
        system.add_force_term(f)
    system.add_force_term(forces["nonbonded"])
    system.add_force_term(forces["pme"], stream="pme")

    temperature = 300.0
    velocities = generate_velocity_from_temperature(temperature, psf.particle_masses)
    system.set_velocities(velocities)

    return system, temperature


def _run_npt_steps(
    system, integrator, barostat, n, temperature, time_step, sync_interval=10
):
    for i in range(n):
        system.update_neighbor_list(sync_interval=sync_interval)
        system.compute_forces(compute_energy=False, compute_virial=True)
        barostat.apply(system, temperature, time_step)
        integrator.step(system)


class TestNPTIntegration:

    def test_berendsen_npt_preserves_energy_bounds(self):
        system, temperature = _setup_6po6_npt()
        integrator = VerletIntegrator(1.0)
        barostat = BerendsenBarostat(
            target_pressure=6e-9,
            pressure_coupling_time=100.0,
        )

        for i in range(50):
            system.update_neighbor_list(sync_interval=10)
            system.compute_forces(compute_energy=True, compute_virial=True)
            barostat.apply(system, temperature, 1.0)
            integrator.step(system)

        energies = system.dump_energy()
        for name, value in energies.items():
            assert np.isfinite(value), f"Energy {name} is not finite: {value}"
            assert abs(value) < 1e6, f"Energy {name} exploded: {value}"

        final_pos, _ = system.dump_state()
        assert np.all(np.isfinite(final_pos))
        assert system.state.box_x > 0

    def test_mc_npt_preserves_energy_bounds(self):
        system, temperature = _setup_6po6_npt()
        integrator = VerletIntegrator(1.0)
        barostat = MonteCarloBarostat(
            target_pressure=6e-9,
            temperature=temperature,
            frequency=5,
        )

        for i in range(50):
            system.update_neighbor_list(sync_interval=10)
            system.compute_forces(compute_energy=True, compute_virial=True)
            barostat.apply(system, temperature, 1.0)
            integrator.step(system)

        energies = system.dump_energy()
        for name, value in energies.items():
            assert np.isfinite(value), f"Energy {name} is not finite: {value}"
            assert abs(value) < 1e6, f"Energy {name} exploded: {value}"

        final_pos, _ = system.dump_state()
        assert np.all(np.isfinite(final_pos))
        assert system.state.box_x > 0
