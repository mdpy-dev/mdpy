from mdpy.core.state import State
from mdpy.io.charmm_toppar_parser import create_parameter_table
from mdpy.force.factories.charmm import create_charmm_forces
from mdpy.constraint.constraint_scheme import create_constraints
from mdpy.system import System


def create_system(psf, pdb, toppar, pbc_matrix, cutoff=12.0,
                  scheme='h-bonds', ewald_rtol=1e-5, fourier_spacing=1.2):
    """Assemble a ready-to-run System from parsed IO objects.

    Reads per-particle data and metadata from the parsers, populates State,
    and wires force terms + constraints. The caller must still call
    system.set_velocities(...) before running (velocities often come
    from generate_velocity_from_temperature, which needs masses).
    """
    topology = psf.topology
    parameter_table = create_parameter_table(
        topology, toppar, type_names=psf.particle_type_names)

    state = State(topology.num_particles)
    state.set_pbc(pbc_matrix)
    state.set_positions(pdb.positions)
    state.set_charges(psf.charges)
    state.set_masses(psf.masses)
    state.set_type_indices(psf.particle_type_indices)

    forces = create_charmm_forces(
        topology, parameter_table, pbc_matrix, cutoff,
        ewald_rtol=ewald_rtol, fourier_spacing=fourier_spacing,
        particle_type_indices=psf.particle_type_indices)

    system = System(topology, state)
    system.add_force_term(forces['bonded'])
    system.add_force_term(forces['nonbonded'])
    system.add_force_term(forces['pme'], stream='pme')  # PME overlaps primary terms on its own stream

    for constraint in create_constraints(
        topology, parameter_table, scheme,
        masses=psf.masses,
        molecule_ids=psf.molecule_ids,
        molecule_types=psf.molecule_types,
    ):
        system.add_constraint(constraint)

    return system
