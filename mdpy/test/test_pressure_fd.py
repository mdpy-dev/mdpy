import numpy as np
from mdpy.core.state import State
from mdpy.core.topology import Topology
from mdpy.system import System, BAR_TO_INTERNAL_PRESSURE
from mdpy.unit import KB, default_energy_unit, kelvin
from mdpy.utils import generate_velocity_from_temperature


def _make_water_system(N_mol=4, box=30.0):
    """Build a small system of 'water-like' 3-atom molecules."""
    N = N_mol * 3
    topo = Topology(); topo.num_particles = N
    state = State(N)
    state.set_pbc(np.diag([box, box, box]).astype(np.float32))
    positions = np.zeros((N, 3), dtype=np.float32)
    mol_ids = np.zeros(N, dtype=np.int32)
    for m in range(N_mol):
        cx = (m + 0.5) * box / N_mol
        positions[m*3] = [cx, box/2, box/2]
        positions[m*3+1] = [cx + 0.96, box/2, box/2]
        positions[m*3+2] = [cx + 0.31, box/2 + 0.92, box/2]
        mol_ids[m*3:m*3+3] = m
    state.set_positions(positions)
    state.set_particle_charges(np.zeros(N, dtype=np.float32))
    state.set_particle_masses(np.ones(N, dtype=np.float32))
    state.set_particle_type_indices(np.zeros(N, dtype=np.int32))
    state.set_particle_molecule_ids(mol_ids)
    state.set_velocities(np.zeros((N, 3), dtype=np.float32))
    system = System(topo, state=state)
    return system


def test_molecule_csr_built():
    """System should build molecule CSR from particle_molecule_ids."""
    system = _make_water_system(N_mol=4)
    system._ensure_molecule_csr()
    assert system._num_molecules == 4
    assert len(system._d_molecule_atoms) == 12
    assert len(system._d_molecule_start_index) == 5


def test_translational_ke_zero_velocity():
    """K_trans = 0 when all velocities are zero."""
    system = _make_water_system(N_mol=4)
    system._ensure_molecule_csr()
    assert system._compute_translational_ke() == 0.0


def test_translational_ke_known_values():
    """K_trans for 4 molecules, each with COM velocity (1,0,0), mass=3."""
    system = _make_water_system(N_mol=4)
    system._ensure_molecule_csr()
    N = 12
    velocities = np.zeros((N, 3), dtype=np.float32)
    velocities[:, 0] = 1.0  # all atoms move at v=(1,0,0)
    system.state.set_velocities(velocities)
    K_trans = system._compute_translational_ke()
    # Each molecule: M=3, v_com=(1,0,0), KE=0.5*3*1=1.5. 4 molecules → 6.0
    assert abs(K_trans - 6.0) < 1e-3, f"Expected K_trans=6.0, got {K_trans}"


def test_translational_ke_excludes_internal_motion():
    """Atoms in same molecule moving in opposite directions → COM velocity = 0 → K_trans = 0."""
    system = _make_water_system(N_mol=1)
    system._ensure_molecule_csr()
    velocities = np.zeros((3, 3), dtype=np.float32)
    velocities[0] = [1.0, 0.0, 0.0]   # atom 0 moves right
    velocities[1] = [-0.5, 0.0, 0.0]  # atom 1 moves left
    velocities[2] = [-0.5, 0.0, 0.0]  # atom 2 moves left
    # COM velocity = (1 - 0.5 - 0.5) / 3 = 0 → K_trans = 0
    system.state.set_velocities(velocities)
    K_trans = system._compute_translational_ke()
    assert abs(K_trans) < 1e-6, f"Expected K_trans≈0 (internal motion only), got {K_trans}"


def test_compute_current_pressure_ideal_gas():
    """For an ideal gas (no force terms): P = N_mol * kT / V.

    Set up N_mol molecules with velocities from temperature T in volume V,
    no force terms. The finite-difference pressure should match N_mol*kT/V.
    """
    BOLTZMANN = float(KB.convert_to(default_energy_unit / kelvin).value)

    N_mol = 200
    system = _make_water_system(N_mol=N_mol, box=80.0)
    N = N_mol * 3
    velocities = generate_velocity_from_temperature(300.0, np.ones(N, dtype=np.float32), seed=42)
    system.state.set_velocities(velocities)

    P_bar = system.compute_current_pressure()
    V = 80.0 ** 3
    P_expected = (N_mol * BOLTZMANN * 300.0 / V) / BAR_TO_INTERNAL_PRESSURE

    assert abs(P_bar - P_expected) < 0.15 * abs(P_expected), (
        f"Pressure {P_bar:.2f} bar vs expected {P_expected:.2f} bar (ideal gas N_mol*kT/V)"
    )
