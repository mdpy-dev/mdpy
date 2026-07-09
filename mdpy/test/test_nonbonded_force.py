import pytest
import numpy as np
import cupy as cp

from mdpy.force.nonbonded_transpiler import nonbonded_expression
from mdpy.force.nonbonded_force import (
    NonbondedForce,
    _assemble_exclusion_kernel,
    _prepare_energy_expression,
    _split_per_particle,
)
from mdpy.force.markers import param, scalar as scalar_marker


@nonbonded_expression
def lj_ad(pos1, pos2, sigma=param, epsilon=param):
    r = distance(pos1, pos2)
    sr = sigma / r
    sr6 = sr * sr * sr * sr * sr * sr
    return 4.0 * epsilon * (sr6 * sr6 - sr6)


@nonbonded_expression
def coulomb_ad(pos1, pos2, charge1, charge2):
    r = distance(pos1, pos2)
    return 0.13893556595455 * charge1 * charge2 / r


@nonbonded_expression
def screened_coulomb_ad(pos1, pos2, charge1, charge2, alpha=scalar_marker):
    r = distance(pos1, pos2)
    return 0.13893556595455 * charge1 * charge2 * erfc(alpha * r) / r


combined_lj_coulomb = lj_ad + coulomb_ad
combined_lj_screened = lj_ad + screened_coulomb_ad


class TestKernelAssembly:
    def test_lj_kernel_no_scaling_masks(self):
        energy_cuda, total_expr = _prepare_energy_expression(lj_ad.energy_cuda)
        src = _assemble_exclusion_kernel(
            lj_ad.expr_info,
            energy_cuda,
            lj_ad.grad_cuda,
            lj_ad.radial_force_cuda,
            total_expr,
        )
        assert "scaling_masks" not in src
        assert "is_14" not in src
        assert "scale_shared" not in src

    def test_lj_kernel_no_14_params(self):
        energy_cuda, total_expr = _prepare_energy_expression(lj_ad.energy_cuda)
        src = _assemble_exclusion_kernel(
            lj_ad.expr_info,
            energy_cuda,
            lj_ad.grad_cuda,
            lj_ad.radial_force_cuda,
            total_expr,
        )
        assert "_14" not in src

    def test_combined_kernel_no_scaling_masks(self):
        energy_cuda, total_expr = _prepare_energy_expression(
            combined_lj_coulomb.energy_cuda
        )
        src = _assemble_exclusion_kernel(
            combined_lj_coulomb.expr_info,
            energy_cuda,
            combined_lj_coulomb.grad_cuda,
            combined_lj_coulomb.radial_force_cuda,
            total_expr,
        )
        assert "scaling_masks" not in src
        assert "is_14" not in src

    def test_combined_kernel_charge_from_position_charge(self):
        energy_cuda, total_expr = _prepare_energy_expression(
            combined_lj_coulomb.energy_cuda
        )
        src = _assemble_exclusion_kernel(
            combined_lj_coulomb.expr_info,
            energy_cuda,
            combined_lj_coulomb.grad_cuda,
            combined_lj_coulomb.radial_force_cuda,
            total_expr,
        )
        assert "sorted_charge" not in src
        assert "d_charge" not in src
        assert "position_charge_i.w" in src
        assert "jdata.w" in src
        assert "jcharge" not in src
        assert "pdb_to_slot" not in src
        assert "sorted_data[j_slot]" in src
        assert "block_atoms[j_slot]" in src

    def test_kernel_has_pair_param_matrices(self):
        energy_cuda, total_expr = _prepare_energy_expression(lj_ad.energy_cuda)
        src = _assemble_exclusion_kernel(
            lj_ad.expr_info,
            energy_cuda,
            lj_ad.grad_cuda,
            lj_ad.radial_force_cuda,
            total_expr,
        )
        assert "d_sigma_matrix" in src
        assert "d_epsilon_matrix" in src

    def test_kernel_has_exclusion_masks_only(self):
        energy_cuda, total_expr = _prepare_energy_expression(
            combined_lj_coulomb.energy_cuda
        )
        src = _assemble_exclusion_kernel(
            combined_lj_coulomb.expr_info,
            energy_cuda,
            combined_lj_coulomb.grad_cuda,
            combined_lj_coulomb.radial_force_cuda,
            total_expr,
        )
        assert "exclusion_masks" in src
        assert "scaling_masks" not in src

    def test_kernel_has_warp_structure(self):
        energy_cuda, total_expr = _prepare_energy_expression(
            combined_lj_coulomb.energy_cuda
        )
        src = _assemble_exclusion_kernel(
            combined_lj_coulomb.expr_info,
            energy_cuda,
            combined_lj_coulomb.grad_cuda,
            combined_lj_coulomb.radial_force_cuda,
            total_expr,
        )
        assert "total_warps" in src
        assert "__shfl_sync" in src
        assert "atom_indices_shared" in src

    def test_kernel_valid_braces(self):
        energy_cuda, total_expr = _prepare_energy_expression(
            combined_lj_coulomb.energy_cuda
        )
        src = _assemble_exclusion_kernel(
            combined_lj_coulomb.expr_info,
            energy_cuda,
            combined_lj_coulomb.grad_cuda,
            combined_lj_coulomb.radial_force_cuda,
            total_expr,
        )
        assert src.count("{") == src.count("}")

    def test_force_only_kernel_no_energy(self):
        energy_cuda, total_expr = _prepare_energy_expression(
            combined_lj_coulomb.energy_cuda
        )
        src = _assemble_exclusion_kernel(
            combined_lj_coulomb.expr_info,
            energy_cuda,
            combined_lj_coulomb.grad_cuda,
            combined_lj_coulomb.radial_force_cuda,
            total_expr,
            compute_energy=False,
        )
        assert "energy_buffer" not in src
        assert "total_energy" not in src

    def test_kernel_writes_force_to_slot_index(self):
        """Force writes must use slot index (block_x*32+tgx), not pdb_id (gi)."""
        energy_cuda, total_expr = _prepare_energy_expression(lj_ad.energy_cuda)
        src = _assemble_exclusion_kernel(
            lj_ad.expr_info,
            energy_cuda,
            lj_ad.grad_cuda,
            lj_ad.radial_force_cuda,
            total_expr,
        )
        # i-side: slot index is computed from block_x and tgx into a local var
        assert (
            "int slot_i = block_x * 32 + tgx" in src
        ), "i-side must compute slot index from block_x and tgx"
        assert (
            "atomicAdd(&f_x[slot_i]" in src
        ), "i-side force must write to slot_i, not pdb_id"
        # j-side: must write to j_slot
        assert (
            "atomicAdd(&f_x[j_slot]" in src
        ), "j-side force must write to j_slot, not pdb_id"
        # Must NOT write to gi or gj
        assert "atomicAdd(&f_x[gi]" not in src, "i-side force must not use pdb_id (gi)"
        assert "atomicAdd(&f_x[gj]" not in src, "j-side force must not use pdb_id (gj)"


class TestHelpers:
    def test_split_per_particle_coulomb(self):
        i_props, j_props = _split_per_particle(coulomb_ad.expr_info.per_particle)
        assert "charge1" in i_props
        assert i_props["charge1"] == "charge"
        assert "charge2" in j_props
        assert j_props["charge2"] == "charge"

    def test_split_per_particle_lj(self):
        i_props, j_props = _split_per_particle(lj_ad.expr_info.per_particle)
        assert len(i_props) == 0
        assert len(j_props) == 0

    def test_prepare_energy_single(self):
        energy_cuda, total_expr = _prepare_energy_expression(lj_ad.energy_cuda)
        assert "_result_energy" in energy_cuda
        assert total_expr == "_result_energy"

    def test_prepare_energy_combined(self):
        energy_cuda, total_expr = _prepare_energy_expression(
            combined_lj_coulomb.energy_cuda
        )
        assert "_result_energy_2" in energy_cuda
        assert total_expr == "_result_energy + _result_energy_2"

    def test_prepare_energy_empty(self):
        energy_cuda, total_expr = _prepare_energy_expression("")
        assert energy_cuda == ""
        assert total_expr == "0.0f"


class TestKernelCompilation:
    def test_lj_kernel_compiles(self):
        energy_cuda, total_expr = _prepare_energy_expression(lj_ad.energy_cuda)
        src = _assemble_exclusion_kernel(
            lj_ad.expr_info,
            energy_cuda,
            lj_ad.grad_cuda,
            lj_ad.radial_force_cuda,
            total_expr,
        )
        kernel = cp.RawKernel(src, "exclusion_block_pair_kernel")
        assert kernel is not None

    def test_coulomb_kernel_compiles(self):
        energy_cuda, total_expr = _prepare_energy_expression(coulomb_ad.energy_cuda)
        src = _assemble_exclusion_kernel(
            coulomb_ad.expr_info,
            energy_cuda,
            coulomb_ad.grad_cuda,
            coulomb_ad.radial_force_cuda,
            total_expr,
        )
        kernel = cp.RawKernel(src, "exclusion_block_pair_kernel")
        assert kernel is not None

    def test_combined_kernel_compiles(self):
        energy_cuda, total_expr = _prepare_energy_expression(
            combined_lj_coulomb.energy_cuda
        )
        src = _assemble_exclusion_kernel(
            combined_lj_coulomb.expr_info,
            energy_cuda,
            combined_lj_coulomb.grad_cuda,
            combined_lj_coulomb.radial_force_cuda,
            total_expr,
        )
        kernel = cp.RawKernel(src, "exclusion_block_pair_kernel")
        assert kernel is not None

    def test_exclusion_kernel_compiles(self):
        energy_cuda, total_expr = _prepare_energy_expression(
            combined_lj_coulomb.energy_cuda
        )
        src = _assemble_exclusion_kernel(
            combined_lj_coulomb.expr_info,
            energy_cuda,
            combined_lj_coulomb.grad_cuda,
            combined_lj_coulomb.radial_force_cuda,
            total_expr,
        )
        kernel = cp.RawKernel(src, "exclusion_block_pair_kernel_v2")
        assert kernel is not None

    def test_screened_coulomb_kernel_compiles(self):
        energy_cuda, total_expr = _prepare_energy_expression(
            screened_coulomb_ad.energy_cuda
        )
        src = _assemble_exclusion_kernel(
            screened_coulomb_ad.expr_info,
            energy_cuda,
            screened_coulomb_ad.grad_cuda,
            screened_coulomb_ad.radial_force_cuda,
            total_expr,
        )
        kernel = cp.RawKernel(src, "exclusion_block_pair_kernel")
        assert kernel is not None


class TestClassInstantiation:
    def test_create_lj(self):
        nb = NonbondedForce(lj_ad)
        assert nb.name == "nonbonded"
        assert nb._expr_info.params == ["sigma", "epsilon"]

    def test_create_combined(self):
        nb = NonbondedForce(combined_lj_coulomb)
        assert "sigma" in nb._expr_info.params
        assert "epsilon" in nb._expr_info.params
        assert "charge" in nb._expr_info.per_particle.values()

    def test_set_pair_parameter(self):
        nb = NonbondedForce(lj_ad)
        sigma = np.eye(3, dtype=np.float32)
        nb.set_pair_parameter("sigma", sigma)
        assert "sigma" in nb._pair_param_data

    def test_set_scalar(self):
        nb = NonbondedForce(screened_coulomb_ad)
        nb.set_scalar("alpha", 0.34)
        assert nb._scalar_data["alpha"] == pytest.approx(0.34)


class TestCoulombBruteForce:
    """Brute-force validation of Coulomb against numpy."""

    @pytest.fixture
    def two_particle_system(self):
        positions = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32)
        charges = np.array([1.0, -1.0], dtype=np.float32)
        expected_energy = 0.13893556595455 * 1.0 * (-1.0) / 3.0
        expected_force_x_on_0 = 0.13893556595455 * 1.0 * (-1.0) / (3.0 * 3.0)
        return positions, charges, expected_energy, expected_force_x_on_0

    def test_coulomb_two_particles(self, two_particle_system):
        positions, charges, expected_energy, _ = two_particle_system
        N = len(positions)
        qq = charges[0] * charges[1]
        r = np.sqrt(np.sum((positions[0] - positions[1]) ** 2))
        brute_energy = 0.13893556595455 * qq / r
        dVdr = -0.13893556595455 * qq / (r * r)
        brute_force_on_0 = -dVdr * (positions[0] - positions[1]) / r
        assert brute_energy == pytest.approx(expected_energy, rel=1e-6)
        assert brute_force_on_0[0] == pytest.approx(0.13893556595455 / 9.0, rel=1e-6)


class TestLJBruteForce:
    """Brute-force validation of LJ against numpy."""

    def test_lj_two_particles(self):
        sigma = 3.0
        epsilon = 0.1
        r = 4.0
        sr = sigma / r
        sr6 = sr**6
        sr12 = sr6**2
        expected_energy = 4.0 * epsilon * (sr12 - sr6)
        dEdr = 4.0 * epsilon * (-12.0 * sr12 / r + 6.0 * sr6 / r)
        assert dEdr > 0  # attractive at r > sigma * 2^(1/6)
        positions = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=np.float32)
        expected_force_x_on_0 = -dEdr * (positions[0][0] - positions[1][0]) / r
        assert abs(expected_force_x_on_0) > 0


def test_nonbonded_virial_zero_default(tmp_path):
    """compute_virial=False (default) must not touch d_virial."""
    import numpy as np
    from mdpy.core.state import State
    from mdpy.core.block_list import BlockList
    from mdpy.core.topology import Topology
    from mdpy.force.expressions.lennard_jones import lennard_jones
    from mdpy.force.expressions.coulomb import coulomb
    from mdpy.force.nonbonded_force import NonbondedForce

    lj_coulomb = lennard_jones + coulomb

    num_particles = 1000
    state = State(num_particles)
    positions = np.random.uniform(-5, 5, (num_particles, 3)).astype(np.float32)
    state.set_positions(positions)
    state.set_velocities(np.zeros((num_particles, 3), dtype=np.float32))
    state.set_particle_charges(np.ones(num_particles, dtype=np.float32))
    state.set_particle_masses(np.ones(num_particles, dtype=np.float32))
    state.set_particle_type_indices(np.zeros(num_particles, dtype=np.int32))
    state.set_pbc(np.diag([50.0, 50.0, 50.0]))
    state.allocate_energy_accumulator(1)

    topo = Topology()
    topo.num_particles = num_particles

    force = NonbondedForce(lj_coulomb, cutoff=12.0)
    force.set_pair_parameter("epsilon", np.ones((1, 1), dtype=np.float32))
    force.set_pair_parameter("sigma", np.ones((1, 1), dtype=np.float32))

    bl = BlockList(12.0, skin=1.0, rebuild_check_interval=100)
    bl.rebuild(topo, state, force=True)
    bl.capture_snapshot(state)
    bl.build_block_pairs(topo, state)
    bl.refresh_sorted_type_indices(state)
    bl.refresh_sorted_posq(state)

    # Set d_virial to sentinel value
    state.d_virial[:] = 99.0
    force.compute(state, block_list=bl, compute_energy=False, compute_virial=False)
    assert (
        float(state.d_virial[0]) == 99.0
    ), "d_virial must be untouched when compute_virial=False"


def test_nonbonded_virial_accumulation(tmp_path):
    """compute_virial=True accumulates nonzero virial to d_virial."""
    import numpy as np
    from mdpy.core.state import State
    from mdpy.core.block_list import BlockList
    from mdpy.core.topology import Topology
    from mdpy.force.expressions.lennard_jones import lennard_jones
    from mdpy.force.expressions.coulomb import coulomb
    from mdpy.force.nonbonded_force import NonbondedForce

    lj_coulomb = lennard_jones + coulomb

    num_particles = 1000
    state = State(num_particles)
    positions = np.random.uniform(-5, 5, (num_particles, 3)).astype(np.float32)
    state.set_positions(positions)
    state.set_velocities(np.zeros((num_particles, 3), dtype=np.float32))
    state.set_particle_charges(np.ones(num_particles, dtype=np.float32))
    state.set_particle_masses(np.ones(num_particles, dtype=np.float32))
    state.set_particle_type_indices(np.zeros(num_particles, dtype=np.int32))
    state.set_pbc(np.diag([50.0, 50.0, 50.0]))
    state.allocate_energy_accumulator(1)

    topo = Topology()
    topo.num_particles = num_particles

    force = NonbondedForce(lj_coulomb, cutoff=12.0)
    force.set_pair_parameter("epsilon", np.ones((1, 1), dtype=np.float32))
    force.set_pair_parameter("sigma", np.ones((1, 1), dtype=np.float32))

    bl = BlockList(12.0, skin=1.0, rebuild_check_interval=100)
    bl.rebuild(topo, state, force=True)
    bl.capture_snapshot(state)
    bl.build_block_pairs(topo, state)
    bl.refresh_sorted_type_indices(state)
    bl.refresh_sorted_posq(state)

    state.zero_forces()
    state.zero_virial()
    bl.refresh_sorted_posq(state)
    force.compute(state, block_list=bl, compute_energy=False, compute_virial=True)
    virial = float(state.d_virial[0])
    assert virial != 0.0, "d_virial must be nonzero when compute_virial=True"
    assert np.isfinite(virial), "d_virial must be finite"


def test_nonbonded_virial_with_energy(tmp_path):
    """compute_virial=True, compute_energy=True must accumulate both."""
    import numpy as np
    from mdpy.core.state import State
    from mdpy.core.block_list import BlockList
    from mdpy.core.topology import Topology
    from mdpy.force.expressions.lennard_jones import lennard_jones
    from mdpy.force.expressions.coulomb import coulomb
    from mdpy.force.nonbonded_force import NonbondedForce

    lj_coulomb = lennard_jones + coulomb

    num_particles = 1000
    state = State(num_particles)
    positions = np.random.uniform(-5, 5, (num_particles, 3)).astype(np.float32)
    state.set_positions(positions)
    state.set_velocities(np.zeros((num_particles, 3), dtype=np.float32))
    state.set_particle_charges(np.ones(num_particles, dtype=np.float32))
    state.set_particle_masses(np.ones(num_particles, dtype=np.float32))
    state.set_particle_type_indices(np.zeros(num_particles, dtype=np.int32))
    state.set_pbc(np.diag([50.0, 50.0, 50.0]))
    state.allocate_energy_accumulator(1)

    topo = Topology()
    topo.num_particles = num_particles

    force = NonbondedForce(lj_coulomb, cutoff=12.0)
    force.set_pair_parameter("epsilon", np.ones((1, 1), dtype=np.float32))
    force.set_pair_parameter("sigma", np.ones((1, 1), dtype=np.float32))

    bl = BlockList(12.0, skin=1.0, rebuild_check_interval=100)
    bl.rebuild(topo, state, force=True)
    bl.capture_snapshot(state)
    bl.build_block_pairs(topo, state)
    bl.refresh_sorted_type_indices(state)
    bl.refresh_sorted_posq(state)

    state.zero_forces()
    state.zero_virial()
    state.zero_energy()
    bl.refresh_sorted_posq(state)
    force.compute(state, block_list=bl, compute_energy=True, compute_virial=True)
    virial = float(state.d_virial[0])
    energy = float(state.d_energy[0])
    assert virial != 0.0, "virial must be nonzero"
    assert energy != 0.0, "energy must be nonzero"
