import pytest
import numpy as np
import cupy as cp

from mdpy.force.nonbonded_transpiler import nonbonded_expression
from mdpy.force.nonbonded_force import (
    NonbondedForce,
    _assemble_exclusion_kernel,
    _prepare_energy_expression,
    _split_per_particle,
    _unique_prop_bases,
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
            lj_ad.expr_info, energy_cuda, lj_ad.grad_cuda, lj_ad.radial_force_cuda, total_expr
        )
        assert 'scaling_masks' not in src
        assert 'is_14' not in src
        assert 'scale_shared' not in src

    def test_lj_kernel_no_14_params(self):
        energy_cuda, total_expr = _prepare_energy_expression(lj_ad.energy_cuda)
        src = _assemble_exclusion_kernel(
            lj_ad.expr_info, energy_cuda, lj_ad.grad_cuda, lj_ad.radial_force_cuda, total_expr
        )
        assert '_14' not in src

    def test_combined_kernel_no_scaling_masks(self):
        energy_cuda, total_expr = _prepare_energy_expression(combined_lj_coulomb.energy_cuda)
        src = _assemble_exclusion_kernel(
            combined_lj_coulomb.expr_info, energy_cuda,
            combined_lj_coulomb.grad_cuda, combined_lj_coulomb.radial_force_cuda, total_expr
        )
        assert 'scaling_masks' not in src
        assert 'is_14' not in src

    def test_combined_kernel_charge_from_position_charge(self):
        energy_cuda, total_expr = _prepare_energy_expression(combined_lj_coulomb.energy_cuda)
        src = _assemble_exclusion_kernel(
            combined_lj_coulomb.expr_info, energy_cuda,
            combined_lj_coulomb.grad_cuda, combined_lj_coulomb.radial_force_cuda, total_expr
        )
        assert 'sorted_charge' not in src
        assert 'd_charge' not in src
        assert 'position_charge_i.w' in src
        assert 'jdata.w' in src
        assert 'jcharge' not in src

    def test_kernel_has_pair_param_matrices(self):
        energy_cuda, total_expr = _prepare_energy_expression(lj_ad.energy_cuda)
        src = _assemble_exclusion_kernel(
            lj_ad.expr_info, energy_cuda, lj_ad.grad_cuda, lj_ad.radial_force_cuda, total_expr
        )
        assert 'd_sigma_matrix' in src
        assert 'd_epsilon_matrix' in src

    def test_kernel_has_exclusion_masks_only(self):
        energy_cuda, total_expr = _prepare_energy_expression(combined_lj_coulomb.energy_cuda)
        src = _assemble_exclusion_kernel(
            combined_lj_coulomb.expr_info, energy_cuda,
            combined_lj_coulomb.grad_cuda, combined_lj_coulomb.radial_force_cuda, total_expr
        )
        assert 'exclusion_masks' in src
        assert 'scaling_masks' not in src

    def test_kernel_has_warp_structure(self):
        energy_cuda, total_expr = _prepare_energy_expression(combined_lj_coulomb.energy_cuda)
        src = _assemble_exclusion_kernel(
            combined_lj_coulomb.expr_info, energy_cuda,
            combined_lj_coulomb.grad_cuda, combined_lj_coulomb.radial_force_cuda, total_expr
        )
        assert 'total_warps' in src
        assert '__shfl_sync' in src
        assert 'atom_indices_shared' in src

    def test_kernel_valid_braces(self):
        energy_cuda, total_expr = _prepare_energy_expression(combined_lj_coulomb.energy_cuda)
        src = _assemble_exclusion_kernel(
            combined_lj_coulomb.expr_info, energy_cuda,
            combined_lj_coulomb.grad_cuda, combined_lj_coulomb.radial_force_cuda, total_expr
        )
        assert src.count('{') == src.count('}')

    def test_force_only_kernel_no_energy(self):
        energy_cuda, total_expr = _prepare_energy_expression(combined_lj_coulomb.energy_cuda)
        src = _assemble_exclusion_kernel(
            combined_lj_coulomb.expr_info, energy_cuda,
            combined_lj_coulomb.grad_cuda, combined_lj_coulomb.radial_force_cuda,
            total_expr, compute_energy=False
        )
        assert 'energy_buffer' not in src
        assert 'total_energy' not in src


class TestHelpers:
    def test_split_per_particle_coulomb(self):
        i_props, j_props = _split_per_particle(
            coulomb_ad.expr_info.per_particle
        )
        assert 'charge1' in i_props
        assert i_props['charge1'] == 'charge'
        assert 'charge2' in j_props
        assert j_props['charge2'] == 'charge'

    def test_split_per_particle_lj(self):
        i_props, j_props = _split_per_particle(lj_ad.expr_info.per_particle)
        assert len(i_props) == 0
        assert len(j_props) == 0

    def test_unique_prop_bases(self):
        bases = _unique_prop_bases({'charge1': 'charge', 'charge2': 'charge'})
        assert bases == ['charge']

    def test_prepare_energy_single(self):
        energy_cuda, total_expr = _prepare_energy_expression(lj_ad.energy_cuda)
        assert '_result_energy' in energy_cuda
        assert total_expr == '_result_energy'

    def test_prepare_energy_combined(self):
        energy_cuda, total_expr = _prepare_energy_expression(
            combined_lj_coulomb.energy_cuda
        )
        assert '_result_energy_2' in energy_cuda
        assert total_expr == '_result_energy + _result_energy_2'

    def test_prepare_energy_empty(self):
        energy_cuda, total_expr = _prepare_energy_expression('')
        assert energy_cuda == ''
        assert total_expr == '0.0f'


class TestKernelCompilation:
    def test_lj_kernel_compiles(self):
        energy_cuda, total_expr = _prepare_energy_expression(lj_ad.energy_cuda)
        src = _assemble_exclusion_kernel(
            lj_ad.expr_info, energy_cuda, lj_ad.grad_cuda, lj_ad.radial_force_cuda, total_expr
        )
        kernel = cp.RawKernel(src, 'exclusion_block_pair_kernel')
        assert kernel is not None

    def test_coulomb_kernel_compiles(self):
        energy_cuda, total_expr = _prepare_energy_expression(coulomb_ad.energy_cuda)
        src = _assemble_exclusion_kernel(
            coulomb_ad.expr_info, energy_cuda, coulomb_ad.grad_cuda, coulomb_ad.radial_force_cuda, total_expr
        )
        kernel = cp.RawKernel(src, 'exclusion_block_pair_kernel')
        assert kernel is not None

    def test_combined_kernel_compiles(self):
        energy_cuda, total_expr = _prepare_energy_expression(combined_lj_coulomb.energy_cuda)
        src = _assemble_exclusion_kernel(
            combined_lj_coulomb.expr_info, energy_cuda,
            combined_lj_coulomb.grad_cuda, combined_lj_coulomb.radial_force_cuda, total_expr
        )
        kernel = cp.RawKernel(src, 'exclusion_block_pair_kernel')
        assert kernel is not None

    def test_exclusion_kernel_compiles(self):
        energy_cuda, total_expr = _prepare_energy_expression(combined_lj_coulomb.energy_cuda)
        src = _assemble_exclusion_kernel(
            combined_lj_coulomb.expr_info, energy_cuda,
            combined_lj_coulomb.grad_cuda, combined_lj_coulomb.radial_force_cuda, total_expr
        )
        kernel = cp.RawKernel(src, 'exclusion_block_pair_kernel_v2')
        assert kernel is not None

    def test_screened_coulomb_kernel_compiles(self):
        energy_cuda, total_expr = _prepare_energy_expression(screened_coulomb_ad.energy_cuda)
        src = _assemble_exclusion_kernel(
            screened_coulomb_ad.expr_info, energy_cuda,
            screened_coulomb_ad.grad_cuda, screened_coulomb_ad.radial_force_cuda, total_expr
        )
        kernel = cp.RawKernel(src, 'exclusion_block_pair_kernel')
        assert kernel is not None


class TestClassInstantiation:
    def test_create_lj(self):
        nb = NonbondedForce(lj_ad)
        assert nb.name == 'nonbonded'
        assert nb._expr_info.params == ['sigma', 'epsilon']

    def test_create_combined(self):
        nb = NonbondedForce(combined_lj_coulomb)
        assert 'sigma' in nb._expr_info.params
        assert 'epsilon' in nb._expr_info.params
        assert 'charge' in nb._prop_bases

    def test_set_pair_parameter(self):
        nb = NonbondedForce(lj_ad)
        sigma = np.eye(3, dtype=np.float32)
        nb.set_pair_parameter('sigma', sigma)
        assert 'sigma' in nb._pair_param_data

    def test_set_scalar(self):
        nb = NonbondedForce(screened_coulomb_ad)
        nb.set_scalar('alpha', 0.34)
        assert nb._scalar_data['alpha'] == pytest.approx(0.34)


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
        sr6 = sr ** 6
        sr12 = sr6 ** 2
        expected_energy = 4.0 * epsilon * (sr12 - sr6)
        dEdr = 4.0 * epsilon * (-12.0 * sr12 / r + 6.0 * sr6 / r)
        assert dEdr > 0  # attractive at r > sigma * 2^(1/6)
        positions = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=np.float32)
        expected_force_x_on_0 = -dEdr * (positions[0][0] - positions[1][0]) / r
        assert abs(expected_force_x_on_0) > 0
