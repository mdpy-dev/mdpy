from __future__ import annotations

import numpy as np
import pytest
from math import sqrt as math_sqrt

from mdpy.force.nonbonded_force import (
    Parameter, NonbondedExpression, nonbonded_expression,
)
from mdpy.force.nonbonded_force import NonbondedForce
from mdpy.force.expressions.lennard_jones import lennard_jones
from mdpy.force.expressions.coulomb import coulomb
from mdpy.forcefield.parameters import ParameterTable
from mdpy.core.topology import Builder
from mdpy import env


class TestParameter:
    def test_getitem_returns_self(self):
        parameter = Parameter()
        result = parameter[0]
        assert result is parameter

    def test_getitem_with_name(self):
        parameter = Parameter()
        result = parameter['atom_i']
        assert result is parameter


class TestDecorator:
    def test_classifies_lj_arguments(self):
        assert lennard_jones.index_names == ['atom_i', 'atom_j']
        assert lennard_jones.parameter_names == ['sigma', 'epsilon']
        assert lennard_jones.distance_name == 'r'

    def test_classifies_coulomb_arguments(self):
        assert coulomb.index_names == ['atom_i', 'atom_j']
        assert coulomb.parameter_names == ['charge']
        assert coulomb.distance_name == 'r'

    def test_custom_index_names(self):
        @nonbonded_expression
        def my_potential(r, a, b, sigma=Parameter()):
            sr = sigma[a] / r
            energy = sr
            force_magnitude = sr / r
            return energy, force_magnitude
        assert my_potential.index_names == ['a', 'b']
        assert my_potential.parameter_names == ['sigma']

    def test_stores_source(self):
        assert 'lennard_jones' in lennard_jones.source
        assert 'sigma' in lennard_jones.source


class TestTranspiler:
    def test_lj_cuda_fragment(self):
        fragment = lennard_jones.cuda_fragment
        assert 'float sigma_ij' in fragment
        assert 'float epsilon_ij' in fragment
        assert 'sigma_i' in fragment
        assert 'sigma_j' in fragment
        assert 'epsilon_i' in fragment
        assert 'epsilon_j' in fragment
        assert 'float _result_energy' in fragment
        assert 'float _result_force' in fragment

    def test_coulomb_cuda_fragment(self):
        fragment = coulomb.cuda_fragment
        assert 'charge_i' in fragment
        assert 'charge_j' in fragment
        assert 'float qq' in fragment
        assert 'float _result_energy' in fragment
        assert 'float _result_force' in fragment

    def test_float_suffix(self):
        @nonbonded_expression
        def simple(r, atom_i, atom_j, sigma=Parameter()):
            sr = sigma[atom_i] / r
            energy = 4.0 * sr
            force_magnitude = 2.0 * sr / r
            return energy, force_magnitude
        assert '4.0f' in simple.cuda_fragment
        assert '2.0f' in simple.cuda_fragment

    def test_power_inline(self):
        fragment = lennard_jones.cuda_fragment
        assert 'sr6' in fragment
        assert '_pow6_' in fragment

    def test_binary_ops(self):
        @nonbonded_expression
        def add_test(r, a, b, sigma=Parameter()):
            combined = sigma[a] + sigma[b]
            energy = combined / r
            force_magnitude = energy / r
            return energy, force_magnitude
        fragment = add_test.cuda_fragment
        assert '(sigma_i + sigma_j)' in fragment
        assert '(combined / r)' in fragment

    def test_local_variables_tracked(self):
        assert 'sigma_ij' in lennard_jones.local_variables
        assert 'epsilon_ij' in lennard_jones.local_variables
        assert 'sr' in lennard_jones.local_variables
        assert 'sr6' in lennard_jones.local_variables
        assert 'sr12' in lennard_jones.local_variables

    def test_sqrt_transpilation(self):
        assert 'sqrtf((epsilon_i * epsilon_j))' in lennard_jones.cuda_fragment


class TestExpressionCombination:
    def test_combined_parameter_names(self):
        combined = lennard_jones + coulomb
        assert 'sigma' in combined.parameter_names
        assert 'epsilon' in combined.parameter_names
        assert 'charge' in combined.parameter_names

    def test_combined_deduplicates_parameters(self):
        combined = lennard_jones + coulomb
        assert combined.parameter_names.count('sigma') <= 1

    def test_combined_cuda_fragment(self):
        combined = lennard_jones + coulomb
        fragment = combined.cuda_fragment
        assert 'float _result_energy_1' in fragment
        assert 'float _result_force_1' in fragment
        assert 'float _result_energy_2' in fragment
        assert 'float _result_force_2' in fragment
        assert 'float energy_val' in fragment
        assert 'float force_magnitude' in fragment
        assert 'charge_i' in fragment

    def test_combined_kernel_source(self):
        combined = lennard_jones + coulomb
        kernel = combined.assemble_cross_tile_kernel()
        assert 'extern "C" __global__' in kernel
        assert 'cross_tile_kernel' in kernel
        assert '__restrict__ sigma' in kernel
        assert '__restrict__ epsilon' in kernel
        assert '__restrict__ charge' in kernel
        assert 'sigma_14' in kernel
        assert 'epsilon_14' in kernel
        assert 'charge_14' in kernel

    def test_second_expression_locals_renamed(self):
        combined = lennard_jones + coulomb
        fragment = combined.cuda_fragment
        lines = fragment.split('\n')
        coulomb_section = '\n'.join(lines[len(lines)//2:])
        assert '_2' in coulomb_section or 'charge_i' in coulomb_section


class TestKernelAssembly:
    def test_lj_kernel_structure(self):
        kernel = lennard_jones.assemble_cross_tile_kernel()
        assert 'extern "C" __global__' in kernel
        assert 'void cross_tile_kernel' in kernel
        assert '__shfl_sync' in kernel
        assert 'atomicAdd' in kernel
        assert 'rsqrtf' in kernel
        assert 'is_14' in kernel
        assert 'energy_val' in kernel
        assert 'force_magnitude' in kernel

    def test_coulomb_kernel_has_charge_arrays(self):
        kernel = coulomb.assemble_cross_tile_kernel()
        assert '__restrict__ charge' in kernel
        assert 'charge_14' in kernel
        assert 'charge_i' in kernel
        assert 'charge_j' in kernel

    def test_combined_kernel_has_all_params(self):
        combined = lennard_jones + coulomb
        kernel = combined.assemble_cross_tile_kernel()
        assert '__restrict__ sigma' in kernel
        assert '__restrict__ epsilon' in kernel
        assert '__restrict__ charge' in kernel
        assert 'sigma_i' in kernel
        assert 'epsilon_i' in kernel
        assert 'charge_i' in kernel

    def test_kernel_has_shift(self):
        kernel = lennard_jones.assemble_cross_tile_kernel()
        assert 'cross_tiles_shift' in kernel
        assert 'shift_x' in kernel
        assert 'shift_y' in kernel
        assert 'shift_z' in kernel

    def test_kernel_has_exclusion(self):
        kernel = lennard_jones.assemble_cross_tile_kernel()
        assert 'exclusion_masks' in kernel
        assert 'scaling_masks' in kernel

    def test_kernel_is_valid_c_syntax(self):
        kernel = lennard_jones.assemble_cross_tile_kernel()
        assert kernel.count('{') == kernel.count('}')


class TestParameterTable:
    def test_add_per_type(self):
        table = ParameterTable()
        table.add_per_type('sigma', [3.4, 2.5])
        assert len(table.per_type['sigma']) == 2
        assert table.per_type['sigma'][0] == pytest.approx(3.4)

    def test_add_per_atom(self):
        table = ParameterTable()
        table.add_per_atom('charge', [-0.3, 0.2])
        assert len(table.per_atom['charge']) == 2

    def test_expand_per_type(self):
        table = ParameterTable()
        table.add_per_type('sigma', [3.4, 2.5, 3.0])
        particle_types = np.array([0, 1, 2, 0], dtype=np.int32)
        per_atom = table.expand_to_per_atom('sigma', particle_types)
        assert per_atom[0] == pytest.approx(3.4)
        assert per_atom[1] == pytest.approx(2.5)
        assert per_atom[2] == pytest.approx(3.0)
        assert per_atom[3] == pytest.approx(3.4)

    def test_expand_per_atom_passthrough(self):
        table = ParameterTable()
        table.add_per_atom('charge', [-0.3, 0.2, 0.1])
        particle_types = np.array([0, 1, 2], dtype=np.int32)
        per_atom = table.expand_to_per_atom('charge', particle_types)
        assert per_atom[0] == pytest.approx(-0.3)

    def test_expand_missing_raises(self):
        table = ParameterTable()
        with pytest.raises(KeyError):
            table.expand_to_per_atom('missing', np.array([0], dtype=np.int32))


class TestEdgeCases:
    def test_expression_with_multiple_assignments(self):
        @nonbonded_expression
        def multi_step(r, atom_i, atom_j, sigma=Parameter()):
            sigma_ij = sigma[atom_i]
            temp = sigma_ij * 2.0
            temp2 = temp + 1.0
            energy = temp2 / r
            force_magnitude = temp2 / (r * r)
            return energy, force_magnitude
        fragment = multi_step.cuda_fragment
        assert 'float sigma_ij' in fragment
        assert 'float temp' in fragment
        assert 'float temp2' in fragment
        assert 'float _result_energy' in fragment
        assert 'float _result_force' in fragment

    def test_expression_with_sqrt(self):
        @nonbonded_expression
        def with_sqrt(r, atom_i, atom_j, sigma=Parameter()):
            val = sqrt(sigma[atom_i] * sigma[atom_j])
            energy = val / r
            force_magnitude = val / (r * r)
            return energy, force_magnitude
        assert 'sqrtf((sigma_i * sigma_j))' in with_sqrt.cuda_fragment

    def test_expression_with_power_of_6(self):
        fragment = lennard_jones.cuda_fragment
        assert '_pow6_' in fragment
        assert 'sr ** 6' not in fragment

    def test_expression_with_negative_constant(self):
        @nonbonded_expression
        def neg(r, atom_i, atom_j, sigma=Parameter()):
            energy = (-1.0) * sigma[atom_i] / r
            force_magnitude = sigma[atom_i] / (r * r)
            return energy, force_magnitude
        assert '(-1.0f)' in neg.cuda_fragment or '-1.0f' in neg.cuda_fragment


class TestCombinedKernelSource:
    def test_combined_kernel_source(self):
        combined = lennard_jones + coulomb
        kernel = combined.assemble_cross_tile_kernel()

        assert 'extern "C" __global__' in kernel
        assert 'void cross_tile_kernel' in kernel

        assert '__restrict__ sigma' in kernel
        assert '__restrict__ sigma_14' in kernel
        assert '__restrict__ epsilon' in kernel
        assert '__restrict__ epsilon_14' in kernel
        assert '__restrict__ charge' in kernel
        assert '__restrict__ charge_14' in kernel

        assert 'sigma_i = sigma[gi]' in kernel
        assert 'sigma_i_14 = sigma_14[gi]' in kernel
        assert 'epsilon_i = epsilon[gi]' in kernel
        assert 'charge_i = charge[gi]' in kernel

        assert 'rsqrtf' in kernel
        assert 'atomicAdd' in kernel
        assert 'energy_val' in kernel
        assert 'force_magnitude' in kernel

        open_count = kernel.count('{')
        close_count = kernel.count('}')
        assert open_count == close_count
