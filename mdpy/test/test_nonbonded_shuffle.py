import pytest


class TestCodeGenerators:
    @pytest.fixture
    def params(self):
        return ['sigma', 'epsilon', 'charge']

    def test_param_load_i(self, params):
        from mdpy.force.nonbonded_force import _generate_param_load_i
        code = _generate_param_load_i(params)
        assert 'float sigma_i = 0.0f' in code
        assert 'sigma_i = sigma[gi]' in code
        assert 'sigma_i_14 = sigma_14[gi]' in code
        assert 'epsilon_i = epsilon[gi]' in code
        assert 'charge_i_14 = charge_14[gi]' in code

    def test_param_load_j_init(self, params):
        from mdpy.force.nonbonded_force import _generate_param_load_j_init
        code = _generate_param_load_j_init(params)
        assert 'float sigma_j = 0.0f' in code
        assert 'sigma_j = sigma[gj_init]' in code
        assert 'sigma_j_14 = sigma_14[gj_init]' in code

    def test_shuffle_warp_data(self, params):
        from mdpy.force.nonbonded_force import _generate_shuffle_warp_data
        code = _generate_shuffle_warp_data(params)
        assert '__shfl_sync(0xffffffff, shfl_px, (tgx + 1) & 31)' in code
        assert '__shfl_sync(0xffffffff, shfl_fx, (tgx + 1) & 31)' in code
        assert '__shfl_sync(0xffffffff, sigma_j, (tgx + 1) & 31)' in code
        assert '__shfl_sync(0xffffffff, sigma_j_14, (tgx + 1) & 31)' in code
        assert '__shfl_sync(0xffffffff, charge_j_14, (tgx + 1) & 31)' in code

    def test_param_select(self, params):
        from mdpy.force.nonbonded_force import _generate_param_select
        code = _generate_param_select(params)
        assert 'sigma_i_saved = sigma_i' in code
        assert 'if (is_14) sigma_i = sigma_i_14' in code
        assert 'sigma_j_saved = sigma_j' in code
        assert 'if (is_14) sigma_j = sigma_j_14' in code
        assert 'if (is_14) epsilon_i = epsilon_i_14' in code

    def test_param_restore(self, params):
        from mdpy.force.nonbonded_force import _generate_param_restore
        code = _generate_param_restore(params)
        assert 'if (is_14) sigma_i = sigma_i_saved' in code
        assert 'if (is_14) sigma_j = sigma_j_saved' in code
        assert 'if (is_14) epsilon_i = epsilon_i_saved' in code
        assert 'if (is_14) charge_j = charge_j_saved' in code


from mdpy.force.expressions.lennard_jones import lennard_jones
from mdpy.force.expressions.coulomb import coulomb


class TestCrossTileKernelV2:
    @pytest.fixture
    def combined_expr(self):
        return lennard_jones + coulomb

    def test_kernel_has_static_dispatch(self, combined_expr):
        source = combined_expr.assemble_cross_tile_kernel()
        assert 'total_warps' in source
        assert 'warp_id * num_cross / total_warps' in source

    def test_kernel_has_shfl_sync(self, combined_expr):
        source = combined_expr.assemble_cross_tile_kernel()
        assert '__shfl_sync' in source

    def test_kernel_has_rotate_right(self, combined_expr):
        source = combined_expr.assemble_cross_tile_kernel()
        assert '(tgx + 1) & 31' in source

    def test_kernel_has_forces_register(self, combined_expr):
        source = combined_expr.assemble_cross_tile_kernel()
        assert 'shfl_fx' in source
        assert 'force_x' in source

    def test_kernel_has_32_step_loop(self, combined_expr):
        source = combined_expr.assemble_cross_tile_kernel()
        assert 'j < 32' in source

    def test_kernel_no_dynamic_counter(self, combined_expr):
        source = combined_expr.assemble_cross_tile_kernel()
        assert 'tile_counter' not in source

    def test_kernel_no_shared_memory(self, combined_expr):
        source = combined_expr.assemble_cross_tile_kernel()
        assert '__shared__' not in source

    def test_kernel_has_exclusion_prerotate(self, combined_expr):
        source = combined_expr.assemble_cross_tile_kernel()
        assert 'excl >> tgx' in source
        assert 'excl << (32 - tgx)' in source

    def test_kernel_exclusion_convention(self, combined_expr):
        source = combined_expr.assemble_cross_tile_kernel()
        assert '(excl & 0x1) != 0' in source

    def test_kernel_has_atomicAdd_force(self, combined_expr):
        source = combined_expr.assemble_cross_tile_kernel()
        assert 'atomicAdd(&forces[gi * 3' in source
        assert 'atomicAdd(&forces[gj * 3' in source

    def test_kernel_has_warp_energy_reduce(self, combined_expr):
        source = combined_expr.assemble_cross_tile_kernel()
        assert '__shfl_down_sync' in source
        assert 'atomicAdd(energy_buffer, energy)' in source

    def test_kernel_has_param_select_both(self, combined_expr):
        source = combined_expr.assemble_cross_tile_kernel()
        assert 'sigma_i_saved' in source
        assert 'if (is_14) sigma_i = sigma_i_14' in source

    def test_kernel_valid_braces(self, combined_expr):
        source = combined_expr.assemble_cross_tile_kernel()
        assert source.count('{') == source.count('}')


class TestSelfTileKernelV2:
    @pytest.fixture
    def combined_expr(self):
        return lennard_jones + coulomb

    def test_self_kernel_has_32_step_loop(self, combined_expr):
        source = combined_expr.assemble_self_tile_kernel()
        assert 'j < 32' in source

    def test_self_kernel_has_broadcast(self, combined_expr):
        source = combined_expr.assemble_self_tile_kernel()
        assert '__shfl_sync(0xffffffff, px_i, j)' in source

    def test_self_kernel_no_naive_pair_loop(self, combined_expr):
        source = combined_expr.assemble_self_tile_kernel()
        assert 'linear = tid * 2 + iter' not in source
        assert 'linear >= 496' not in source

    def test_self_kernel_no_upper_triangle(self, combined_expr):
        source = combined_expr.assemble_self_tile_kernel()
        assert 'j > tgx' not in source

    def test_self_kernel_has_j_neq_tgx(self, combined_expr):
        source = combined_expr.assemble_self_tile_kernel()
        assert 'j != tgx' in source

    def test_self_kernel_has_half_energy(self, combined_expr):
        source = combined_expr.assemble_self_tile_kernel()
        assert '0.5f * energy_val' in source

    def test_self_kernel_no_shared_mem_positions(self, combined_expr):
        source = combined_expr.assemble_self_tile_kernel()
        assert 'smem_pos' not in source

    def test_self_kernel_has_exclusion_shift(self, combined_expr):
        source = combined_expr.assemble_self_tile_kernel()
        assert 'excl >>= 1' in source

    def test_self_kernel_has_only_i_force(self, combined_expr):
        source = combined_expr.assemble_self_tile_kernel()
        assert 'force_x' in source
        lines = source.split('\n')
        shfl_force_lines = [l for l in lines if 'shfl_f' in l and '__shfl_sync' not in l]
        assert len(shfl_force_lines) == 0

    def test_self_kernel_has_warp_energy_reduce(self, combined_expr):
        source = combined_expr.assemble_self_tile_kernel()
        assert '__shfl_down_sync' in source

    def test_self_kernel_has_param_broadcast(self, combined_expr):
        source = combined_expr.assemble_self_tile_kernel()
        assert '__shfl_sync(0xffffffff, sigma_i, j)' in source
        assert '__shfl_sync(0xffffffff, sigma_i_14, j)' in source

    def test_self_kernel_valid_braces(self, combined_expr):
        source = combined_expr.assemble_self_tile_kernel()
        assert source.count('{') == source.count('}')
