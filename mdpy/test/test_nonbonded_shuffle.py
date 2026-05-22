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
