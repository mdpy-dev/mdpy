import pytest
from mdpy.force.expressions.lennard_jones import lennard_jones
from mdpy.force.expressions.coulomb import coulomb


class TestExclusionTileKernel:
    @pytest.fixture
    def combined_expr(self):
        return lennard_jones + coulomb

    def test_kernel_has_extern_c(self, combined_expr):
        source = combined_expr.assemble_tile_kernel()
        assert 'extern "C" __global__' in source

    def test_kernel_has_tile_kernel_name(self, combined_expr):
        source = combined_expr.assemble_tile_kernel()
        assert 'void tile_kernel' in source

    def test_kernel_has_block_atoms(self, combined_expr):
        source = combined_expr.assemble_tile_kernel()
        assert '__restrict__ block_atoms' in source

    def test_kernel_has_exclusion_masks(self, combined_expr):
        source = combined_expr.assemble_tile_kernel()
        assert 'exclusion_masks' in source
        assert 'scaling_masks' in source

    def test_kernel_has_warp_dispatch(self, combined_expr):
        source = combined_expr.assemble_tile_kernel()
        assert 'total_warps' in source
        assert 'warp_id' in source

    def test_kernel_has_shfl_sync(self, combined_expr):
        source = combined_expr.assemble_tile_kernel()
        assert '__shfl_sync' in source

    def test_kernel_has_is_14(self, combined_expr):
        source = combined_expr.assemble_tile_kernel()
        assert 'is_14' in source

    def test_kernel_valid_braces(self, combined_expr):
        source = combined_expr.assemble_tile_kernel()
        assert source.count('{') == source.count('}')

    def test_kernel_has_param_select(self, combined_expr):
        source = combined_expr.assemble_tile_kernel()
        assert 'charge_i_saved' in source


class TestNonbondedForceCompute:
    def test_compute_no_tile_counter(self):
        from mdpy.force.nonbonded_force import NonbondedForce
        nf = NonbondedForce(lennard_jones + coulomb)
        assert not hasattr(nf, '_d_tile_counter') or nf._d_tile_counter is None
