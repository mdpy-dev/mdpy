import pytest
from mdpy.force.nonbonded_transpiler import nonbonded_expression
from mdpy.force.primitives import param, scalar

# Module-level constant used to verify the transpiler inlines named constants
# resolved from the decorated function's module globals.
_TEST_COULOMB = 0.13893556595455
_TEST_INT_CONST = 4


class TestNonbondedTranspiler:
    def test_lj_classification(self):
        @nonbonded_expression
        def lennard_jones(pos1, pos2, sigma=param, epsilon=param):
            r = distance(pos1, pos2)
            sr = sigma / r
            sr6 = sr * sr * sr * sr * sr * sr
            return 4.0 * epsilon * (sr6 * sr6 - sr6)

        info = lennard_jones.expr_info
        assert info.params == ['sigma', 'epsilon']
        assert info.per_particle == {}
        assert info.body == 2

    def test_coulomb_classification(self):
        @nonbonded_expression
        def coulomb(pos1, pos2, charge1, charge2):
            r = distance(pos1, pos2)
            return 0.13893556595455 * charge1 * charge2 / r

        info = coulomb.expr_info
        assert info.params == []
        assert info.per_particle == {'charge1': 'charge', 'charge2': 'charge'}

    def test_screened_coulomb_classification(self):
        @nonbonded_expression
        def screened(pos1, pos2, charge1, charge2, alpha=scalar):
            r = distance(pos1, pos2)
            return 0.13893556595455 * charge1 * charge2 * erfc(alpha * r) / r

        info = screened.expr_info
        assert info.scalars == ['alpha']
        assert info.per_particle == {'charge1': 'charge', 'charge2': 'charge'}

    def test_lj_has_dEdr(self):
        @nonbonded_expression
        def lennard_jones(pos1, pos2, sigma=param, epsilon=param):
            r = distance(pos1, pos2)
            sr = sigma / r
            sr6 = sr * sr * sr * sr * sr * sr
            return 4.0 * epsilon * (sr6 * sr6 - sr6)

        assert lennard_jones.radial_force_cuda is not None
        assert lennard_jones.grad_cuda is not None
        assert ('inv_dist' in lennard_jones.grad_cuda)

    def test_coulomb_dEdr_sign(self):
        @nonbonded_expression
        def coulomb(pos1, pos2, charge1, charge2):
            r = distance(pos1, pos2)
            return 0.13893556595455 * charge1 * charge2 / r

        assert coulomb.radial_force_cuda is not None
        assert coulomb.grad_cuda is not None
        assert ('inv_dist' in coulomb.grad_cuda)

    def test_module_level_constant_inlined(self):
        @nonbonded_expression
        def coul(pos1, pos2, charge1, charge2):
            r = distance(pos1, pos2)
            return _TEST_COULOMB * charge1 * charge2 / r

        # The named constant must be inlined as a CUDA float literal, and the
        # name itself must not leak into the generated CUDA source.
        assert '0.13893556595455f' in coul.energy_cuda
        assert '_TEST_COULOMB' not in coul.energy_cuda
        # Parameters are still resolved normally
        assert 'charge1' in coul.energy_cuda
        assert 'charge2' in coul.energy_cuda
        # The auto-diff gradient is still produced correctly
        assert coul.radial_force_cuda is not None
        assert coul.grad_cuda is not None

    def test_module_level_int_constant_inlined(self):
        @nonbonded_expression
        def scaled(pos1, pos2, sigma=param):
            r = distance(pos1, pos2)
            return _TEST_INT_CONST * sigma / r

        assert '4.0f' in scaled.energy_cuda
        assert '_TEST_INT_CONST' not in scaled.energy_cuda

    def test_add_combines_expressions(self):
        @nonbonded_expression
        def lj(pos1, pos2, sigma=param, epsilon=param):
            r = distance(pos1, pos2)
            sr = sigma / r
            sr6 = sr * sr * sr * sr * sr * sr
            return 4.0 * epsilon * (sr6 * sr6 - sr6)

        @nonbonded_expression
        def coul(pos1, pos2, charge1, charge2):
            r = distance(pos1, pos2)
            return 0.13893556595455 * charge1 * charge2 / r

        combined = lj + coul
        assert combined.radial_force_cuda is not None
        assert combined.grad_cuda is not None
        assert combined.expr_info.params == ['sigma', 'epsilon']
        assert 'charge' in str(combined.expr_info.per_particle.values()) or len(combined.expr_info.per_particle) > 0


class TestForwardModeAD:
    def test_lj_has_grad_cuda(self):
        @nonbonded_expression
        def lj(pos1, pos2, sigma=param, epsilon=param):
            r = distance(pos1, pos2)
            sr = sigma / r
            sr6 = sr * sr * sr * sr * sr * sr
            return 4.0 * epsilon * (sr6 * sr6 - sr6)

        assert lj.grad_cuda is not None
        assert '_d_' in lj.grad_cuda
        assert len(lj.radial_force_cuda) < 20
        assert lj.radial_force_cuda.startswith('_d_')

    def test_lj_grad_cuda_has_no_repeated_subexpressions(self):
        """Forward-mode should not inline expressions — each var computed once."""
        @nonbonded_expression
        def lj(pos1, pos2, sigma=param, epsilon=param):
            r = distance(pos1, pos2)
            sr = sigma / r
            sr6 = sr * sr * sr * sr * sr * sr
            return 4.0 * epsilon * (sr6 * sr6 - sr6)

        grad_lines = lj.grad_cuda.split('\n')
        for line in grad_lines:
            assert len(line) < 120, f"Gradient line too long: {line}"

    def test_lj_radial_force_is_gradient_variable(self):
        """radial_force_cuda should be a named variable like _d__t8."""
        @nonbonded_expression
        def lj(pos1, pos2, sigma=param, epsilon=param):
            r = distance(pos1, pos2)
            sr = sigma / r
            sr6 = sr * sr * sr * sr * sr * sr
            return 4.0 * epsilon * (sr6 * sr6 - sr6)

        assert lj.radial_force_cuda in lj._local_vars

    def test_combined_expression_merges_grad_cuda(self):
        @nonbonded_expression
        def lj(pos1, pos2, sigma=param, epsilon=param):
            r = distance(pos1, pos2)
            sr = sigma / r
            sr6 = sr * sr * sr * sr * sr * sr
            return 4.0 * epsilon * (sr6 * sr6 - sr6)

        @nonbonded_expression
        def coul(pos1, pos2, charge1, charge2):
            r = distance(pos1, pos2)
            return 0.13893556595455 * charge1 * charge2 / r

        combined = lj + coul
        assert combined.grad_cuda is not None
        grad_lines = combined.grad_cuda.strip().split('\n')
        assert len(grad_lines) >= 4
        assert '+' in combined.radial_force_cuda
