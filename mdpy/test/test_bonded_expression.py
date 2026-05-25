import pytest


class TestBondedTranspiler:

    def test_harmonic_bond_transpilation(self):
        from mdpy.force.bonded_force import bonded_expression

        @bonded_expression(body=2)
        def harmonic_bond(r, k=0.0, r0=0.0):
            dr = r - r0
            return k * dr * dr, 2.0 * k * dr

        assert harmonic_bond.body == 2
        assert harmonic_bond.parameter_names == ['k', 'r0']
        assert harmonic_bond.geometric_names == ['r']
        assert '_result_energy' in harmonic_bond.cuda_fragment
        assert '_result_force' in harmonic_bond.cuda_fragment

    def test_periodic_dihedral_transpilation(self):
        from mdpy.force.bonded_force import bonded_expression

        @bonded_expression(body=4)
        def periodic_dihedral(phi, k=0.0, n=0.0, delta=0.0):
            return k * (1.0 + cos(n * phi - delta)), -k * n * sin(n * phi - delta)

        assert periodic_dihedral.body == 4
        assert 'k' in periodic_dihedral.parameter_names
        assert 'n' in periodic_dihedral.parameter_names
        assert 'delta' in periodic_dihedral.parameter_names
        assert 'cosf' in periodic_dihedral.cuda_fragment
        assert 'sinf' in periodic_dihedral.cuda_fragment

    def test_three_return_values(self):
        from mdpy.force.bonded_force import bonded_expression

        @bonded_expression(body=3)
        def test_angle(theta, r13, k=0.0, theta0=0.0):
            dt = theta - theta0
            return k * dt * dt, 2.0 * k * dt, 0.0

        assert '_result_force_0' in test_angle.cuda_fragment
        assert '_result_force_1' in test_angle.cuda_fragment
