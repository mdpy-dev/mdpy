import pytest
from mdpy.force.bonded_transpiler import bonded_expression


class TestBondedTranspiler:
    def test_harmonic_bond_classification(self):
        @bonded_expression(body=2)
        def harmonic_bond(pos1, pos2, k=1, r0=1):
            pass
        expr = harmonic_bond
        assert expr._expr_info.body == 2
        assert expr._expr_info.positions == ['pos1', 'pos2']
        assert expr._expr_info.params == ['k', 'r0']
        assert expr._expr_info.per_particle == {}

    def test_harmonic_bond_cuda_contains_energy(self):
        @bonded_expression(body=2)
        def harmonic_bond(pos1, pos2, k=1, r0=1):
            r = distance(pos1, pos2)
            dr = r - r0
            return k * dr * dr
        assert '_result_energy' in harmonic_bond.cuda_fragment

    def test_harmonic_bond_cuda_contains_ad_gradient(self):
        @bonded_expression(body=2)
        def harmonic_bond(pos1, pos2, k=1, r0=1):
            r = distance(pos1, pos2)
            dr = r - r0
            return k * dr * dr
        assert '_grad_r' in harmonic_bond.cuda_fragment or 'grad' in harmonic_bond.cuda_fragment

    def test_charmm_angle_classification(self):
        from mdpy.force.primitives import param
        @bonded_expression(body=3)
        def charmm_angle(pos1, pos2, pos3, k=param, theta0=param, k_ub=param, r_ub=param):
            theta = angle(pos1, pos2, pos3)
            r13 = distance(pos1, pos3)
            dt = theta - theta0
            dr13 = r13 - r_ub
            return k * dt * dt + k_ub * dr13 * dr13
        info = charmm_angle._expr_info
        assert info.body == 3
        assert info.params == ['k', 'theta0', 'k_ub', 'r_ub']

    def test_periodic_dihedral_cuda(self):
        from mdpy.force.primitives import param
        @bonded_expression(body=4)
        def periodic_dihedral(pos1, pos2, pos3, pos4, k=param, n=param, delta=param):
            phi = dihedral(pos1, pos2, pos3, pos4)
            return k * (1.0 + cos(n * phi - delta))
        assert 'cosf' in periodic_dihedral.cuda_fragment
        assert '_result_energy' in periodic_dihedral.cuda_fragment

    def test_harmonic_improper_cuda(self):
        from mdpy.force.primitives import param
        @bonded_expression(body=4)
        def harmonic_improper(pos1, pos2, pos3, pos4, k=param, psi0=param):
            psi = dihedral(pos1, pos2, pos3, pos4)
            dp = psi - psi0
            return k * dp * dp
        assert '_result_energy' in harmonic_improper.cuda_fragment


def test_point_marker_classification():
    from mdpy.force.primitives import param, point

    @bonded_expression(body=1)
    def expr(p1, ref=point, k=param):
        return k

    info = expr._expr_info
    assert info.body == 1
    assert info.positions == ['p1']
    assert info.params == ['k', 'ref_x', 'ref_y', 'ref_z']
    assert info.per_particle == {}
