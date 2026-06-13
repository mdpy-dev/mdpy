from mdpy.force.nonbonded_transpiler import nonbonded_expression
from mdpy.force.primitives import scalar as scalar_marker

# NOTE: This expression uses a "decorate then override" pattern.
# The @nonbonded_expression decorator auto-compiles the energy via AD,
# but the module then overwrites energy_cuda, dEdr_cuda, and sets
# grad_cuda=None with hand-tuned CUDA code using a rational minimax
# polynomial approximation for erfc(). This is faster and more
# numerically stable than the AD-generated version. The auto-compiled
# output is intentionally discarded.


@nonbonded_expression
def screened_coulomb(pos1, pos2, charge1, charge2, alpha=scalar_marker):
    r = distance(pos1, pos2)
    return 0.13893556595455 * charge1 * charge2 * erfc(alpha * r) / r


screened_coulomb.energy_cuda = '''\
        float qq = 0.13893556595455f * charge1 * charge2;
        float alpha_r = alpha * r;
        float z2 = alpha_r * alpha_r;
        float z4 = z2 * z2;
        float inv_r_local = 1.0f / r;
        float _result_energy = qq * erfcf(alpha_r) * inv_r_local;
        float fd_a = 0.0011193462567257629232f * z4 + 0.11583842382862377919f;
        float fd_b = 0.014866955030185295499f * z4 + 0.50736591960530292870f;
        float fd_c = fd_a * z4 + 1.0f;
        float fd_d = fd_b * z2 + fd_c;
        float fn_a = -1.7357322914161492954e-8f * z4 - 5.3401640219807709149e-5f;
        float fn_b = 1.4703624142580877519e-6f * z4 + 1.0054721316683106153e-3f;
        float fn_c = fn_a * z4 - 1.927831726488838059e-2f;
        float fn_d = fn_b * z4 + 6.9670166153766424023e-2f;
        float fn_e = fn_c * z4 - 0.75225204789749321333f;
        float fn_f = fn_d * z2 + fn_e;
        float _pme_corr = fn_f / fd_d;
        float alpha3 = alpha * alpha * alpha;'''

screened_coulomb.dEdr_cuda = '(-qq * (inv_r_local * inv_r_local + alpha3 * r * _pme_corr))'

screened_coulomb._local_vars = {
    'qq', 'alpha_r', 'z2', 'z4', 'inv_r_local', '_result_energy',
    'fd_a', 'fd_b', 'fd_c', 'fd_d',
    'fn_a', 'fn_b', 'fn_c', 'fn_d', 'fn_e', 'fn_f',
    '_pme_corr', 'alpha3',
}
screened_coulomb.grad_cuda = None
