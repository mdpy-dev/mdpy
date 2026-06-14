from mdpy.force.nonbonded_transpiler import nonbonded_expression
from mdpy.force.primitives import param, distance

# NOTE: This expression uses a "decorate then override" pattern.
# The @nonbonded_expression decorator auto-compiles the energy via AD,
# but the module then overwrites energy_cuda, dEdr_cuda, and sets
# grad_cuda=None with hand-tuned CUDA code. The closed-form gradient
# -24*eps*(2*sr12-sr6)/r uses fewer live variables than the AD-generated
# power-chain gradient, reducing register pressure.


@nonbonded_expression
def lennard_jones(pos1, pos2, sigma=param, epsilon=param):
    r = distance(pos1, pos2)
    sigma_ratio = sigma / r
    sigma_ratio_6 = sigma_ratio ** 6
    return 4.0 * epsilon * (sigma_ratio_6 * sigma_ratio_6 - sigma_ratio_6)


lennard_jones.energy_cuda = '''\
        float sr = sigma * inv_dist;
        float sr3 = sr * sr * sr;
        float sr6 = sr3 * sr3;
        float sr12 = sr6 * sr6;
        float _result_energy = (4.0f * epsilon * (sr12 - sr6));
        float _lj_force = (-24.0f * epsilon * (sr12 + sr12 - sr6) * inv_dist);'''

lennard_jones.dEdr_cuda = '_lj_force'

lennard_jones._local_vars = {'sr', 'sr3', 'sr6', 'sr12', '_result_energy', '_lj_force'}
lennard_jones.grad_cuda = None
