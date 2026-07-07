import math

from mdpy.unit import EPSILON0
from mdpy.force.nonbonded_transpiler import nonbonded_expression
from mdpy.force.markers import scalar as scalar_marker

# Coulomb constant 1/(4*pi*epsilon0) in mdpy internal units (file-local).
COULOMB_CONST = 1.0 / (4.0 * math.pi * float(EPSILON0.value))
_COULOMB_CUDA = f"{COULOMB_CONST}f"

# NOTE: This expression uses a "decorate then override" pattern.
# The @nonbonded_expression decorator auto-compiles the energy via AD,
# but the module then overwrites energy_cuda, radial_force_cuda, and sets
# grad_cuda=None with hand-tuned CUDA code using a rational minimax
# polynomial approximation for erfc(). This is faster and more
# numerically stable than the AD-generated version. The auto-compiled
# output is intentionally discarded.


@nonbonded_expression
def screened_coulomb(pos1, pos2, charge1, charge2, alpha=scalar_marker):
    r = distance(pos1, pos2)
    return COULOMB_CONST * charge1 * charge2 * erfc(alpha * r) / r


screened_coulomb.energy_cuda = """\
        float qq = (charge1 * charge2);
        float alpha_r = (alpha * r);
        float z2 = (alpha_r * alpha_r);
        float z4 = (z2 * z2);
        float fd_a = ((0.0011193462567257629232f * z4) + 0.11583842382862377919f);
        float fd_b = ((0.014866955030185295499f * z4) + 0.50736591960530292870f);
        float fd_c = ((fd_a * z4) + 1.0f);
        float fd_d = ((fd_b * z2) + fd_c);
        float inv_fd = (1.0f / fd_d);
        float fn_a = (((-1.7357322914161492954e-8f) * z4) - 5.3401640219807709149e-5f);
        float fn_b = ((1.4703624142580877519e-6f * z4) + 1.0054721316683106153e-3f);
        float fn_c = ((fn_a * z4) - 1.927831726488838059e-2f);
        float fn_d = ((fn_b * z4) + 6.9670166153766424023e-2f);
        float fn_e = ((fn_c * z4) - 0.75225204789749321333f);
        float fn_f = ((fn_d * z2) + fn_e);
        float corr = (fn_f * inv_fd);
        float alpha3 = ((alpha * alpha) * alpha);
        float _coulomb_force = (((-__MDPY_COULOMB__) * qq) * ((inv_dist * inv_dist) + ((alpha3 * r) * corr)));
        float erf_val = erff(alpha_r);
        float _result_energy = (((__MDPY_COULOMB__ * qq) * (1.0f - erf_val)) * inv_dist);""".replace(
    "__MDPY_COULOMB__", _COULOMB_CUDA
)

screened_coulomb.radial_force_cuda = "_coulomb_force"

screened_coulomb._local_vars = {
    "qq",
    "alpha_r",
    "z2",
    "z4",
    "_result_energy",
    "fd_a",
    "fd_b",
    "fd_c",
    "fd_d",
    "inv_fd",
    "fn_a",
    "fn_b",
    "fn_c",
    "fn_d",
    "fn_e",
    "fn_f",
    "corr",
    "alpha3",
    "_coulomb_force",
    "erf_val",
}
screened_coulomb.grad_cuda = None
