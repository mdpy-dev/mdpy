from __future__ import annotations

import cupy as cp
import numpy as np

from mdpy.force.force_term import ForceTerm
from mdpy.force.expressions.bonded import (
    harmonic_bond, charmm_angle, periodic_dihedral, harmonic_improper,
)


_PREAMBLE = r'''
__device__ __forceinline__ float3 make_f3(float x, float y, float z) {
    return make_float3(x, y, z);
}
__device__ __forceinline__ float3 sub_f3(float3 a, float3 b) {
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}
__device__ __forceinline__ float3 add_f3(float3 a, float3 b) {
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}
__device__ __forceinline__ float3 scale_f3(float3 a, float s) {
    return make_float3(a.x*s, a.y*s, a.z*s);
}
__device__ __forceinline__ float dot_f3(float3 a, float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
__device__ __forceinline__ float3 cross_f3(float3 a, float3 b) {
    return make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}
__device__ __forceinline__ float len_f3(float3 a) {
    return sqrtf(dot_f3(a,a));
}
__device__ __forceinline__ float3 pbc_ortho(float3 d, const float* box) {
    return make_float3(
        d.x - box[0]*roundf(d.x*box[3]),
        d.y - box[1]*roundf(d.y*box[4]),
        d.z - box[2]*roundf(d.z*box[5])
    );
}
__device__ __forceinline__ float3 load_pos(const float* __restrict__ p, int i) {
    return make_float3(p[i*3], p[i*3+1], p[i*3+2]);
}
__device__ __forceinline__ void add_force(float* __restrict__ f, int i, float3 v) {
    atomicAdd(&f[i*3],   v.x);
    atomicAdd(&f[i*3+1], v.y);
    atomicAdd(&f[i*3+2], v.z);
}
'''

_TWO_BODY_TEMPLATE = r'''
    // --- Phase: {name} (2-body) ---
    for (int idx = tid; idx < num_{name}; idx += stride) {{
        int a1 = {name}_idx[idx*2];
        int a2 = {name}_idx[idx*2+1];
        {param_loads}
        float3 delta = pbc_ortho(sub_f3(load_pos(pos,a2), load_pos(pos,a1)), box);
        float {geo0} = len_f3(delta);
        if ({geo0} < 1e-12f) continue;
        float inv_r = 1.0f / {geo0};
        {expression_fragment}
        float f_common = _result_force * inv_r;
        float3 fvec = scale_f3(delta, f_common);
        add_force(f, a1, fvec);
        add_force(f, a2, scale_f3(fvec, -1.0f));
        e += _result_energy;
    }}
'''

_THREE_BODY_TEMPLATE = r'''
    // --- Phase: {name} (3-body) ---
    for (int idx = tid; idx < num_{name}; idx += stride) {{
        int a1 = {name}_idx[idx*3];
        int a2 = {name}_idx[idx*3+1];
        int a3 = {name}_idx[idx*3+2];
        {param_loads}
        float3 r21 = pbc_ortho(sub_f3(load_pos(pos,a1), load_pos(pos,a2)), box);
        float3 r23 = pbc_ortho(sub_f3(load_pos(pos,a3), load_pos(pos,a2)), box);
        float l21 = len_f3(r21);
        float l23 = len_f3(r23);
        if (l21 < 1e-12f || l23 < 1e-12f) continue;
        float inv_l21 = 1.0f / l21;
        float inv_l23 = 1.0f / l23;
        float ct = dot_f3(r21, r23) * inv_l21 * inv_l23;
        ct = fmaxf(-1.0f, fminf(1.0f, ct));
        float {geo0} = acosf(ct);
        float3 r13v = pbc_ortho(sub_f3(load_pos(pos,a3), load_pos(pos,a1)), box);
        float {geo1} = len_f3(r13v);
        {expression_fragment}
        float neg_dEdtheta = -_result_force_0;
        float3 n = cross_f3(r21, r23);
        float3 c1 = cross_f3(r21, n);
        float lc1 = len_f3(c1);
        if (lc1 > 1e-12f) {{
            float inv = neg_dEdtheta / (lc1 * l21);
            float3 fv1 = scale_f3(c1, inv);
            add_force(f, a1, fv1);
            add_force(f, a2, scale_f3(fv1, -1.0f));
        }}
        float3 c3 = cross_f3(scale_f3(r23, -1.0f), n);
        float lc3 = len_f3(c3);
        if (lc3 > 1e-12f) {{
            float inv = neg_dEdtheta / (lc3 * l23);
            float3 fv3 = scale_f3(c3, inv);
            add_force(f, a3, fv3);
            add_force(f, a2, scale_f3(fv3, -1.0f));
        }}
        if ({geo1} >= 1e-12f) {{
            float inv_l13 = 1.0f / {geo1};
            float f_ub = _result_force_1 * inv_l13;
            float3 f13 = scale_f3(r13v, f_ub);
            add_force(f, a1, f13);
            add_force(f, a3, scale_f3(f13, -1.0f));
        }}
        e += _result_energy;
    }}
'''

_FOUR_BODY_TEMPLATE = r'''
    // --- Phase: {name} (4-body) ---
    for (int idx = tid; idx < num_{name}; idx += stride) {{
        int a1 = {name}_idx[idx*4];
        int a2 = {name}_idx[idx*4+1];
        int a3 = {name}_idx[idx*4+2];
        int a4 = {name}_idx[idx*4+3];
        {param_loads}
        float3 rab = pbc_ortho(sub_f3(load_pos(pos,a2), load_pos(pos,a1)), box);
        float3 rbc = pbc_ortho(sub_f3(load_pos(pos,a3), load_pos(pos,a2)), box);
        float3 rcd = pbc_ortho(sub_f3(load_pos(pos,a4), load_pos(pos,a3)), box);
        float lab = len_f3(rab), lbc = len_f3(rbc), lcd = len_f3(rcd);
        if (lab < 1e-12f || lbc < 1e-12f || lcd < 1e-12f) continue;
        float3 n1 = cross_f3(rab, rbc);
        float3 n2 = cross_f3(rbc, rcd);
        float dn = dot_f3(n1, n2);
        float drn = dot_f3(rab, n2);
        float {geo0} = atan2f(lbc * drn, dn);
        float n1s = dot_f3(n1, n1);
        float n2s = dot_f3(n2, n2);
        if (n1s < 1e-12f || n2s < 1e-12f) continue;
        {expression_fragment}
        float fv = -_result_force;
        float fa = fv * lbc / n1s;
        float fd = fv * lbc / n2s;
        float3 f_a = scale_f3(n1, -fa);
        float3 f_d = scale_f3(n2, fd);
        float3 voc = scale_f3(rbc, 0.5f);
        float loc = lbc * 0.5f;
        float ils = 1.0f / (loc * loc);
        float3 t1 = cross_f3(voc, f_d);
        float3 t2 = scale_f3(cross_f3(rcd, f_d), 0.5f);
        float3 t3 = scale_f3(cross_f3(scale_f3(rab, -1.0f), f_a), 0.5f);
        float3 st = scale_f3(add_f3(t1, add_f3(t2, t3)), -1.0f);
        float3 f_c = scale_f3(cross_f3(st, voc), ils);
        float3 f_b = scale_f3(add_f3(f_a, add_f3(f_c, f_d)), -1.0f);
        add_force(f, a1, f_a);
        add_force(f, a2, f_b);
        add_force(f, a3, f_c);
        add_force(f, a4, f_d);
        e += _result_energy;
    }}
'''

_MAIN_TEMPLATE = r'''
extern "C" __global__
void compute_bonded(
    const float* __restrict__ pos, float* __restrict__ f, float* __restrict__ energy_buf,
    const float* __restrict__ box,
    {kernel_params}
) {{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float e = 0.0f;

    {body_phases}

    for (int off = 16; off > 0; off >>= 1)
        e += __shfl_down_sync(0xffffffff, e, off);
    if ((threadIdx.x & 31) == 0)
        atomicAdd(energy_buf, e);
}}
'''


def _generate_param_loads(term_name, param_names, params_per_term):
    lines = []
    for i, pname in enumerate(param_names):
        lines.append(
            f'float {pname} = {term_name}_prm[idx*{params_per_term} + {i}];'
        )
    return '\n        '.join(lines)


def _assemble_kernel(term_specs):
    kernel_param_lines = []
    body_phases = []
    for spec in term_specs:
        name = spec['name']
        expr = spec['expression']
        n_atoms = spec['n_atoms']
        d_idx = spec['d_indices']
        d_prm = spec['d_params']
        count = spec['count']
        params_per_term = len(expr.param_names)

        kernel_param_lines.append(
            f'const int* __restrict__ {name}_idx, const float* __restrict__ {name}_prm, int num_{name}'
        )

        param_loads = _generate_param_loads(name, expr.param_names, params_per_term)

        if n_atoms == 2:
            geo_vars = {'geo0': expr.geometric_names[0]}
        elif n_atoms == 3:
            geo_vars = {'geo0': expr.geometric_names[0], 'geo1': expr.geometric_names[1]}
        else:
            geo_vars = {'geo0': expr.geometric_names[0]}

        if n_atoms == 2:
            template = _TWO_BODY_TEMPLATE
        elif n_atoms == 3:
            template = _THREE_BODY_TEMPLATE
        else:
            template = _FOUR_BODY_TEMPLATE

        phase = template.format(
            name=name,
            param_loads=param_loads,
            expression_fragment=expr.cuda_fragment,
            **geo_vars,
        )
        body_phases.append(phase)

    kernel_params = ',\n    '.join(kernel_param_lines)
    source = _PREAMBLE + _MAIN_TEMPLATE.format(
        kernel_params=kernel_params,
        body_phases='\n'.join(body_phases),
    )
    return source


class BondedForce(ForceTerm):
    name = 'bonded'

    def __init__(self):
        self._term_specs = []
        self._kernel = None
        self._kernel_source = None
        self._term_data = []
        self._num_sm = None

    def add_expression(self, expression, term_name):
        n_atoms_map = {2: 2, 3: 3, 4: 4}
        n_atoms = n_atoms_map.get(expression.body, 4)
        self._term_specs.append({
            'expression': expression,
            'term_name': term_name,
            'n_atoms': n_atoms,
        })

    def bind(self, topology, parameter_table):
        self._term_data = []
        for spec in self._term_specs:
            term_name = spec['term_name']
            expr = spec['expression']
            n_atoms = spec['n_atoms']

            indices_field = f'{term_name}_indices'
            indices = getattr(topology, indices_field, None)
            if indices is None or indices.shape[0] == 0:
                continue

            params_matrix = parameter_table.get_per_term(term_name)

            d_indices = cp.asarray(
                np.ascontiguousarray(indices.astype(np.int32).ravel())
            )
            d_params = cp.asarray(
                np.ascontiguousarray(params_matrix.astype(np.float32).ravel())
            )

            self._term_data.append({
                'name': term_name,
                'expression': expr,
                'n_atoms': n_atoms,
                'd_indices': d_indices,
                'd_params': d_params,
                'count': indices.shape[0],
            })

        self._kernel_source = _assemble_kernel(self._term_data)

    @classmethod
    def charmm(cls, topology, parameter_table):
        bonded = cls()
        bonded.add_expression(harmonic_bond, 'bond')
        bonded.add_expression(charmm_angle, 'angle')
        bonded.add_expression(periodic_dihedral, 'dihedral')
        bonded.add_expression(harmonic_improper, 'improper')
        bonded.bind(topology, parameter_table)
        return bonded

    def _ensure_compiled(self):
        if self._kernel is not None:
            return
        self._kernel = cp.RawKernel(self._kernel_source, 'compute_bonded')
        if self._num_sm is None:
            self._num_sm = cp.cuda.runtime.getDeviceProperties(0)['multiProcessorCount']

    def compute(self, gpu_context, tile_list=None):
        if not self._term_data:
            return
        self._ensure_compiled()

        total = sum(td['count'] for td in self._term_data)
        block_size = 128
        max_blocks = 6 * self._num_sm
        grid_size = max(min((total + block_size - 1) // block_size, max_blocks), 1)

        args = [
            gpu_context.d_positions,
            gpu_context.d_forces,
            gpu_context.d_energy,
            gpu_context.d_box_dims,
        ]
        for td in self._term_data:
            args.append(td['d_indices'])
            args.append(td['d_params'])
            args.append(np.int32(td['count']))

        self._kernel((grid_size,), (block_size,), tuple(args))
