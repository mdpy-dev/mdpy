from __future__ import annotations

import ast
import cupy as cp
import inspect
import numpy as np
import textwrap

from mdpy import env
from mdpy.force.force_term import ForceTerm

_MATH_FUNCTIONS = {
    'sqrt': 'sqrtf', 'sin': 'sinf', 'cos': 'cosf',
    'tan': 'tanf', 'acos': 'acosf', 'asin': 'asinf',
    'atan': 'atanf', 'atan2': 'atan2f', 'exp': 'expf',
    'log': 'logf', 'abs': 'fabsf', 'floor': 'floorf',
    'ceil': 'ceilf', 'min': 'fminf', 'max': 'fmaxf',
}

import re as _re

def _extract_trailing_digit(name):
    m = _re.search(r'(\d+)$', name)
    if m:
        return int(m.group(1))
    return None

_REMAP_INDICES_KERNEL = r"""
extern "C" __global__
void remap_indices_kernel(
    const int* __restrict__ d_remap,
    int* __restrict__ d_indices,
    int num_indices
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_indices) return;
    d_indices[i] = d_remap[d_indices[i]];
}
"""


class BondedExpression:
    def __init__(self, body, parameter_names, geometric_names, cuda_fragment, local_variables):
        self.body = body
        self.parameter_names = parameter_names
        self.geometric_names = geometric_names
        self.cuda_fragment = cuda_fragment
        self.local_variables = local_variables


def bonded_expression(body):
    def decorator(func):
        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
        function_definition = tree.body[0]
        func_name = function_definition.name
        args = function_definition.args.args

        geometric_names = []
        parameter_names = []
        for i, arg in enumerate(args):
            if arg.arg == 'self':
                continue
            if i < len(args) and not _has_parameter_default(function_definition, arg.arg):
                geometric_names.append(arg.arg)
            else:
                parameter_names.append(arg.arg)

        transpiler = _BondedTranspiler(geometric_names, parameter_names)
        cuda_fragment = transpiler.transpile(function_definition)
        local_variables = transpiler.local_variables

        return BondedExpression(
            body=body,
            parameter_names=parameter_names,
            geometric_names=geometric_names,
            cuda_fragment=cuda_fragment,
            local_variables=local_variables,
        )
    return decorator


def _has_parameter_default(function_definition, arg_name):
    defaults = function_definition.args.defaults
    args = function_definition.args.args
    number_defaults = len(defaults)
    number_arguments = len(args)
    for i, arg in enumerate(args):
        if arg.arg == arg_name:
            return i >= (number_arguments - number_defaults)
    return False


class _BondedTranspiler(ast.NodeVisitor):
    def __init__(self, geometric_names, parameter_names):
        self.geometric_names = geometric_names
        self.parameter_names = parameter_names
        self.lines = []
        self.local_variables = set()

    def transpile(self, function_definition):
        for stmt in function_definition.body:
            self.visit(stmt)
        return '\n'.join(self.lines)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.local_variables.add(target.id)
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        value = self._expression(node.value)
        for name in names:
            self.lines.append(f'float {name} = {value};')

    def visit_Return(self, node):
        if isinstance(node.value, ast.Tuple):
            elements = node.value.elts
            if len(elements) == 2:
                self.lines.append(f'float _result_energy = {self._expression(elements[0])};')
                self.lines.append(f'float _result_force = {self._expression(elements[1])};')
            elif len(elements) == 3:
                self.lines.append(f'float _result_energy = {self._expression(elements[0])};')
                self.lines.append(f'float _result_force_0 = {self._expression(elements[1])};')
                self.lines.append(f'float _result_force_1 = {self._expression(elements[2])};')
        else:
            self.lines.append(f'float _result_energy = {self._expression(node.value)};')

    def _expression(self, node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, float):
                return f'{node.value}f'
            return str(float(node.value)) + 'f'
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.BinOp):
            left = self._expression(node.left)
            right = self._expression(node.right)
            op = self._binary_operator(node.op)
            return f'({left} {op} {right})'
        if isinstance(node, ast.UnaryOp):
            operand = self._expression(node.operand)
            if isinstance(node.op, ast.USub):
                return f'(-{operand})'
            if isinstance(node.op, ast.UAdd):
                return f'(+{operand})'
        if isinstance(node, ast.Call):
            func_name = self._call_name(node.func)
            if func_name in _MATH_FUNCTIONS:
                cuda_name = _MATH_FUNCTIONS[func_name]
                args = ', '.join(self._expression(a) for a in node.args)
                return f'{cuda_name}({args})'
            if func_name == 'pow':
                base = self._expression(node.args[0])
                exp_node = node.args[1]
                if isinstance(exp_node, ast.Constant):
                    exp_val = exp_node.value
                    if exp_val == 0:
                        return '1.0f'
                    if exp_val == 1:
                        return base
                    if exp_val == 2:
                        return f'({base} * {base})'
                args = ', '.join(self._expression(a) for a in node.args)
                return f'powf({args})'
            args = ', '.join(self._expression(a) for a in node.args)
            return f'{func_name}({args})'
        if isinstance(node, ast.BoolOp):
            op = ' && ' if isinstance(node.op, ast.And) else ' || '
            return op.join(self._expression(v) for v in node.values)
        if isinstance(node, ast.Compare):
            left = self._expression(node.left)
            parts = []
            for op, comparator in zip(node.ops, node.comparators):
                right = self._expression(comparator)
                parts.append(f'({left} {self._comparison_operator(op)} {right})')
            return ' && '.join(parts)
        return '0.0f'

    def _binary_operator(self, op):
        ops = {
            ast.Add: '+', ast.Sub: '-', ast.Mult: '*',
            ast.Div: '/', ast.Mod: '%',
        }
        return ops.get(type(op), '?')

    def _comparison_operator(self, op):
        ops = {
            ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>',
            ast.GtE: '>=', ast.Eq: '==', ast.NotEq: '!=',
        }
        return ops.get(type(op), '?')

    def _call_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ''


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
__device__ __forceinline__ float3 pbc_wrap_vec(float3 d, const float* pbc_inv, const float* pbc_matrix) {
    float fx = d.x*pbc_inv[0] + d.y*pbc_inv[3] + d.z*pbc_inv[6];
    float fy = d.x*pbc_inv[1] + d.y*pbc_inv[4] + d.z*pbc_inv[7];
    float fz = d.x*pbc_inv[2] + d.y*pbc_inv[5] + d.z*pbc_inv[8];
    fx -= roundf(fx); fy -= roundf(fy); fz -= roundf(fz);
    return make_float3(
        fx*pbc_matrix[0] + fy*pbc_matrix[3] + fz*pbc_matrix[6],
        fx*pbc_matrix[1] + fy*pbc_matrix[4] + fz*pbc_matrix[7],
        fx*pbc_matrix[2] + fy*pbc_matrix[5] + fz*pbc_matrix[8]
    );
}
__device__ __forceinline__ float3 load_pos(
    const float* __restrict__ px,
    const float* __restrict__ py,
    const float* __restrict__ pz,
    int i
) {
    return make_float3(px[i], py[i], pz[i]);
}
__device__ __forceinline__ void add_force(
    float* __restrict__ fx,
    float* __restrict__ fy,
    float* __restrict__ fz,
    int i, float3 v
) {
    atomicAdd(&fx[i], v.x);
    atomicAdd(&fy[i], v.y);
    atomicAdd(&fz[i], v.z);
}
'''

_TWO_BODY_TEMPLATE = r'''
    // --- Phase: {name} (2-body) ---
    for (int idx = tid; idx < num_{name}; idx += stride) {{
        int a1 = {name}_idx[idx*2];
        int a2 = {name}_idx[idx*2+1];
        {param_loads}
        float3 delta = pbc_wrap_vec(sub_f3(load_pos(pos_x,pos_y,pos_z,a2), load_pos(pos_x,pos_y,pos_z,a1)), pbc_inv, pbc_matrix);
        float {geo0} = len_f3(delta);
        if ({geo0} < 1e-12f) continue;
        float inv_r = 1.0f / {geo0};
        {expression_fragment}
        float f_common = _result_force * inv_r;
        float3 fvec = scale_f3(delta, f_common);
        add_force(f_x,f_y,f_z, a1, fvec);
        add_force(f_x,f_y,f_z, a2, scale_f3(fvec, -1.0f));
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
        float3 r21 = pbc_wrap_vec(sub_f3(load_pos(pos_x,pos_y,pos_z,a1), load_pos(pos_x,pos_y,pos_z,a2)), pbc_inv, pbc_matrix);
        float3 r23 = pbc_wrap_vec(sub_f3(load_pos(pos_x,pos_y,pos_z,a3), load_pos(pos_x,pos_y,pos_z,a2)), pbc_inv, pbc_matrix);
        float l21 = len_f3(r21);
        float l23 = len_f3(r23);
        if (l21 < 1e-12f || l23 < 1e-12f) continue;
        float inv_l21 = 1.0f / l21;
        float inv_l23 = 1.0f / l23;
        float ct = dot_f3(r21, r23) * inv_l21 * inv_l23;
        ct = fmaxf(-1.0f, fminf(1.0f, ct));
        float {geo0} = acosf(ct);
        float3 r13v = pbc_wrap_vec(sub_f3(load_pos(pos_x,pos_y,pos_z,a3), load_pos(pos_x,pos_y,pos_z,a1)), pbc_inv, pbc_matrix);
        float {geo1} = len_f3(r13v);
        {expression_fragment}
        float neg_dEdtheta = -_result_force_0;
        float3 n = cross_f3(r21, r23);
        float3 c1 = cross_f3(r21, n);
        float lc1 = len_f3(c1);
        if (lc1 > 1e-12f) {{
            float inv = neg_dEdtheta / (lc1 * l21);
            float3 fv1 = scale_f3(c1, inv);
            add_force(f_x,f_y,f_z, a1, fv1);
            add_force(f_x,f_y,f_z, a2, scale_f3(fv1, -1.0f));
        }}
        float3 c3 = cross_f3(scale_f3(r23, -1.0f), n);
        float lc3 = len_f3(c3);
        if (lc3 > 1e-12f) {{
            float inv = neg_dEdtheta / (lc3 * l23);
            float3 fv3 = scale_f3(c3, inv);
            add_force(f_x,f_y,f_z, a3, fv3);
            add_force(f_x,f_y,f_z, a2, scale_f3(fv3, -1.0f));
        }}
        if ({geo1} >= 1e-12f) {{
            float inv_l13 = 1.0f / {geo1};
            float f_ub = _result_force_1 * inv_l13;
            float3 f13 = scale_f3(r13v, f_ub);
            add_force(f_x,f_y,f_z, a1, f13);
            add_force(f_x,f_y,f_z, a3, scale_f3(f13, -1.0f));
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
        float3 rab = pbc_wrap_vec(sub_f3(load_pos(pos_x,pos_y,pos_z,a2), load_pos(pos_x,pos_y,pos_z,a1)), pbc_inv, pbc_matrix);
        float3 rbc = pbc_wrap_vec(sub_f3(load_pos(pos_x,pos_y,pos_z,a3), load_pos(pos_x,pos_y,pos_z,a2)), pbc_inv, pbc_matrix);
        float3 rcd = pbc_wrap_vec(sub_f3(load_pos(pos_x,pos_y,pos_z,a4), load_pos(pos_x,pos_y,pos_z,a3)), pbc_inv, pbc_matrix);
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
        add_force(f_x,f_y,f_z, a1, f_a);
        add_force(f_x,f_y,f_z, a2, f_b);
        add_force(f_x,f_y,f_z, a3, f_c);
        add_force(f_x,f_y,f_z, a4, f_d);
        e += _result_energy;
    }}
'''

_MAIN_TEMPLATE = r'''
extern "C" __global__
void compute_bonded(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    float* __restrict__ f_x,
    float* __restrict__ f_y,
    float* __restrict__ f_z,
    float* __restrict__ energy_buf,
    const float* __restrict__ pbc_inv,
    const float* __restrict__ pbc_matrix,
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


def _generate_parameter_loads(term_name, parameter_names, parameters_per_term):
    lines = []
    for i, parameter_name in enumerate(parameter_names):
        lines.append(
            f'float {parameter_name} = {term_name}_prm[idx*{parameters_per_term} + {i}];'
        )
    return '\n        '.join(lines)


def _assemble_kernel(term_specs):
    kernel_param_lines = []
    body_phases = []
    for specification in term_specs:
        name = specification['name']
        expression = specification['expression']
        number_atoms = specification['number_atoms']
        d_idx = specification['d_indices']
        d_parameters = specification['d_parameters']
        count = specification['count']
        parameters_per_term = len(expression.parameter_names)

        kernel_param_lines.append(
            f'const int* __restrict__ {name}_idx, const float* __restrict__ {name}_prm, int num_{name}'
        )

        parameter_loads = _generate_parameter_loads(name, expression.parameter_names, parameters_per_term)

        if number_atoms == 2:
            geometric_variables = {'geo0': expression.geometric_names[0]}
        elif number_atoms == 3:
            geometric_variables = {'geo0': expression.geometric_names[0], 'geo1': expression.geometric_names[1]}
        else:
            geometric_variables = {'geo0': expression.geometric_names[0]}

        if number_atoms == 2:
            template = _TWO_BODY_TEMPLATE
        elif number_atoms == 3:
            template = _THREE_BODY_TEMPLATE
        else:
            template = _FOUR_BODY_TEMPLATE

        phase = template.format(
            name=name,
            param_loads=parameter_loads,
            expression_fragment=expression.cuda_fragment,
            **geometric_variables,
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
    _remap_kernel = None

    @classmethod
    def _get_remap_kernel(cls):
        if cls._remap_kernel is None:
            cls._remap_kernel = cp.RawKernel(
                _REMAP_INDICES_KERNEL, "remap_indices_kernel"
            )
        return cls._remap_kernel

    def __init__(self):
        self._term_specs = []
        self._kernel = None
        self._kernel_source = None
        self._term_data = []
        self._num_sm = None

    def add_expression(self, expression, term_name):
        number_atoms_map = {2: 2, 3: 3, 4: 4}
        number_atoms = number_atoms_map.get(expression.body, 4)
        self._term_specs.append({
            'expression': expression,
            'term_name': term_name,
            'number_atoms': number_atoms,
        })

    def bind(self, topology, parameter_table):
        self._term_data = []
        for specification in self._term_specs:
            term_name = specification['term_name']
            expression = specification['expression']
            number_atoms = specification['number_atoms']

            indices_field = f'{term_name}_indices'
            indices = getattr(topology, indices_field, None)
            if indices is None or indices.shape[0] == 0:
                continue

            parameters_matrix = parameter_table.get_term_parameter(term_name)

            d_indices = cp.asarray(
                np.ascontiguousarray(indices.astype(np.int32).ravel())
            )
            d_parameters = cp.asarray(
                np.ascontiguousarray(parameters_matrix.astype(np.float32).ravel())
            )

            self._term_data.append({
                'name': term_name,
                'expression': expression,
                'number_atoms': number_atoms,
                'd_indices': d_indices,
                'd_parameters': d_parameters,
                'count': indices.shape[0],
            })

        self._kernel_source = _assemble_kernel(self._term_data)

    @classmethod
    def charmm(cls, topology, parameter_table):
        from mdpy.force.expressions.bonded import (
            harmonic_bond, charmm_angle, periodic_dihedral, harmonic_improper,
        )
        bonded = cls()
        bonded.add_expression(harmonic_bond, 'bond')
        bonded.add_expression(charmm_angle, 'angle')
        bonded.add_expression(periodic_dihedral, 'dihedral')
        bonded.add_expression(harmonic_improper, 'improper')
        bonded.bind(topology, parameter_table)
        return bonded

    def remap_indices_gpu(self, d_remap):
        active_terms = [td for td in self._term_data if td['count'] > 0]
        if not active_terms:
            return
        kernel = self._get_remap_kernel()
        for td in active_terms:
            indices = td['d_indices']
            n = indices.size
            tpb = 256
            grid = ((n + tpb - 1) // tpb,)
            kernel(grid, (tpb,), (d_remap, indices, np.int32(n)))

    def _ensure_compiled(self):
        if self._kernel is not None:
            return
        self._kernel = cp.RawKernel(self._kernel_source, 'compute_bonded')
        if self._num_sm is None:
            self._num_sm = cp.cuda.runtime.getDeviceProperties(0)['multiProcessorCount']

    def compute(self, gpu_context, block_list=None, compute_energy=True):
        if not self._term_data:
            return
        self._ensure_compiled()

        total = sum(term_data['count'] for term_data in self._term_data)
        block_size = 128
        max_blocks = 6 * self._num_sm
        grid_size = max(min((total + block_size - 1) // block_size, max_blocks), 1)

        args = [
            gpu_context.d_positions_x,
            gpu_context.d_positions_y,
            gpu_context.d_positions_z,
            gpu_context.d_forces_x,
            gpu_context.d_forces_y,
            gpu_context.d_forces_z,
            gpu_context.d_energy,
            gpu_context.d_pbc_inv,
            gpu_context.d_pbc_matrix,
        ]
        for term_data in self._term_data:
            args.append(term_data['d_indices'])
            args.append(term_data['d_parameters'])
            args.append(np.int32(term_data['count']))

        self._kernel((grid_size,), (block_size,), tuple(args))


_V2_BODY_TEMPLATES = {
    2: r'''
    for (int idx = tid; idx < num_terms; idx += stride) {{
        int a1 = d_indices[idx*2];
        int a2 = d_indices[idx*2+1];
        {param_loads}
        {expression_fragment}
        e += _result_energy;
    }}
''',
    3: r'''
    for (int idx = tid; idx < num_terms; idx += stride) {{
        int a1 = d_indices[idx*3];
        int a2 = d_indices[idx*3+1];
        int a3 = d_indices[idx*3+2];
        {param_loads}
        {expression_fragment}
        e += _result_energy;
    }}
''',
    4: r'''
    for (int idx = tid; idx < num_terms; idx += stride) {{
        int a1 = d_indices[idx*4];
        int a2 = d_indices[idx*4+1];
        int a3 = d_indices[idx*4+2];
        int a4 = d_indices[idx*4+3];
        {param_loads}
        {expression_fragment}
        e += _result_energy;
    }}
''',
}

_V2_MAIN_TEMPLATE = r'''
extern "C" __global__
void compute_bonded_v2(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    float* __restrict__ f_x,
    float* __restrict__ f_y,
    float* __restrict__ f_z,
    float* __restrict__ energy_buf,
    const float* __restrict__ pbc_inv,
    const float* __restrict__ pbc_matrix,
    const int* __restrict__ d_indices,
    const float* __restrict__ d_parameters,
    int num_terms{extra_params}
) {{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float e = 0.0f;

    {body}

    for (int off = 16; off > 0; off >>= 1)
        e += __shfl_down_sync(0xffffffff, e, off);
    if ((threadIdx.x & 31) == 0)
        atomicAdd(energy_buf, e);
}}
'''


class BondedForceV2:
    name = 'bonded_v2'

    def __init__(self, expression):
        self._expression = expression
        self._body = expression.body
        self._parameter_names = expression.parameter_names
        self._parameters_per_term = len(self._parameter_names)
        self._per_particle = expression.per_particle
        self._per_particle_gpu = {}
        self._per_particle_properties = list(dict.fromkeys(
            expression.per_particle.values()
        ))
        self._pending_indices = []
        self._pending_parameters = []
        self._count = 0
        self._capacity = 0
        self._d_indices = None
        self._d_parameters = None
        self._kernel = None
        self._kernel_source = None
        self._num_sm = None
        self._dirty = True

    def set_parameter(self, name, array):
        arr = np.asarray(array, dtype=env.NUMPY_FLOAT).ravel()
        self._per_particle_gpu[name] = cp.asarray(arr)

    def add(self, indices, **params):
        self._pending_indices.append(list(indices))
        self._pending_parameters.append([params.get(name, 0.0) for name in self._parameter_names])
        self._count += 1
        self._dirty = True

    def sync(self):
        if not self._pending_indices:
            return
        indices = np.array(self._pending_indices, dtype=np.int32)
        parameters = np.array(self._pending_parameters, dtype=np.float32)
        new_count = indices.shape[0]
        if self._capacity < new_count:
            new_capacity = max(new_count, max(64, int(self._capacity * 1.5)))
            self._d_indices = cp.zeros((new_capacity, self._body), dtype=np.int32)
            self._d_parameters = cp.zeros((new_capacity, self._parameters_per_term), dtype=np.float32)
            self._capacity = new_capacity
        self._d_indices[:new_count] = cp.asarray(indices)
        self._d_parameters[:new_count] = cp.asarray(parameters)
        self._pending_indices.clear()
        self._pending_parameters.clear()
        self._dirty = False

    def bind(self):
        self.sync()
        self._assemble_kernel()

    def _assemble_kernel(self):
        param_loads_lines = []
        for i, parameter_name in enumerate(self._parameter_names):
            param_loads_lines.append(
                f'float {parameter_name} = d_parameters[idx*{self._parameters_per_term} + {i}];'
            )
        atom_index_names = ['a1', 'a2', 'a3', 'a4']
        for arg_name in self._per_particle:
            base_name = self._per_particle[arg_name]
            trailing = _extract_trailing_digit(arg_name)
            atom_idx = atom_index_names[trailing - 1] if trailing is not None else 'a1'
            param_loads_lines.append(
                f'float {arg_name} = d_{base_name}[{atom_idx}];'
            )
        param_loads = '\n        '.join(param_loads_lines)
        body_template = _V2_BODY_TEMPLATES[self._body]
        body = body_template.format(
            param_loads=param_loads,
            expression_fragment=self._expression.cuda_fragment,
        )
        extra_param_lines = []
        for prop_name in self._per_particle_properties:
            extra_param_lines.append(
                f'const float* __restrict__ d_{prop_name}'
            )
        extra_params = ''
        if extra_param_lines:
            extra_params = ',\n    ' + ',\n    '.join(extra_param_lines)
        self._kernel_source = _PREAMBLE + _V2_MAIN_TEMPLATE.format(
            body=body, extra_params=extra_params,
        )

    def bind_sorted(self, gpu_context, sort_order):
        for prop_name in self._per_particle_properties:
            d_arr = self._per_particle_gpu[prop_name]
            arrays = {prop_name: d_arr}
            gpu_context.permute_to_sorted(sort_order, arrays)
            self._per_particle_gpu[prop_name] = arrays[prop_name]

    def _ensure_compiled(self):
        if self._kernel is not None:
            return
        self._kernel = cp.RawKernel(self._kernel_source, 'compute_bonded_v2')
        if self._num_sm is None:
            self._num_sm = cp.cuda.runtime.getDeviceProperties(0)['multiProcessorCount']

    def compute(self, gpu_context, block_list=None):
        if self._count == 0:
            return
        if self._dirty:
            self.sync()
        self._ensure_compiled()

        block_size = 128
        max_blocks = 6 * self._num_sm
        grid_size = max(min((self._count + block_size - 1) // block_size, max_blocks), 1)

        args = [
            gpu_context.d_positions_x,
            gpu_context.d_positions_y,
            gpu_context.d_positions_z,
            gpu_context.d_forces_x,
            gpu_context.d_forces_y,
            gpu_context.d_forces_z,
            gpu_context.d_energy,
            gpu_context.d_pbc_inv,
            gpu_context.d_pbc_matrix,
            self._d_indices.ravel(),
            self._d_parameters.ravel(),
            np.int32(self._count),
        ]
        for prop_name in self._per_particle_properties:
            args.append(self._per_particle_gpu[prop_name])
        self._kernel((grid_size,), (block_size,), tuple(args))
