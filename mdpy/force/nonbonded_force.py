from __future__ import annotations

import ast
import inspect
import re
import textwrap

import cupy as cp
import numpy as np

from mdpy.core.tile_list import _GATHER_SORTED_KERNEL_4COMP
from mdpy.force.force_term import ForceTerm


class Parameter:
    def __getitem__(self, index):
        return self


_MATH_FUNCTIONS = {
    'sqrt': 'sqrtf',
    'exp': 'expf',
    'log': 'logf',
    'abs': 'fabsf',
    'sin': 'sinf',
    'cos': 'cosf',
    'tan': 'tanf',
}

_PACKED_PARAMS = {
    'sigma_half': ('sigma_epsilon', 'x'),
    'sqrt_epsilon': ('sigma_epsilon', 'y'),
}


def _unique_gpu_arrays(parameter_names):
    seen = set()
    result = []
    for name in parameter_names:
        arr = _PACKED_PARAMS[name][0] if name in _PACKED_PARAMS else name
        if arr not in seen:
            seen.add(arr)
            result.append(arr)
    return result


_PACK_POSQ_KERNEL = r'''
extern "C" __global__
void pack_posq_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    const float* __restrict__ charge,
    float* __restrict__ posq,
    int num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    posq[idx * 4 + 0] = pos_x[idx];
    posq[idx * 4 + 1] = pos_y[idx];
    posq[idx * 4 + 2] = pos_z[idx];
    posq[idx * 4 + 3] = charge[idx];
}
'''


class _Transpiler(ast.NodeVisitor):
    def __init__(self, index_names, parameter_names):
        self.index_names = index_names
        self.parameter_names = set(parameter_names)
        self.lines = []
        self.local_variables = set()

    def _emit(self, line):
        self.lines.append(line)

    def transpile(self, func_body):
        for statement in func_body:
            self.visit(statement)
        return '\n'.join(self.lines)

    def visit_Assign(self, node):
        if len(node.targets) != 1:
            raise NotImplementedError('Multiple assignment targets not supported')
        target_name = node.targets[0].id
        self.local_variables.add(target_name)
        value = self._translate_expr(node.value)
        self._emit(f'float {target_name} = {value};')

    def visit_Return(self, node):
        if not isinstance(node.value, ast.Tuple) or len(node.value.elts) != 2:
            raise NotImplementedError('Return must be a 2-tuple (energy, force_magnitude)')
        energy_expr = self._translate_expr(node.value.elts[0])
        force_expr = self._translate_expr(node.value.elts[1])
        self._emit(f'float _result_energy = {energy_expr};')
        self._emit(f'float _result_force = {force_expr};')

    def _translate_expr(self, node):
        if isinstance(node, ast.Constant):
            return self._translate_constant(node)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.BinOp):
            return self._translate_binop(node)
        elif isinstance(node, ast.UnaryOp):
            return self._translate_unaryop(node)
        elif isinstance(node, ast.Subscript):
            return self._translate_subscript(node)
        elif isinstance(node, ast.Call):
            return self._translate_call(node)
        else:
            raise NotImplementedError(f'Unsupported AST node type: {type(node).__name__}')

    def _translate_constant(self, node):
        if isinstance(node.value, float):
            s = repr(node.value)
            if '.' not in s and 'e' not in s and 'E' not in s:
                s += '.0'
            return s + 'f'
        elif isinstance(node.value, int):
            return repr(float(node.value)) + 'f'
        return repr(node.value)

    def _translate_binop(self, node):
        if isinstance(node.op, ast.Pow):
            return self._translate_pow(node)
        left = self._translate_expr(node.left)
        right = self._translate_expr(node.right)
        op_map = {
            ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/',
        }
        op_type = type(node.op)
        if op_type in op_map:
            return f'({left} {op_map[op_type]} {right})'
        raise NotImplementedError(f'Unsupported binary op: {op_type.__name__}')

    def _translate_unaryop(self, node):
        operand = self._translate_expr(node.operand)
        if isinstance(node.op, ast.USub):
            return f'(-{operand})'
        elif isinstance(node.op, ast.UAdd):
            return f'(+{operand})'
        raise NotImplementedError(f'Unsupported unary op: {type(node.op).__name__}')

    def _translate_subscript(self, node):
        if not isinstance(node.value, ast.Name):
            raise NotImplementedError('Only simple name subscripts supported')
        param_name = node.value.id
        if param_name not in self.parameter_names:
            raise ValueError(f'{param_name} is not a declared Parameter')
        if not isinstance(node.slice, ast.Name):
            raise NotImplementedError('Parameter index must be a variable name')
        index_name = node.slice.id
        if index_name == self.index_names[0]:
            return f'{param_name}_i'
        elif index_name == self.index_names[1]:
            return f'{param_name}_j'
        raise ValueError(f'Index variable {index_name} is not a recognized particle index')

    def _translate_call(self, node):
        if not isinstance(node.func, ast.Name):
            raise NotImplementedError('Only simple function calls supported')
        func_name = node.func.id
        if func_name in _MATH_FUNCTIONS:
            if len(node.args) != 1:
                raise NotImplementedError(f'{func_name} expects exactly 1 argument')
            arg = self._translate_expr(node.args[0])
            return f'{_MATH_FUNCTIONS[func_name]}({arg})'
        raise NotImplementedError(f'Unsupported function: {func_name}')

    def _translate_pow(self, node):
        base = self._translate_expr(node.left)
        if not isinstance(node.right, ast.Constant) or not isinstance(node.right.value, int):
            raise NotImplementedError('Only integer constant powers are supported')
        exponent = node.right.value
        if exponent < 0:
            raise NotImplementedError('Negative powers not supported')
        if exponent == 0:
            return '1.0f'
        if exponent == 1:
            return base
        if exponent == 2:
            return f'({base} * {base})'
        if exponent == 3:
            return f'({base} * {base} * {base})'
        if exponent == 6:
            temp = f'_pow6_{abs(hash(node)) % 10000}'
            self.local_variables.add(temp)
            self._emit(f'float {temp} = ({base} * {base} * {base});')
            return f'({temp} * {temp})'
        if exponent == 12:
            temp = f'_pow12_{abs(hash(node)) % 10000}'
            half = self._translate_pow_node_6(base, node)
            self._emit(f'float {temp} = {half};')
            return f'({temp} * {temp})'
        return self._inline_power_chain(base, exponent)

    def _translate_pow_node_6(self, base, node):
        temp = f'_pow6_{abs(hash(node)) % 10000}'
        self.local_variables.add(temp)
        self._emit(f'float {temp} = ({base} * {base} * {base});')
        return f'({temp} * {temp})'

    def _inline_power_chain(self, base, exponent):
        parts = [base] * exponent
        return '(' + ' * '.join(parts) + ')'


class NonbondedExpression:
    def __init__(self, func, source, ast_tree, index_names, parameter_names,
                 distance_name, cuda_fragment, local_variables):
        self.func = func
        self.source = source
        self.ast_tree = ast_tree
        self.index_names = index_names
        self.parameter_names = parameter_names
        self.distance_name = distance_name
        self.cuda_fragment = cuda_fragment
        self.local_variables = local_variables

    def __add__(self, other):
        if not isinstance(other, NonbondedExpression):
            return NotImplemented
        merged_params = list(dict.fromkeys(self.parameter_names + other.parameter_names))
        suffix = '_2'
        other_locals = set()
        for var in other.local_variables:
            if var in self.local_variables or var in set(self.parameter_names):
                other_locals.add(var + suffix)
            else:
                other_locals.add(var)

        renamed_fragment_1 = _rename_output_vars(
            _rename_locals_in_cuda(
                self.cuda_fragment, {'force_magnitude'},
                {'force_magnitude'},
                '_1'
            ),
            '_1'
        )
        renamed_fragment_2 = _rename_output_vars(
            _rename_locals_in_cuda(
                other.cuda_fragment, other.local_variables,
                self.local_variables | set(self.parameter_names) | {'force_magnitude', 'energy_val'},
                suffix
            ),
            suffix
        )
        combined_fragment = (
            renamed_fragment_1 + '\n'
            + renamed_fragment_2 + '\n'
            + 'float energy_val = _result_energy_1 + _result_energy' + suffix + ';\n'
            + 'float force_magnitude = _result_force_1 + _result_force' + suffix + ';'
        )
        combined_locals = self.local_variables | other_locals
        return NonbondedExpression(
            func=None,
            source=self.source + '\n--- combined ---\n' + other.source,
            ast_tree=None,
            index_names=self.index_names,
            parameter_names=merged_params,
            distance_name=self.distance_name,
            cuda_fragment=combined_fragment,
            local_variables=combined_locals,
        )

    def assemble_self_tile_kernel(self):
        fragment = self.cuda_fragment
        if '_result_energy_1' not in fragment:
            fragment += '\nfloat energy_val = _result_energy;'
            fragment += '\nfloat force_magnitude = _result_force;'
        return _assemble_self_tile_kernel_v2(self.parameter_names, fragment)

    def assemble_cross_tile_kernel(self):
        fragment = self.cuda_fragment
        if '_result_energy_1' not in fragment:
            fragment += '\nfloat energy_val = _result_energy;'
            fragment += '\nfloat force_magnitude = _result_force;'
        return _assemble_cross_tile_kernel_v2(self.parameter_names, fragment)

    def assemble_tile_kernel(self):
        fragment = self.cuda_fragment
        if '_result_energy_1' not in fragment:
            fragment += '\nfloat energy_val = _result_energy;'
            fragment += '\nfloat force_magnitude = _result_force;'
        return _assemble_tile_kernel_v2(self.parameter_names, fragment)

    def assemble_main_tile_kernel(self):
        fragment = self.cuda_fragment
        if '_result_energy_1' not in fragment:
            fragment += '\nfloat energy_val = _result_energy;'
            fragment += '\nfloat force_magnitude = _result_force;'
        return _assemble_main_tile_kernel_v3(self.parameter_names, fragment)


def _rename_output_vars(cuda_fragment, tag):
    lines = cuda_fragment.split('\n')
    result = []
    for line in lines:
        new_line = line
        stripped = line.strip()
        if stripped.startswith('float _result_energy ='):
            new_line = line.replace('float _result_energy =', f'float _result_energy{tag} =', 1)
        if stripped.startswith('float _result_force ='):
            new_line = line.replace('float _result_force =', f'float _result_force{tag} =', 1)
        result.append(new_line)
    return '\n'.join(result)


def _rename_locals_in_cuda(cuda_fragment, local_variables, conflict_set, suffix):
    lines = cuda_fragment.split('\n')
    result = []
    for line in lines:
        new_line = line
        for var in sorted(local_variables, key=len, reverse=True):
            if var in conflict_set:
                new_line = re.sub(
                    r'\b' + re.escape(var) + r'\b',
                    var + suffix,
                    new_line
                )
        result.append(new_line)
    return '\n'.join(result)


def _is_parameter_call(node):
    return (isinstance(node, ast.Call) and
            isinstance(node.func, ast.Name) and
            node.func.id == 'Parameter')


def _generate_param_decls(parameter_names):
    param_decls = ''
    for param_name in parameter_names:
        param_decls += f',\n    const float* __restrict__ {param_name}'
        param_decls += f',\n    const float* __restrict__ {param_name}_14'
    return param_decls


def _generate_sorted_param_decls(parameter_names):
    decls = ''
    for name in parameter_names:
        decls += f',\n    const float* __restrict__ sorted_{name}'
        decls += f',\n    const float* __restrict__ sorted_{name}_14'
    return decls


def _generate_pre_fetch(parameter_names):
    pre_fetch_lines = []
    for param_name in parameter_names:
        pre_fetch_lines.append(
            f'float {param_name}_i = is_14 ? {param_name}_14[gi] : {param_name}[gi];'
        )
        pre_fetch_lines.append(
            f'float {param_name}_j = is_14 ? {param_name}_14[gj] : {param_name}[gj];'
        )
    return '\n        '.join(pre_fetch_lines)


def _generate_param_load_i(parameter_names):
    lines = []
    for name in parameter_names:
        lines.append(f'float {name}_i = 0.0f, {name}_i_14 = 0.0f;')
        lines.append(f'if (gi >= 0) {{ {name}_i = {name}[gi]; {name}_i_14 = {name}_14[gi]; }}')
    return '\n        '.join(lines)


def _generate_sorted_param_load_i(parameter_names):
    lines = []
    for name in parameter_names:
        lines.append(f'float {name}_i = sorted_{name}[block_x * 32 + tgx];')
        lines.append(f'float {name}_i_14 = sorted_{name}_14[block_x * 32 + tgx];')
    return '\n        '.join(lines)


def _generate_param_load_j_init(parameter_names):
    lines = []
    for name in parameter_names:
        lines.append(f'float {name}_j = 0.0f, {name}_j_14 = 0.0f;')
        lines.append(f'if (gj_init >= 0) {{ {name}_j = {name}[gj_init]; {name}_j_14 = {name}_14[gj_init]; }}')
    return '\n        '.join(lines)


def _generate_param_load_j_tile(parameter_names):
    lines = []
    for name in parameter_names:
        lines.append(f'float {name}_j = 0.0f, {name}_j_14 = 0.0f;')
        lines.append(f'if (gj >= 0 && gj < num_particles) {{ {name}_j = {name}[gj]; {name}_j_14 = {name}_14[gj]; }}')
    return '\n        '.join(lines)


def _generate_shuffle_warp_data(parameter_names):
    lines = [
        'shfl_px = __shfl_sync(0xffffffff, shfl_px, (tgx + 1) & 31);',
        'shfl_py = __shfl_sync(0xffffffff, shfl_py, (tgx + 1) & 31);',
        'shfl_pz = __shfl_sync(0xffffffff, shfl_pz, (tgx + 1) & 31);',
        'shfl_fx = __shfl_sync(0xffffffff, shfl_fx, (tgx + 1) & 31);',
        'shfl_fy = __shfl_sync(0xffffffff, shfl_fy, (tgx + 1) & 31);',
        'shfl_fz = __shfl_sync(0xffffffff, shfl_fz, (tgx + 1) & 31);',
    ]
    for name in parameter_names:
        lines.append(
            f'{name}_j = __shfl_sync(0xffffffff, {name}_j, (tgx + 1) & 31);'
        )
        lines.append(
            f'{name}_j_14 = __shfl_sync(0xffffffff, {name}_j_14, (tgx + 1) & 31);'
        )
    return '\n        '.join(lines)


def _generate_param_select(parameter_names):
    lines = []
    for name in parameter_names:
        lines.append(f'float {name}_i_saved = {name}_i;')
        lines.append(f'if (is_14) {name}_i = {name}_i_14;')
        lines.append(f'float {name}_j_saved = {name}_j;')
        lines.append(f'if (is_14) {name}_j = {name}_j_14;')
    return '\n            '.join(lines)


def _generate_param_restore(parameter_names):
    lines = []
    for name in parameter_names:
        lines.append(f'if (is_14) {name}_i = {name}_i_saved;')
        lines.append(f'if (is_14) {name}_j = {name}_j_saved;')
    return '\n            '.join(lines)


def _generate_param_use(parameter_names):
    lines = []
    for name in parameter_names:
        lines.append(f'float {name}_i_use = is_14 ? {name}_i_14 : {name}_i;')
        lines.append(f'float {name}_j_use = is_14 ? {name}_j_14 : {name}_j;')
    return '\n                '.join(lines)


def _generate_param_decls_v2(parameter_names):
    decls = ''
    for arr in _unique_gpu_arrays(parameter_names):
        if arr in {v[0] for v in _PACKED_PARAMS.values()}:
            decls += f',\n    const float2* __restrict__ {arr}'
            decls += f',\n    const float2* __restrict__ {arr}_14'
        else:
            decls += f',\n    const float* __restrict__ {arr}'
            decls += f',\n    const float* __restrict__ {arr}_14'
    return decls


def _generate_sorted_param_decls_v2(parameter_names):
    decls = ''
    for arr in _unique_gpu_arrays(parameter_names):
        if arr in {v[0] for v in _PACKED_PARAMS.values()}:
            decls += f',\n    const float2* __restrict__ sorted_{arr}'
            decls += f',\n    const float2* __restrict__ sorted_{arr}_14'
        else:
            decls += f',\n    const float* __restrict__ sorted_{arr}'
            decls += f',\n    const float* __restrict__ sorted_{arr}_14'
    return decls


def _generate_sorted_param_load_i_v2(parameter_names):
    lines = []
    loaded = set()
    for name in parameter_names:
        if name in _PACKED_PARAMS:
            arr, comp = _PACKED_PARAMS[name]
            if arr not in loaded:
                lines.append(f'float2 {arr}_i_v = sorted_{arr}[block_x * 32 + tgx];')
                lines.append(f'float2 {arr}_i_14_v = sorted_{arr}_14[block_x * 32 + tgx];')
                loaded.add(arr)
            lines.append(f'float {name}_i = {arr}_i_v.{comp};')
            lines.append(f'float {name}_i_14 = {arr}_i_14_v.{comp};')
        else:
            lines.append(f'float {name}_i = sorted_{name}[block_x * 32 + tgx];')
            lines.append(f'float {name}_i_14 = sorted_{name}_14[block_x * 32 + tgx];')
    return '\n        '.join(lines)


def _generate_param_load_j_tile_v2(parameter_names):
    lines = []
    loaded_j = set()
    for name in parameter_names:
        if name in _PACKED_PARAMS:
            arr, comp = _PACKED_PARAMS[name]
            lines.append(f'float {name}_j = 0.0f;')
            lines.append(f'float {name}_j_14 = 0.0f;')
        else:
            lines.append(f'float {name}_j = 0.0f, {name}_j_14 = 0.0f;')
            lines.append(f'if (gj >= 0 && gj < num_particles) {{ {name}_j = {name}[gj]; {name}_j_14 = {name}_14[gj]; }}')
    for name in parameter_names:
        if name in _PACKED_PARAMS:
            arr, comp = _PACKED_PARAMS[name]
            if arr not in loaded_j:
                lines.append(f'if (gj >= 0 && gj < num_particles) {{ float2 _{arr}_j_v = {arr}[gj]; float2 _{arr}_j_14_v = {arr}_14[gj];')
                loaded_j.add(arr)
    if loaded_j:
        for name in parameter_names:
            if name in _PACKED_PARAMS:
                arr, comp = _PACKED_PARAMS[name]
                lines.append(f'{name}_j = _{arr}_j_v.{comp};')
                lines.append(f'{name}_j_14 = _{arr}_j_14_v.{comp};')
        lines.append('}')
    return '\n        '.join(lines)


def _generate_shuffle_warp_data_v2(parameter_names):
    lines = [
        'shfl_px = __shfl_sync(0xffffffff, shfl_px, (tgx + 1) & 31);',
        'shfl_py = __shfl_sync(0xffffffff, shfl_py, (tgx + 1) & 31);',
        'shfl_pz = __shfl_sync(0xffffffff, shfl_pz, (tgx + 1) & 31);',
        'shfl_fx = __shfl_sync(0xffffffff, shfl_fx, (tgx + 1) & 31);',
        'shfl_fy = __shfl_sync(0xffffffff, shfl_fy, (tgx + 1) & 31);',
        'shfl_fz = __shfl_sync(0xffffffff, shfl_fz, (tgx + 1) & 31);',
    ]
    for name in parameter_names:
        lines.append(f'{name}_j = __shfl_sync(0xffffffff, {name}_j, (tgx + 1) & 31);')
        lines.append(f'{name}_j_14 = __shfl_sync(0xffffffff, {name}_j_14, (tgx + 1) & 31);')
    return '\n        '.join(lines)


def _generate_param_select_v2(parameter_names):
    lines = []
    for name in parameter_names:
        lines.append(f'float {name}_i_saved = {name}_i; if (is_14) {name}_i = {name}_i_14;')
        lines.append(f'float {name}_j_saved = {name}_j; if (is_14) {name}_j = {name}_j_14;')
    return '\n            '.join(lines)


def _generate_param_restore_v2(parameter_names):
    lines = []
    for name in parameter_names:
        lines.append(f'if (is_14) {name}_i = {name}_i_saved;')
        lines.append(f'if (is_14) {name}_j = {name}_j_saved;')
    return '\n            '.join(lines)


def _generate_param_decls_main(parameter_names):
    decls = ''
    for arr in _unique_gpu_arrays(parameter_names):
        if arr == 'charge':
            pass
        else:
            if arr in {v[0] for v in _PACKED_PARAMS.values()}:
                decls += f',\n    const float2* __restrict__ {arr}'
            else:
                decls += f',\n    const float* __restrict__ {arr}'
    return decls


def _generate_sorted_param_decls_main(parameter_names):
    decls = ''
    for arr in _unique_gpu_arrays(parameter_names):
        if arr == 'charge':
            pass
        else:
            if arr in {v[0] for v in _PACKED_PARAMS.values()}:
                decls += f',\n    const float2* __restrict__ sorted_{arr}'
            else:
                decls += f',\n    const float* __restrict__ sorted_{arr}'
    return decls


def _generate_sorted_param_load_i_main(parameter_names):
    lines = []
    loaded = set()
    for name in parameter_names:
        if name == 'charge':
            lines.append('float charge_i = posq_i.w;')
        elif name in _PACKED_PARAMS:
            arr, comp = _PACKED_PARAMS[name]
            if arr not in loaded:
                lines.append(f'float2 {arr}_i_v = sorted_{arr}[block_x * 32 + tgx];')
                loaded.add(arr)
            lines.append(f'float {name}_i = {arr}_i_v.{comp};')
        else:
            lines.append(f'float {name}_i = sorted_{name}[block_x * 32 + tgx];')
    return '\n        '.join(lines)


def _generate_param_load_j_tile_main(parameter_names):
    lines = []
    loaded_j = set()
    for name in parameter_names:
        if name == 'charge':
            lines.append('float charge_j = _charge_j_posq;')
        elif name in _PACKED_PARAMS:
            lines.append(f'float {name}_j = 0.0f;')
        else:
            lines.append(f'float {name}_j = 0.0f;')
            lines.append(f'if (gj >= 0 && gj < num_particles) {{ {name}_j = {name}[gj]; }}')
    for name in parameter_names:
        if name in _PACKED_PARAMS:
            arr, comp = _PACKED_PARAMS[name]
            if arr not in loaded_j:
                lines.append(f'if (gj >= 0 && gj < num_particles) {{ float2 _{arr}_j_v = {arr}[gj];')
                loaded_j.add(arr)
    if loaded_j:
        for name in parameter_names:
            if name in _PACKED_PARAMS:
                arr, comp = _PACKED_PARAMS[name]
                lines.append(f'{name}_j = _{arr}_j_v.{comp};')
        lines.append('}')
    return '\n        '.join(lines)


def _generate_shuffle_warp_data_main(parameter_names):
    lines = [
        'shfl_px = __shfl_sync(0xffffffff, shfl_px, (tgx + 1) & 31);',
        'shfl_py = __shfl_sync(0xffffffff, shfl_py, (tgx + 1) & 31);',
        'shfl_pz = __shfl_sync(0xffffffff, shfl_pz, (tgx + 1) & 31);',
        'shfl_fx = __shfl_sync(0xffffffff, shfl_fx, (tgx + 1) & 31);',
        'shfl_fy = __shfl_sync(0xffffffff, shfl_fy, (tgx + 1) & 31);',
        'shfl_fz = __shfl_sync(0xffffffff, shfl_fz, (tgx + 1) & 31);',
    ]
    for name in parameter_names:
        if name == 'charge':
            lines.append('charge_j = __shfl_sync(0xffffffff, charge_j, (tgx + 1) & 31);')
        else:
            lines.append(f'{name}_j = __shfl_sync(0xffffffff, {name}_j, (tgx + 1) & 31);')
    return '\n        '.join(lines)


def _generate_param_decls_v2_main(parameter_names):
    decls = ''
    for arr in _unique_gpu_arrays(parameter_names):
        if arr in {v[0] for v in _PACKED_PARAMS.values()}:
            decls += f',\n    const float2* __restrict__ {arr}'
        else:
            decls += f',\n    const float* __restrict__ {arr}'
    return decls


def _generate_sorted_param_decls_v2_main(parameter_names):
    decls = ''
    for arr in _unique_gpu_arrays(parameter_names):
        if arr in {v[0] for v in _PACKED_PARAMS.values()}:
            decls += f',\n    const float2* __restrict__ sorted_{arr}'
        else:
            decls += f',\n    const float* __restrict__ sorted_{arr}'
    return decls


def _generate_sorted_param_load_i_v2_main(parameter_names):
    lines = []
    loaded = set()
    for name in parameter_names:
        if name in _PACKED_PARAMS:
            arr, comp = _PACKED_PARAMS[name]
            if arr not in loaded:
                lines.append(f'float2 {arr}_i_v = sorted_{arr}[block_x * 32 + tgx];')
                loaded.add(arr)
            lines.append(f'float {name}_i = {arr}_i_v.{comp};')
        else:
            lines.append(f'float {name}_i = sorted_{name}[block_x * 32 + tgx];')
    return '\n        '.join(lines)


def _generate_param_load_j_tile_v2_main(parameter_names):
    lines = []
    loaded_j = set()
    for name in parameter_names:
        if name in _PACKED_PARAMS:
            lines.append(f'float {name}_j = 0.0f;')
        else:
            lines.append(f'float {name}_j = 0.0f;')
            lines.append(f'if (gj >= 0 && gj < num_particles) {{ {name}_j = {name}[gj]; }}')
    for name in parameter_names:
        if name in _PACKED_PARAMS:
            arr, comp = _PACKED_PARAMS[name]
            if arr not in loaded_j:
                lines.append(f'if (gj >= 0 && gj < num_particles) {{ float2 _{arr}_j_v = {arr}[gj];')
                loaded_j.add(arr)
    if loaded_j:
        for name in parameter_names:
            if name in _PACKED_PARAMS:
                arr, comp = _PACKED_PARAMS[name]
                lines.append(f'{name}_j = _{arr}_j_v.{comp};')
        lines.append('}')
    return '\n        '.join(lines)


def _generate_param_decls_posq_v2(parameter_names):
    decls = ''
    for arr in _unique_gpu_arrays(parameter_names):
        if arr == 'charge':
            decls += ',\n    const float* __restrict__ charge_14'
        else:
            if arr in {v[0] for v in _PACKED_PARAMS.values()}:
                decls += f',\n    const float2* __restrict__ {arr}'
                decls += f',\n    const float2* __restrict__ {arr}_14'
            else:
                decls += f',\n    const float* __restrict__ {arr}'
                decls += f',\n    const float* __restrict__ {arr}_14'
    return decls


def _generate_sorted_param_decls_posq_v2(parameter_names):
    decls = ''
    for arr in _unique_gpu_arrays(parameter_names):
        if arr == 'charge':
            decls += ',\n    const float* __restrict__ sorted_charge_14'
        else:
            if arr in {v[0] for v in _PACKED_PARAMS.values()}:
                decls += f',\n    const float2* __restrict__ sorted_{arr}'
                decls += f',\n    const float2* __restrict__ sorted_{arr}_14'
            else:
                decls += f',\n    const float* __restrict__ sorted_{arr}'
                decls += f',\n    const float* __restrict__ sorted_{arr}_14'
    return decls


def _generate_sorted_param_load_i_posq_v2(parameter_names):
    lines = []
    loaded = set()
    for name in parameter_names:
        if name == 'charge':
            lines.append('float charge_i = posq_i.w;')
            lines.append('float charge_i_14 = sorted_charge_14[block_x * 32 + tgx];')
        elif name in _PACKED_PARAMS:
            arr, comp = _PACKED_PARAMS[name]
            if arr not in loaded:
                lines.append(f'float2 {arr}_i_v = sorted_{arr}[block_x * 32 + tgx];')
                lines.append(f'float2 {arr}_i_14_v = sorted_{arr}_14[block_x * 32 + tgx];')
                loaded.add(arr)
            lines.append(f'float {name}_i = {arr}_i_v.{comp};')
            lines.append(f'float {name}_i_14 = {arr}_i_14_v.{comp};')
        else:
            lines.append(f'float {name}_i = sorted_{name}[block_x * 32 + tgx];')
            lines.append(f'float {name}_i_14 = sorted_{name}_14[block_x * 32 + tgx];')
    return '\n        '.join(lines)


def _generate_param_load_j_tile_posq_v2(parameter_names):
    lines = []
    loaded_j = set()
    for name in parameter_names:
        if name == 'charge':
            lines.append('float charge_j = _charge_j_posq;')
            lines.append('float charge_j_14 = 0.0f;')
        elif name in _PACKED_PARAMS:
            lines.append(f'float {name}_j = 0.0f;')
            lines.append(f'float {name}_j_14 = 0.0f;')
        else:
            lines.append(f'float {name}_j = 0.0f, {name}_j_14 = 0.0f;')
            lines.append(f'if (gj >= 0 && gj < num_particles) {{ {name}_j = {name}[gj]; {name}_j_14 = {name}_14[gj]; }}')
    for name in parameter_names:
        if name in _PACKED_PARAMS:
            arr, comp = _PACKED_PARAMS[name]
            if arr not in loaded_j:
                lines.append(f'if (gj >= 0 && gj < num_particles) {{ float2 _{arr}_j_v = {arr}[gj]; float2 _{arr}_j_14_v = {arr}_14[gj];')
                loaded_j.add(arr)
    if loaded_j:
        for name in parameter_names:
            if name in _PACKED_PARAMS:
                arr, comp = _PACKED_PARAMS[name]
                lines.append(f'{name}_j = _{arr}_j_v.{comp};')
                lines.append(f'{name}_j_14 = _{arr}_j_14_v.{comp};')
        lines.append('}')
    if 'charge' in parameter_names:
        lines.append('if (gj >= 0 && gj < num_particles) { charge_j_14 = charge_14[gj]; }')
    return '\n        '.join(lines)


def _assemble_tile_kernel_v2(parameter_names, expression_fragment):
    use_posq = 'charge' in parameter_names

    if use_posq:
        param_decls = _generate_param_decls_posq_v2(parameter_names)
        sorted_param_decls = _generate_sorted_param_decls_posq_v2(parameter_names)
        sorted_param_load_i = _generate_sorted_param_load_i_posq_v2(parameter_names)
        param_load_j = _generate_param_load_j_tile_posq_v2(parameter_names)
        pos_args_decl = (
            '    const float4* __restrict__ sorted_posq,\n'
            '    const float4* __restrict__ posq,'
        )
        i_pos_load = (
            '        float4 posq_i = sorted_posq[block_x * 32 + tgx];\n'
            '        float px_i = posq_i.x;\n'
            '        float py_i = posq_i.y;\n'
            '        float pz_i = posq_i.z;'
        )
        j_pos_load = (
            '        float shfl_px = 0.0f, shfl_py = 0.0f, shfl_pz = 0.0f, _charge_j_posq = 0.0f;\n'
            '        if (gj >= 0 && gj < num_particles) {\n'
            '            float4 _pj = posq[gj];\n'
            '            shfl_px = _pj.x;\n'
            '            shfl_py = _pj.y;\n'
            '            shfl_pz = _pj.z;\n'
            '            _charge_j_posq = _pj.w;\n'
            '        }'
        )
    else:
        param_decls = _generate_param_decls_v2(parameter_names)
        sorted_param_decls = _generate_sorted_param_decls_v2(parameter_names)
        sorted_param_load_i = _generate_sorted_param_load_i_v2(parameter_names)
        param_load_j = _generate_param_load_j_tile_v2(parameter_names)
        pos_args_decl = (
            '    const float* __restrict__ sorted_pos_x,\n'
            '    const float* __restrict__ sorted_pos_y,\n'
            '    const float* __restrict__ sorted_pos_z,\n'
            '    const float* __restrict__ pos_x,\n'
            '    const float* __restrict__ pos_y,\n'
            '    const float* __restrict__ pos_z,'
        )
        i_pos_load = (
            '        float px_i = sorted_pos_x[block_x * 32 + tgx];\n'
            '        float py_i = sorted_pos_y[block_x * 32 + tgx];\n'
            '        float pz_i = sorted_pos_z[block_x * 32 + tgx];'
        )
        j_pos_load = (
            '        float shfl_px = 0.0f, shfl_py = 0.0f, shfl_pz = 0.0f;\n'
            '        if (gj >= 0 && gj < num_particles) {\n'
            '            shfl_px = pos_x[gj];\n'
            '            shfl_py = pos_y[gj];\n'
            '            shfl_pz = pos_z[gj];\n'
            '        }'
        )

    shuffle_code = _generate_shuffle_warp_data_v2(parameter_names)
    param_select = _generate_param_select_v2(parameter_names)
    param_restore = _generate_param_restore_v2(parameter_names)

    kernel = f'''extern "C" __global__
void tile_kernel(
{pos_args_decl}
    float* __restrict__ f_x,
    float* __restrict__ f_y,
    float* __restrict__ f_z,
    float* __restrict__ energy_buffer,
    const int* __restrict__ block_atoms,
    const int* __restrict__ tiles,
    const int* __restrict__ interacting_atoms,
    const unsigned int* __restrict__ exclusion_masks,
    const unsigned int* __restrict__ scaling_masks,
    float cutoff_sq,
    int num_tiles,
    int num_particles,
    float box_x, float box_y, float box_z,
    float inv_box_x, float inv_box_y, float inv_box_z{param_decls}{sorted_param_decls}
) {{
    int total_warps = (blockDim.x * gridDim.x) / 32;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int tgx = threadIdx.x & 31;
    int tbx = threadIdx.x - tgx;

    int pos = (int)((long long)warp_id * num_tiles / total_warps);
    int end = (int)((long long)(warp_id + 1) * num_tiles / total_warps);

    float total_energy = 0.0f;

    __shared__ int atom_indices_shared[256];
    __shared__ unsigned int excl_shared[256];
    __shared__ unsigned int scale_shared[256];

    for (; pos < end; pos++) {{
        int block_x = tiles[pos];

        int gi = block_atoms[block_x * 32 + tgx];
{i_pos_load}
        {sorted_param_load_i}

        int gj = interacting_atoms[pos * 32 + tgx];
{j_pos_load}
        {param_load_j}

        atom_indices_shared[threadIdx.x] = gj;
        excl_shared[threadIdx.x] = exclusion_masks[pos * 32 + tgx];
        scale_shared[threadIdx.x] = scaling_masks[pos * 32 + tgx];

        float force_x = 0.0f, force_y = 0.0f, force_z = 0.0f;
        float shfl_fx = 0.0f, shfl_fy = 0.0f, shfl_fz = 0.0f;

        int tj = tgx;
        for (int j = 0; j < 32; j++) {{
            unsigned int excl_j = excl_shared[tbx + tj];
            unsigned int scale_j = scale_shared[tbx + tj];
            int atom2 = atom_indices_shared[tbx + tj];

            float dx = shfl_px - px_i;
            float dy = shfl_py - py_i;
            float dz = shfl_pz - pz_i;
            dx -= box_x * roundf(dx * inv_box_x);
            dy -= box_y * roundf(dy * inv_box_y);
            dz -= box_z * roundf(dz * inv_box_z);
            float dist_sq = dx * dx + dy * dy + dz * dz;

            bool excluded = (atom2 < 0 || atom2 >= num_particles)
                         || ((excl_j >> tgx) & 1);
            bool is_14 = (scale_j >> tgx) & 1;

            if (!excluded && dist_sq > 1.0e-12f && dist_sq <= cutoff_sq && gi >= 0 && gi < num_particles) {{
                float inv_dist = rsqrtf(dist_sq);
                float r = dist_sq * inv_dist;
                {param_select}
                {expression_fragment}
                float inv_dist_force = force_magnitude * inv_dist;
                float fx = dx * inv_dist_force;
                float fy = dy * inv_dist_force;
                float fz = dz * inv_dist_force;
                force_x += fx; force_y += fy; force_z += fz;
                shfl_fx -= fx; shfl_fy -= fy; shfl_fz -= fz;
                total_energy += energy_val;
                {param_restore}
            }}
            {shuffle_code}
            tj = (tj + 1) & 31;
        }}

        if (gi >= 0 && gi < num_particles) {{
            atomicAdd(&f_x[gi], force_x);
            atomicAdd(&f_y[gi], force_y);
            atomicAdd(&f_z[gi], force_z);
        }}
        int gj_out = atom_indices_shared[threadIdx.x];
        if (gj_out >= 0 && gj_out < num_particles) {{
            atomicAdd(&f_x[gj_out], shfl_fx);
            atomicAdd(&f_y[gj_out], shfl_fy);
            atomicAdd(&f_z[gj_out], shfl_fz);
        }}
    }}

    for (int offset = 16; offset > 0; offset >>= 1) {{
        total_energy += __shfl_down_sync(0xffffffff, total_energy, offset);
    }}
    if (tgx == 0) atomicAdd(energy_buffer, total_energy);
}}'''
    return kernel


def _assemble_main_tile_kernel_v3(parameter_names, expression_fragment):
    use_posq = 'charge' in parameter_names

    if use_posq:
        param_decls = _generate_param_decls_main(parameter_names)
        sorted_param_decls = _generate_sorted_param_decls_main(parameter_names)
        sorted_param_load_i = _generate_sorted_param_load_i_main(parameter_names)
        param_load_j = _generate_param_load_j_tile_main(parameter_names)
        pos_args_decl = (
            '    const float4* __restrict__ sorted_posq,\n'
            '    const float4* __restrict__ posq,'
        )
        i_pos_load = (
            '        float4 posq_i = sorted_posq[block_x * 32 + tgx];\n'
            '        float px_i = posq_i.x;\n'
            '        float py_i = posq_i.y;\n'
            '        float pz_i = posq_i.z;'
        )
        j_pos_load = (
            '        float shfl_px = 0.0f, shfl_py = 0.0f, shfl_pz = 0.0f, _charge_j_posq = 0.0f;\n'
            '        if (gj >= 0 && gj < num_particles) {\n'
            '            float4 _pj = posq[gj];\n'
            '            shfl_px = _pj.x;\n'
            '            shfl_py = _pj.y;\n'
            '            shfl_pz = _pj.z;\n'
            '            _charge_j_posq = _pj.w;\n'
            '        }'
        )
    else:
        param_decls = _generate_param_decls_v2_main(parameter_names)
        sorted_param_decls = _generate_sorted_param_decls_v2_main(parameter_names)
        sorted_param_load_i = _generate_sorted_param_load_i_v2_main(parameter_names)
        param_load_j = _generate_param_load_j_tile_v2_main(parameter_names)
        pos_args_decl = (
            '    const float* __restrict__ sorted_pos_x,\n'
            '    const float* __restrict__ sorted_pos_y,\n'
            '    const float* __restrict__ sorted_pos_z,\n'
            '    const float* __restrict__ pos_x,\n'
            '    const float* __restrict__ pos_y,\n'
            '    const float* __restrict__ pos_z,'
        )
        i_pos_load = (
            '        float px_i = sorted_pos_x[block_x * 32 + tgx];\n'
            '        float py_i = sorted_pos_y[block_x * 32 + tgx];\n'
            '        float pz_i = sorted_pos_z[block_x * 32 + tgx];'
        )
        j_pos_load = (
            '        float shfl_px = 0.0f, shfl_py = 0.0f, shfl_pz = 0.0f;\n'
            '        if (gj >= 0 && gj < num_particles) {\n'
            '            shfl_px = pos_x[gj];\n'
            '            shfl_py = pos_y[gj];\n'
            '            shfl_pz = pos_z[gj];\n'
            '        }'
        )

    shuffle_code = _generate_shuffle_warp_data_main(parameter_names)

    kernel = f'''extern "C" __global__
void main_tile_kernel(
{pos_args_decl}
    float* __restrict__ f_x,
    float* __restrict__ f_y,
    float* __restrict__ f_z,
    float* __restrict__ energy_buffer,
    const int* __restrict__ block_atoms,
    const int* __restrict__ tiles,
    const int* __restrict__ interacting_atoms,
    float cutoff_sq,
    int num_tiles,
    int num_particles,
    float box_x, float box_y, float box_z,
    float inv_box_x, float inv_box_y, float inv_box_z{param_decls}{sorted_param_decls}
) {{
    int total_warps = (blockDim.x * gridDim.x) / 32;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int tgx = threadIdx.x & 31;
    int tbx = threadIdx.x - tgx;

    int pos = (int)((long long)warp_id * num_tiles / total_warps);
    int end = (int)((long long)(warp_id + 1) * num_tiles / total_warps);

    float total_energy = 0.0f;

    __shared__ int atom_indices_shared[256];

    for (; pos < end; pos++) {{
        int block_x = tiles[pos];

        int gi = block_atoms[block_x * 32 + tgx];
{i_pos_load}
        {sorted_param_load_i}

        int gj = interacting_atoms[pos * 32 + tgx];
{j_pos_load}
        {param_load_j}

        atom_indices_shared[threadIdx.x] = gj;

        float force_x = 0.0f, force_y = 0.0f, force_z = 0.0f;
        float shfl_fx = 0.0f, shfl_fy = 0.0f, shfl_fz = 0.0f;

        int tj = tgx;
        for (int j = 0; j < 32; j++) {{
            int atom2 = atom_indices_shared[tbx + tj];

            float dx = shfl_px - px_i;
            float dy = shfl_py - py_i;
            float dz = shfl_pz - pz_i;
            dx -= box_x * roundf(dx * inv_box_x);
            dy -= box_y * roundf(dy * inv_box_y);
            dz -= box_z * roundf(dz * inv_box_z);
            float dist_sq = dx * dx + dy * dy + dz * dz;

            bool excluded = (atom2 < 0 || atom2 >= num_particles);

            if (!excluded && dist_sq > 1.0e-12f && dist_sq <= cutoff_sq && gi >= 0 && gi < num_particles) {{
                float inv_dist = rsqrtf(dist_sq);
                float r = dist_sq * inv_dist;
                {expression_fragment}
                float inv_dist_force = force_magnitude * inv_dist;
                float fx = dx * inv_dist_force;
                float fy = dy * inv_dist_force;
                float fz = dz * inv_dist_force;
                force_x += fx; force_y += fy; force_z += fz;
                shfl_fx -= fx; shfl_fy -= fy; shfl_fz -= fz;
                total_energy += energy_val;
            }}
            {shuffle_code}
            tj = (tj + 1) & 31;
        }}

        if (gi >= 0 && gi < num_particles) {{
            atomicAdd(&f_x[gi], force_x);
            atomicAdd(&f_y[gi], force_y);
            atomicAdd(&f_z[gi], force_z);
        }}
        int gj_out = atom_indices_shared[threadIdx.x];
        if (gj_out >= 0 && gj_out < num_particles) {{
            atomicAdd(&f_x[gj_out], shfl_fx);
            atomicAdd(&f_y[gj_out], shfl_fy);
            atomicAdd(&f_z[gj_out], shfl_fz);
        }}
    }}

    for (int offset = 16; offset > 0; offset >>= 1) {{
        total_energy += __shfl_down_sync(0xffffffff, total_energy, offset);
    }}
    if (tgx == 0) atomicAdd(energy_buffer, total_energy);
}}'''
    return kernel


def _assemble_tile_kernel(parameter_names, expression_fragment):
    param_decls = _generate_param_decls(parameter_names)
    sorted_param_decls = _generate_sorted_param_decls(parameter_names)
    sorted_param_load_i = _generate_sorted_param_load_i(parameter_names)
    param_load_j = _generate_param_load_j_tile(parameter_names)
    shuffle_code = _generate_shuffle_warp_data(parameter_names)
    param_select = _generate_param_select(parameter_names)
    param_restore = _generate_param_restore(parameter_names)

    kernel = f'''extern "C" __global__
void tile_kernel(
    const float* __restrict__ sorted_pos_x,
    const float* __restrict__ sorted_pos_y,
    const float* __restrict__ sorted_pos_z,
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    float* __restrict__ f_x,
    float* __restrict__ f_y,
    float* __restrict__ f_z,
    float* __restrict__ energy_buffer,
    const int* __restrict__ block_atoms,
    const int* __restrict__ tiles,
    const int* __restrict__ interacting_atoms,
    const unsigned int* __restrict__ exclusion_masks,
    const unsigned int* __restrict__ scaling_masks,
    float cutoff_sq,
    int num_tiles,
    int num_particles,
    float box_x, float box_y, float box_z,
    float inv_box_x, float inv_box_y, float inv_box_z{param_decls}{sorted_param_decls}
) {{
    int total_warps = (blockDim.x * gridDim.x) / 32;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int tgx = threadIdx.x & 31;
    int tbx = threadIdx.x - tgx;

    int pos = (int)((long long)warp_id * num_tiles / total_warps);
    int end = (int)((long long)(warp_id + 1) * num_tiles / total_warps);

    float total_energy = 0.0f;

    __shared__ int atom_indices_shared[256];
    __shared__ unsigned int excl_shared[256];
    __shared__ unsigned int scale_shared[256];

    for (; pos < end; pos++) {{
        int block_x = tiles[pos];

        int gi = block_atoms[block_x * 32 + tgx];
        float px_i = sorted_pos_x[block_x * 32 + tgx];
        float py_i = sorted_pos_y[block_x * 32 + tgx];
        float pz_i = sorted_pos_z[block_x * 32 + tgx];
        {sorted_param_load_i}

        int gj = interacting_atoms[pos * 32 + tgx];
        float shfl_px = 0.0f, shfl_py = 0.0f, shfl_pz = 0.0f;
        if (gj >= 0 && gj < num_particles) {{
            shfl_px = pos_x[gj];
            shfl_py = pos_y[gj];
            shfl_pz = pos_z[gj];
        }}
        {param_load_j}

        atom_indices_shared[threadIdx.x] = gj;
        excl_shared[threadIdx.x] = exclusion_masks[pos * 32 + tgx];
        scale_shared[threadIdx.x] = scaling_masks[pos * 32 + tgx];

        float force_x = 0.0f, force_y = 0.0f, force_z = 0.0f;
        float shfl_fx = 0.0f, shfl_fy = 0.0f, shfl_fz = 0.0f;

        int tj = tgx;
        for (int j = 0; j < 32; j++) {{
            unsigned int excl_j = excl_shared[tbx + tj];
            unsigned int scale_j = scale_shared[tbx + tj];
            int atom2 = atom_indices_shared[tbx + tj];

            float dx = shfl_px - px_i;
            float dy = shfl_py - py_i;
            float dz = shfl_pz - pz_i;
            dx -= box_x * roundf(dx * inv_box_x);
            dy -= box_y * roundf(dy * inv_box_y);
            dz -= box_z * roundf(dz * inv_box_z);
            float dist_sq = dx * dx + dy * dy + dz * dz;

            bool excluded = (atom2 < 0 || atom2 >= num_particles)
                         || ((excl_j >> tgx) & 1);
            bool is_14 = (scale_j >> tgx) & 1;

            if (!excluded && dist_sq > 1.0e-12f && dist_sq <= cutoff_sq && gi >= 0 && gi < num_particles) {{
                float inv_dist = rsqrtf(dist_sq);
                float r = dist_sq * inv_dist;
                {param_select}
                {expression_fragment}
                float inv_dist_force = force_magnitude * inv_dist;
                float fx = dx * inv_dist_force;
                float fy = dy * inv_dist_force;
                float fz = dz * inv_dist_force;
                force_x += fx; force_y += fy; force_z += fz;
                shfl_fx -= fx; shfl_fy -= fy; shfl_fz -= fz;
                total_energy += energy_val;
                {param_restore}
            }}
            {shuffle_code}
            tj = (tj + 1) & 31;
        }}

        if (gi >= 0 && gi < num_particles) {{
            atomicAdd(&f_x[gi], force_x);
            atomicAdd(&f_y[gi], force_y);
            atomicAdd(&f_z[gi], force_z);
        }}
        int gj_out = atom_indices_shared[threadIdx.x];
        if (gj_out >= 0 && gj_out < num_particles) {{
            atomicAdd(&f_x[gj_out], shfl_fx);
            atomicAdd(&f_y[gj_out], shfl_fy);
            atomicAdd(&f_z[gj_out], shfl_fz);
        }}
    }}

    for (int offset = 16; offset > 0; offset >>= 1) {{
        total_energy += __shfl_down_sync(0xffffffff, total_energy, offset);
    }}
    if (tgx == 0) atomicAdd(energy_buffer, total_energy);
}}'''
    return kernel


def _assemble_cross_tile_kernel_v2(parameter_names, expression_fragment):
    param_decls = _generate_param_decls(parameter_names)
    param_load_i = _generate_param_load_i(parameter_names)
    param_load_j = _generate_param_load_j_init(parameter_names)
    shuffle_code = _generate_shuffle_warp_data(parameter_names)
    param_select = _generate_param_select(parameter_names)
    param_restore = _generate_param_restore(parameter_names)

    kernel = f'''extern "C" __global__
void cross_tile_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    float* __restrict__ f_x,
    float* __restrict__ f_y,
    float* __restrict__ f_z,
    float* __restrict__ energy_buffer,
    const int* __restrict__ block_atoms,
    const int* __restrict__ cross_tiles_i,
    const int* __restrict__ cross_tiles_j,
    const float* __restrict__ cross_tiles_shift,
    const unsigned int* __restrict__ cross_exclusion_masks,
    const unsigned int* __restrict__ cross_scaling_masks,
    float cutoff_sq,
    int num_cross{param_decls}
) {{
    int total_warps = (blockDim.x * gridDim.x) / 32;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int tgx = threadIdx.x & 31;

    int pos = warp_id * num_cross / total_warps;
    int end = (warp_id + 1) * num_cross / total_warps;

    float total_energy = 0.0f;

    for (; pos < end; pos++) {{
        int bi = cross_tiles_i[pos];
        int bj = cross_tiles_j[pos];
        float shift_x = cross_tiles_shift[pos * 3 + 0];
        float shift_y = cross_tiles_shift[pos * 3 + 1];
        float shift_z = cross_tiles_shift[pos * 3 + 2];

        int gi = block_atoms[bi * 32 + tgx];
        float px_i = 0.0f, py_i = 0.0f, pz_i = 0.0f;
        if (gi >= 0) {{
            px_i = pos_x[gi];
            py_i = pos_y[gi];
            pz_i = pos_z[gi];
        }}
        {param_load_i}

        int gj_init = block_atoms[bj * 32 + tgx];
        float shfl_px = 0.0f, shfl_py = 0.0f, shfl_pz = 0.0f;
        if (gj_init >= 0) {{
            shfl_px = pos_x[gj_init] + shift_x;
            shfl_py = pos_y[gj_init] + shift_y;
            shfl_pz = pos_z[gj_init] + shift_z;
        }}
        {param_load_j}

        float force_x = 0.0f, force_y = 0.0f, force_z = 0.0f;
        float shfl_fx = 0.0f, shfl_fy = 0.0f, shfl_fz = 0.0f;

        unsigned int excl = cross_exclusion_masks[pos * 32 + tgx];
        excl = (excl >> tgx) | (excl << (32 - tgx));
        unsigned int scale = cross_scaling_masks[pos * 32 + tgx];
        scale = (scale >> tgx) | (scale << (32 - tgx));

        for (int j = 0; j < 32; j++) {{
            float dx = shfl_px - px_i;
            float dy = shfl_py - py_i;
            float dz = shfl_pz - pz_i;
            float dist_sq = dx * dx + dy * dy + dz * dz;

            bool excluded = (excl & 0x1) != 0;
            bool is_14 = (scale & 0x1) != 0;

            if (!excluded && dist_sq <= cutoff_sq && dist_sq > 1.0e-12f && gi >= 0) {{
                float inv_dist = rsqrtf(dist_sq);
                float r = dist_sq * inv_dist;

                {param_select}

                {expression_fragment}

                float inv_dist_force = force_magnitude * inv_dist;
                float fx = dx * inv_dist_force;
                float fy = dy * inv_dist_force;
                float fz = dz * inv_dist_force;

                force_x += fx;
                force_y += fy;
                force_z += fz;
                shfl_fx -= fx;
                shfl_fy -= fy;
                shfl_fz -= fz;
                total_energy += energy_val;

                {param_restore}
            }}

            {shuffle_code}

            excl >>= 1;
            scale >>= 1;
        }}

        if (gi >= 0) {{
            atomicAdd(&f_x[gi], force_x);
            atomicAdd(&f_y[gi], force_y);
            atomicAdd(&f_z[gi], force_z);
        }}
        int gj = block_atoms[bj * 32 + tgx];
        if (gj >= 0) {{
            atomicAdd(&f_x[gj], shfl_fx);
            atomicAdd(&f_y[gj], shfl_fy);
            atomicAdd(&f_z[gj], shfl_fz);
        }}
    }}

    for (int offset = 16; offset > 0; offset >>= 1) {{
        total_energy += __shfl_down_sync(0xffffffff, total_energy, offset);
    }}
    if (tgx == 0) atomicAdd(energy_buffer, total_energy);
}}'''
    return kernel


def _assemble_self_tile_kernel_v2(parameter_names, expression_fragment):
    param_decls = _generate_param_decls(parameter_names)
    param_load_i_lines = []
    broadcast_j_lines = []
    select_lines = []
    restore_lines = []
    for name in parameter_names:
        param_load_i_lines.append(f'float {name}_i = 0.0f, {name}_i_14 = 0.0f;')
        param_load_i_lines.append(f'if (gi >= 0) {{ {name}_i = {name}[gi]; {name}_i_14 = {name}_14[gi]; }}')
        broadcast_j_lines.append(
            f'float {name}_j = __shfl_sync(0xffffffff, {name}_i, j);'
        )
        broadcast_j_lines.append(
            f'float {name}_j_14 = __shfl_sync(0xffffffff, {name}_i_14, j);'
        )
        select_lines.append(f'float {name}_i_saved = {name}_i;')
        select_lines.append(f'if (is_14) {name}_i = {name}_i_14;')
        select_lines.append(f'float {name}_j_saved = {name}_j;')
        select_lines.append(f'if (is_14) {name}_j = {name}_j_14;')
        restore_lines.append(f'if (is_14) {name}_i = {name}_i_saved;')
        restore_lines.append(f'if (is_14) {name}_j = {name}_j_saved;')

    param_load_i = '\n    '.join(param_load_i_lines)
    broadcast_j = '\n            '.join(broadcast_j_lines)
    param_select = '\n            '.join(select_lines)
    param_restore = '\n            '.join(restore_lines)

    kernel = f'''extern "C" __global__
void self_tile_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    float* __restrict__ f_x,
    float* __restrict__ f_y,
    float* __restrict__ f_z,
    float* __restrict__ energy_buffer,
    const int* __restrict__ block_atoms,
    const int* __restrict__ self_tile_indices,
    const unsigned int* __restrict__ self_exclusion_masks,
    const unsigned int* __restrict__ self_scaling_masks,
    float cutoff_sq{param_decls}
) {{
    int tile_idx = blockIdx.x;
    int tgx = threadIdx.x & 31;
    if (threadIdx.x >= 32) return;

    int block_k = self_tile_indices[tile_idx];
    float total_energy = 0.0f;

    int gi = block_atoms[block_k * 32 + tgx];
    float px_i = 0.0f, py_i = 0.0f, pz_i = 0.0f;
    if (gi >= 0) {{
        px_i = pos_x[gi];
        py_i = pos_y[gi];
        pz_i = pos_z[gi];
    }}
    {param_load_i}

    float force_x = 0.0f, force_y = 0.0f, force_z = 0.0f;

    unsigned int excl = self_exclusion_masks[tile_idx * 32 + tgx];
    unsigned int scale = self_scaling_masks[tile_idx * 32 + tgx];

    for (int j = 0; j < 32; j++) {{
        int gj = block_atoms[block_k * 32 + j];
        float px_j = __shfl_sync(0xffffffff, px_i, j);
        float py_j = __shfl_sync(0xffffffff, py_i, j);
        float pz_j = __shfl_sync(0xffffffff, pz_i, j);

        {broadcast_j}

        float dx = px_j - px_i;
        float dy = py_j - py_i;
        float dz = pz_j - pz_i;
        float dist_sq = dx * dx + dy * dy + dz * dz;

        bool excluded = (excl & 0x1) != 0;
        bool is_14 = (scale & 0x1) != 0;

        if (!excluded && dist_sq <= cutoff_sq && dist_sq > 1.0e-12f
            && gi >= 0 && gj >= 0 && j != tgx) {{
            float inv_dist = rsqrtf(dist_sq);
            float r = dist_sq * inv_dist;

            {param_select}

            {expression_fragment}

            float inv_dist_force = force_magnitude * inv_dist;
            force_x += dx * inv_dist_force;
            force_y += dy * inv_dist_force;
            force_z += dz * inv_dist_force;
            total_energy += energy_val;

            {param_restore}
        }}

        excl >>= 1;
        scale >>= 1;
    }}

    if (gi >= 0) {{
        atomicAdd(&f_x[gi], force_x);
        atomicAdd(&f_y[gi], force_y);
        atomicAdd(&f_z[gi], force_z);
    }}

    for (int offset = 16; offset > 0; offset >>= 1) {{
        total_energy += __shfl_down_sync(0xffffffff, total_energy, offset);
    }}
    if (tgx == 0) atomicAdd(energy_buffer, total_energy);
}}'''
    return kernel


_FORCE_ACCUMULATE = '''
        float inv_dist_force = force_magnitude * inv_dist;
        float fx = dx * inv_dist_force;
        float fy = dy * inv_dist_force;
        float fz = dz * inv_dist_force;

        atomicAdd(&f_x[gi],  fx);
        atomicAdd(&f_y[gi],  fy);
        atomicAdd(&f_z[gi],  fz);
        atomicAdd(&f_x[gj], -fx);
        atomicAdd(&f_y[gj], -fy);
        atomicAdd(&f_z[gj], -fz);
        atomicAdd(energy_buffer, energy_val);
'''


def _assemble_self_tile_kernel(parameter_names, expression_fragment):
    param_decls = _generate_param_decls(parameter_names)
    pre_fetch_code = _generate_pre_fetch(parameter_names)

    kernel = f'''extern "C" __global__
void self_tile_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    float* __restrict__ f_x,
    float* __restrict__ f_y,
    float* __restrict__ f_z,
    float* __restrict__ energy_buffer,
    const int* __restrict__ block_atoms,
    const int* __restrict__ self_tile_indices,
    const unsigned int* __restrict__ self_exclusion_masks,
    const unsigned int* __restrict__ self_scaling_masks,
    float cutoff_sq{param_decls}
) {{
    __shared__ float smem_pos[32*3];

    int tile_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_k = self_tile_indices[tile_idx];

    if (tid < 32) {{
        int gi_load = block_atoms[block_k * 32 + tid];
        if (gi_load >= 0) {{
            smem_pos[tid*3+0] = pos_x[gi_load];
            smem_pos[tid*3+1] = pos_y[gi_load];
            smem_pos[tid*3+2] = pos_z[gi_load];
        }} else {{
            smem_pos[tid*3+0] = 0.0f;
            smem_pos[tid*3+1] = 0.0f;
            smem_pos[tid*3+2] = 0.0f;
        }}
    }}
    __syncthreads();

    for (int iter = 0; iter < 2; iter++) {{
        int linear = tid * 2 + iter;
        if (linear >= 496) break;

        int row = 0;
        int cum = 0;
        while (linear >= cum + 31 - row) {{
            cum += 31 - row;
            row++;
        }}
        int col = linear - cum + row + 1;

        unsigned int excl = self_exclusion_masks[tile_idx * 32 + row];
        if (excl & (1u << col)) continue;

        unsigned int scale = self_scaling_masks[tile_idx * 32 + row];
        bool is_14 = (scale & (1u << col)) != 0;

        int gi = block_atoms[block_k * 32 + row];
        int gj = block_atoms[block_k * 32 + col];
        if (gi < 0 || gj < 0) continue;

        float dx = smem_pos[col*3+0] - smem_pos[row*3+0];
        float dy = smem_pos[col*3+1] - smem_pos[row*3+1];
        float dz = smem_pos[col*3+2] - smem_pos[row*3+2];

        float dist_sq = dx*dx + dy*dy + dz*dz;
        if (dist_sq > cutoff_sq || dist_sq < 1.0e-12f) continue;

        float inv_dist = rsqrtf(dist_sq);
        float r = 1.0f / inv_dist;

        {pre_fetch_code}

        {expression_fragment}
{_FORCE_ACCUMULATE}
    }}
}}'''
    return kernel


def _assemble_cross_tile_kernel(parameter_names, expression_fragment):
    param_decls = _generate_param_decls(parameter_names)
    pre_fetch_code = _generate_pre_fetch(parameter_names)

    kernel = f'''extern "C" __global__
void cross_tile_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    float* __restrict__ f_x,
    float* __restrict__ f_y,
    float* __restrict__ f_z,
    float* __restrict__ energy_buffer,
    const int* __restrict__ block_atoms,
    const int* __restrict__ cross_tiles_i,
    const int* __restrict__ cross_tiles_j,
    const float* __restrict__ cross_tiles_shift,
    const unsigned int* __restrict__ cross_exclusion_masks,
    const unsigned int* __restrict__ cross_scaling_masks,
    float cutoff_sq,
    int num_cross,
    int* __restrict__ tile_counter{param_decls}
) {{
    constexpr int W = 32;
    constexpr int NUM_WARPS = 8;

    __shared__ float smem_pos_i[W * 3];
    __shared__ float smem_pos_j[W * 3];
    __shared__ int smem_tile_idx;
    __shared__ int smem_bi, smem_bj;
    __shared__ float smem_energy;

    int tid = threadIdx.x;

    if (tid == 0) smem_energy = 0.0f;
    __syncthreads();

    for (;;) {{
        if (tid == 0)
            smem_tile_idx = atomicAdd(tile_counter, 1);
        __syncthreads();
        if (smem_tile_idx >= num_cross) break;

        if (tid == 0) {{
            smem_bi = cross_tiles_i[smem_tile_idx];
            smem_bj = cross_tiles_j[smem_tile_idx];
        }}
        __syncthreads();
        int bi = smem_bi;
        int bj = smem_bj;
        float shift_x = cross_tiles_shift[smem_tile_idx * 3 + 0];
        float shift_y = cross_tiles_shift[smem_tile_idx * 3 + 1];
        float shift_z = cross_tiles_shift[smem_tile_idx * 3 + 2];

        if (tid < W) {{
            int gi_load = block_atoms[bi * W + tid];
            if (gi_load >= 0) {{
                smem_pos_i[tid * 3 + 0] = pos_x[gi_load];
                smem_pos_i[tid * 3 + 1] = pos_y[gi_load];
                smem_pos_i[tid * 3 + 2] = pos_z[gi_load];
            }} else {{
                smem_pos_i[tid * 3 + 0] = 0.0f;
                smem_pos_i[tid * 3 + 1] = 0.0f;
                smem_pos_i[tid * 3 + 2] = 0.0f;
            }}
        }} else if (tid < 2 * W) {{
            int local = tid - W;
            int gj_load = block_atoms[bj * W + local];
            if (gj_load >= 0) {{
                smem_pos_j[local * 3 + 0] = pos_x[gj_load] + shift_x;
                smem_pos_j[local * 3 + 1] = pos_y[gj_load] + shift_y;
                smem_pos_j[local * 3 + 2] = pos_z[gj_load] + shift_z;
            }} else {{
                smem_pos_j[local * 3 + 0] = 0.0f;
                smem_pos_j[local * 3 + 1] = 0.0f;
                smem_pos_j[local * 3 + 2] = 0.0f;
            }}
        }}
        __syncthreads();

        for (int pair_offset = 0; pair_offset < 4; pair_offset++) {{
            int linear = tid * 4 + pair_offset;
            int tx = linear >> 5;
            int ty = linear & 31;

            unsigned int excl = cross_exclusion_masks[smem_tile_idx * W + tx];
            if (excl & (1u << ty)) continue;

            unsigned int scale_mask = cross_scaling_masks[smem_tile_idx * W + tx];
            bool is_14 = (scale_mask & (1u << ty)) != 0;

            int gi = block_atoms[bi * W + tx];
            int gj = block_atoms[bj * W + ty];
            if (gi < 0 || gj < 0) continue;

            float dx = smem_pos_j[ty * 3 + 0] - smem_pos_i[tx * 3 + 0];
            float dy = smem_pos_j[ty * 3 + 1] - smem_pos_i[tx * 3 + 1];
            float dz = smem_pos_j[ty * 3 + 2] - smem_pos_i[tx * 3 + 2];

            float dist_sq = dx * dx + dy * dy + dz * dz;
            if (dist_sq > cutoff_sq || dist_sq < 1.0e-12f) continue;

            float inv_dist = rsqrtf(dist_sq);
            float r = 1.0f / inv_dist;

            {pre_fetch_code}

            {expression_fragment}

            float inv_dist_force = force_magnitude * inv_dist;
            float fx = dx * inv_dist_force;
            float fy = dy * inv_dist_force;
            float fz = dz * inv_dist_force;

            atomicAdd(&f_x[gi],  fx);
            atomicAdd(&f_y[gi],  fy);
            atomicAdd(&f_z[gi],  fz);
            atomicAdd(&f_x[gj], -fx);
            atomicAdd(&f_y[gj], -fy);
            atomicAdd(&f_z[gj], -fz);
            atomicAdd(&smem_energy, energy_val);
        }}
        __syncthreads();
    }}

    if (tid == 0 && smem_energy != 0.0f)
        atomicAdd(energy_buffer, smem_energy);
}}'''
    return kernel


def nonbonded_expression(func):
    try:
        source = inspect.getsource(func)
    except OSError as exc:
        raise OSError(
            f'Cannot read source for {func.__name__}. '
            'The @nonbonded_expression decorator requires access to the '
            'function source code. Make sure the function is defined in a '
            '.py file (not in an interactive session).'
        ) from exc
    source = textwrap.dedent(source)
    tree = ast.parse(source)

    func_def = tree.body[0]
    if not isinstance(func_def, ast.FunctionDef):
        raise TypeError('Decorator must be applied to a function definition')

    all_args = func_def.args.args
    defaults = func_def.args.defaults
    number_defaults = len(defaults)
    number_args = len(all_args)
    number_positional = number_args - number_defaults

    index_names = []
    parameter_names = []
    distance_name = None

    for position, arg in enumerate(all_args):
        arg_name = arg.arg
        default_position = position - number_positional
        has_default = default_position >= 0

        if arg_name == 'r':
            distance_name = arg_name
        elif has_default:
            default_node = defaults[default_position]
            if _is_parameter_call(default_node):
                parameter_names.append(arg_name)
        else:
            index_names.append(arg_name)

    if distance_name is None:
        raise ValueError('Expression must have a parameter named "r"')
    if len(index_names) != 2:
        raise ValueError(
            f'Expected exactly 2 particle index arguments, got {len(index_names)}: {index_names}'
        )

    transpiler = _Transpiler(index_names, parameter_names)
    cuda_fragment = transpiler.transpile(func_def.body)
    local_variables = transpiler.local_variables

    return NonbondedExpression(
        func=func,
        source=source,
        ast_tree=tree,
        index_names=index_names,
        parameter_names=parameter_names,
        distance_name=distance_name,
        cuda_fragment=cuda_fragment,
        local_variables=local_variables,
    )


class NonbondedForce(ForceTerm):
    name = 'nonbonded'

    def __init__(self, expression):
        self.expression = expression
        self._kernel = None
        self._exclusion_kernel = None
        self._kernel_source = None
        self._exclusion_kernel_source = None
        self._d_parameter_arrays = {}
        self._parameter_arrays = {}
        self._cutoff = None
        self._cutoff_sq = None
        self._d_posq = None
        self._d_sorted_posq = None
        self._pack_posq_kernel = None
        self._gather_4comp_kernel = None

    def bind(self, topology, parameter_table, cutoff):
        self._cutoff = cutoff
        self._cutoff_sq = cutoff * cutoff
        self._num_sm = cp.cuda.runtime.getDeviceProperties(0)['multiProcessorCount']
        particle_types = topology.particle_types

        _TABLE_PARAM_MAP = {
            'sigma_half': 'sigma',
            'sqrt_epsilon': 'epsilon',
        }

        for param_name in self.expression.parameter_names:
            table_name = _TABLE_PARAM_MAP.get(param_name, param_name)
            per_atom = parameter_table.expand_to_per_atom(table_name, particle_types)
            if param_name == 'sigma_half':
                per_atom = 0.5 * per_atom
            elif param_name == 'sqrt_epsilon':
                per_atom = np.sqrt(np.maximum(per_atom, 0.0))
            self._parameter_arrays[param_name] = per_atom.astype(np.float32)

            name_14 = param_name + '_14'
            table_name_14 = _TABLE_PARAM_MAP.get(param_name, param_name) + '_14'
            has_14 = table_name_14 in parameter_table.per_type or table_name_14 in parameter_table.per_atom
            if has_14:
                per_atom_14 = parameter_table.expand_to_per_atom(table_name_14, particle_types)
                if param_name == 'sigma_half':
                    per_atom_14 = 0.5 * per_atom_14
                elif param_name == 'sqrt_epsilon':
                    per_atom_14 = np.sqrt(np.maximum(per_atom_14, 0.0))
            else:
                per_atom_14 = per_atom
            self._parameter_arrays[name_14] = per_atom_14.astype(np.float32)

        if 'sigma_half' in self.expression.parameter_names and 'sqrt_epsilon' in self.expression.parameter_names:
            N = len(particle_types)
            se = np.empty(N * 2, dtype=np.float32)
            se[0::2] = self._parameter_arrays['sigma_half']
            se[1::2] = self._parameter_arrays['sqrt_epsilon']
            self._parameter_arrays['sigma_epsilon'] = se

            se_14 = np.empty(N * 2, dtype=np.float32)
            se_14[0::2] = self._parameter_arrays['sigma_half_14']
            se_14[1::2] = self._parameter_arrays['sqrt_epsilon_14']
            self._parameter_arrays['sigma_epsilon_14'] = se_14

        self._kernel_source = self.expression.assemble_main_tile_kernel()
        self._exclusion_kernel_source = self.expression.assemble_tile_kernel()

    def _ensure_compiled(self):
        if self._kernel is not None:
            return
        self._kernel = cp.RawKernel(self._kernel_source, 'main_tile_kernel')
        self._exclusion_kernel = cp.RawKernel(self._exclusion_kernel_source.replace('void tile_kernel(', 'void exclusion_tile_kernel('), 'exclusion_tile_kernel')
        for param_name in self._parameter_arrays:
            self._d_parameter_arrays[param_name] = cp.asarray(
                self._parameter_arrays[param_name]
            )
        if self._use_posq():
            self._pack_posq_kernel = cp.RawKernel(_PACK_POSQ_KERNEL, 'pack_posq_kernel')
            N = self._parameter_arrays['charge'].shape[0]
            self._d_posq = cp.zeros(N * 4, dtype=np.float32)

    def _use_posq(self):
        return 'charge' in self.expression.parameter_names

    def bind_sorted_params(self, tile_list):
        self._ensure_compiled()
        param_arrays = {}
        for arr in _unique_gpu_arrays(self.expression.parameter_names):
            if self._use_posq() and arr == 'charge':
                param_arrays[arr + '_14'] = self._d_parameter_arrays[arr + '_14']
            else:
                param_arrays[arr] = self._d_parameter_arrays[arr]
                param_arrays[arr + '_14'] = self._d_parameter_arrays[arr + '_14']
        tile_list.gather_sorted_params(param_arrays)
        if self._use_posq():
            total_slots = tile_list.num_blocks * 32
            self._d_sorted_posq = cp.empty(total_slots * 4, dtype=np.float32)

    def _param_args(self):
        args = []
        for arr in _unique_gpu_arrays(self.expression.parameter_names):
            if self._use_posq() and arr == 'charge':
                args.append(self._d_parameter_arrays[arr + '_14'])
            else:
                args.append(self._d_parameter_arrays[arr])
                args.append(self._d_parameter_arrays[arr + '_14'])
        return args

    def _main_param_args(self):
        args = []
        for arr in _unique_gpu_arrays(self.expression.parameter_names):
            if arr == 'charge':
                continue
            if arr in {v[0] for v in _PACKED_PARAMS.values()}:
                args.append(self._d_parameter_arrays[arr])
            else:
                args.append(self._d_parameter_arrays[arr])
        return args

    def _main_sorted_param_args(self, tile_list):
        args = []
        for arr in _unique_gpu_arrays(self.expression.parameter_names):
            if arr == 'charge':
                continue
            if arr in {v[0] for v in _PACKED_PARAMS.values()}:
                args.append(getattr(tile_list, f'd_sorted_{arr}'))
            else:
                args.append(getattr(tile_list, f'd_sorted_{arr}'))
        return args

    def _sorted_param_args(self, tile_list):
        args = []
        for arr in _unique_gpu_arrays(self.expression.parameter_names):
            if self._use_posq() and arr == 'charge':
                args.append(getattr(tile_list, f'd_sorted_{arr}_14'))
            else:
                args.append(getattr(tile_list, f'd_sorted_{arr}'))
                args.append(getattr(tile_list, f'd_sorted_{arr}_14'))
        return args

    def compute(self, gpu_context, tile_list=None):
        self._ensure_compiled()

        if tile_list is None or tile_list.num_tiles == 0:
            return

        num_sm = self._num_sm
        grid_size = 4 * num_sm

        if self._use_posq():
            N = gpu_context.number_particles
            tpb = 256
            grid = ((N + tpb - 1) // tpb,)
            self._pack_posq_kernel(grid, (tpb,),
                (gpu_context.d_positions_x, gpu_context.d_positions_y,
                 gpu_context.d_positions_z,
                 self._d_parameter_arrays['charge'], self._d_posq,
                 np.int32(N)))

            total_slots = tile_list.num_blocks * 32
            grid_gather = ((total_slots + tpb - 1) // tpb,)
            if self._gather_4comp_kernel is None:
                self._gather_4comp_kernel = cp.RawKernel(
                    _GATHER_SORTED_KERNEL_4COMP, 'gather_sorted_kernel_4comp')
            self._gather_4comp_kernel(grid_gather, (tpb,),
                (self._d_posq, tile_list.d_block_atoms,
                 np.int32(total_slots), np.int32(N),
                 self._d_sorted_posq))

            num_main = getattr(tile_list, 'num_main_tiles', 0)
            if num_main > 0:
                main_args = [
                    self._d_sorted_posq,
                    self._d_posq,
                    gpu_context.d_forces_x,
                    gpu_context.d_forces_y,
                    gpu_context.d_forces_z,
                    gpu_context.d_energy,
                    tile_list.d_block_atoms,
                    tile_list.d_main_tiles,
                    tile_list.d_main_interacting_atoms,
                    np.float32(self._cutoff_sq),
                    np.int32(num_main),
                    np.int32(gpu_context.number_particles),
                    np.float32(gpu_context._box_x),
                    np.float32(gpu_context._box_y),
                    np.float32(gpu_context._box_z),
                    np.float32(gpu_context._inv_box_x),
                    np.float32(gpu_context._inv_box_y),
                    np.float32(gpu_context._inv_box_z),
                ] + self._main_param_args() + self._main_sorted_param_args(tile_list)
                self._kernel((grid_size,), (256,), tuple(main_args))

            num_excl = getattr(tile_list, 'num_exclusion_tiles', 0)
            if num_excl > 0:
                excl_grid_size = max(grid_size, (num_excl + 7) // 8)
                excl_args = [
                    self._d_sorted_posq,
                    self._d_posq,
                    gpu_context.d_forces_x,
                    gpu_context.d_forces_y,
                    gpu_context.d_forces_z,
                    gpu_context.d_energy,
                    tile_list.d_block_atoms,
                    tile_list.d_excl_tiles,
                    tile_list.d_excl_interacting_atoms,
                    tile_list.d_excl_exclusion_masks,
                    tile_list.d_excl_scaling_masks,
                    np.float32(self._cutoff_sq),
                    np.int32(num_excl),
                    np.int32(gpu_context.number_particles),
                    np.float32(gpu_context._box_x),
                    np.float32(gpu_context._box_y),
                    np.float32(gpu_context._box_z),
                    np.float32(gpu_context._inv_box_x),
                    np.float32(gpu_context._inv_box_y),
                    np.float32(gpu_context._inv_box_z),
                ] + self._param_args() + self._sorted_param_args(tile_list)
                self._exclusion_kernel((excl_grid_size,), (256,), tuple(excl_args))
        else:
            num_main = getattr(tile_list, 'num_main_tiles', 0)
            if num_main > 0:
                main_args = [
                    tile_list.d_sorted_pos_x,
                    tile_list.d_sorted_pos_y,
                    tile_list.d_sorted_pos_z,
                    gpu_context.d_positions_x,
                    gpu_context.d_positions_y,
                    gpu_context.d_positions_z,
                    gpu_context.d_forces_x,
                    gpu_context.d_forces_y,
                    gpu_context.d_forces_z,
                    gpu_context.d_energy,
                    tile_list.d_block_atoms,
                    tile_list.d_main_tiles,
                    tile_list.d_main_interacting_atoms,
                    np.float32(self._cutoff_sq),
                    np.int32(num_main),
                    np.int32(gpu_context.number_particles),
                    np.float32(gpu_context._box_x),
                    np.float32(gpu_context._box_y),
                    np.float32(gpu_context._box_z),
                    np.float32(gpu_context._inv_box_x),
                    np.float32(gpu_context._inv_box_y),
                    np.float32(gpu_context._inv_box_z),
                ] + self._main_param_args() + self._main_sorted_param_args(tile_list)
                self._kernel((grid_size,), (256,), tuple(main_args))

            num_excl = getattr(tile_list, 'num_exclusion_tiles', 0)
            if num_excl > 0:
                excl_grid_size = max(grid_size, (num_excl + 7) // 8)
                excl_args = [
                    tile_list.d_sorted_pos_x,
                    tile_list.d_sorted_pos_y,
                    tile_list.d_sorted_pos_z,
                    gpu_context.d_positions_x,
                    gpu_context.d_positions_y,
                    gpu_context.d_positions_z,
                    gpu_context.d_forces_x,
                    gpu_context.d_forces_y,
                    gpu_context.d_forces_z,
                    gpu_context.d_energy,
                    tile_list.d_block_atoms,
                    tile_list.d_excl_tiles,
                    tile_list.d_excl_interacting_atoms,
                    tile_list.d_excl_exclusion_masks,
                    tile_list.d_excl_scaling_masks,
                    np.float32(self._cutoff_sq),
                    np.int32(num_excl),
                    np.int32(gpu_context.number_particles),
                    np.float32(gpu_context._box_x),
                    np.float32(gpu_context._box_y),
                    np.float32(gpu_context._box_z),
                    np.float32(gpu_context._inv_box_x),
                    np.float32(gpu_context._inv_box_y),
                    np.float32(gpu_context._inv_box_z),
                ] + self._param_args() + self._sorted_param_args(tile_list)
                self._exclusion_kernel((excl_grid_size,), (256,), tuple(excl_args))
