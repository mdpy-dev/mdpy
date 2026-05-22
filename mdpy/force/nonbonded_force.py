from __future__ import annotations

import ast
import inspect
import re
import textwrap

import cupy as cp
import numpy as np

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
        param_decls += f',\n    const float* {param_name}'
        param_decls += f',\n    const float* {param_name}_14'
    return param_decls


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


def _generate_param_load_j_init(parameter_names):
    lines = []
    for name in parameter_names:
        lines.append(f'float {name}_j = 0.0f, {name}_j_14 = 0.0f;')
        lines.append(f'if (gj_init >= 0) {{ {name}_j = {name}[gj_init]; {name}_j_14 = {name}_14[gj_init]; }}')
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


def _assemble_cross_tile_kernel_v2(parameter_names, expression_fragment):
    param_decls = _generate_param_decls(parameter_names)
    param_load_i = _generate_param_load_i(parameter_names)
    param_load_j = _generate_param_load_j_init(parameter_names)
    shuffle_code = _generate_shuffle_warp_data(parameter_names)
    param_select = _generate_param_select(parameter_names)
    param_restore = _generate_param_restore(parameter_names)

    kernel = f'''extern "C" __global__
void cross_tile_kernel(
    const float* positions,
    float* forces,
    float* energy_buffer,
    const int* block_atoms,
    const int* cross_tiles_i,
    const int* cross_tiles_j,
    const float* cross_tiles_shift,
    const unsigned int* cross_exclusion_masks,
    const unsigned int* cross_scaling_masks,
    float cutoff_sq,
    int num_cross{param_decls}
) {{
    int total_warps = (blockDim.x * gridDim.x) / 32;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int tgx = threadIdx.x & 31;

    int pos = warp_id * num_cross / total_warps;
    int end = (warp_id + 1) * num_cross / total_warps;

    float energy = 0.0f;

    for (; pos < end; pos++) {{
        int bi = cross_tiles_i[pos];
        int bj = cross_tiles_j[pos];
        float shift_x = cross_tiles_shift[pos * 3 + 0];
        float shift_y = cross_tiles_shift[pos * 3 + 1];
        float shift_z = cross_tiles_shift[pos * 3 + 2];

        int gi = block_atoms[bi * 32 + tgx];
        float px_i = 0.0f, py_i = 0.0f, pz_i = 0.0f;
        if (gi >= 0) {{
            px_i = positions[gi * 3 + 0];
            py_i = positions[gi * 3 + 1];
            pz_i = positions[gi * 3 + 2];
        }}
        {param_load_i}

        int gj_init = block_atoms[bj * 32 + tgx];
        float shfl_px = 0.0f, shfl_py = 0.0f, shfl_pz = 0.0f;
        if (gj_init >= 0) {{
            shfl_px = positions[gj_init * 3 + 0] + shift_x;
            shfl_py = positions[gj_init * 3 + 1] + shift_y;
            shfl_pz = positions[gj_init * 3 + 2] + shift_z;
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

                force_x -= fx;
                force_y -= fy;
                force_z -= fz;
                shfl_fx += fx;
                shfl_fy += fy;
                shfl_fz += fz;
                energy += energy_val;

                {param_restore}
            }}

            {shuffle_code}

            excl >>= 1;
            scale >>= 1;
        }}

        if (gi >= 0) {{
            atomicAdd(&forces[gi * 3 + 0], force_x);
            atomicAdd(&forces[gi * 3 + 1], force_y);
            atomicAdd(&forces[gi * 3 + 2], force_z);
        }}
        int gj = block_atoms[bj * 32 + tgx];
        if (gj >= 0) {{
            atomicAdd(&forces[gj * 3 + 0], shfl_fx);
            atomicAdd(&forces[gj * 3 + 1], shfl_fy);
            atomicAdd(&forces[gj * 3 + 2], shfl_fz);
        }}
    }}

    for (int offset = 16; offset > 0; offset >>= 1) {{
        energy += __shfl_down_sync(0xffffffff, energy, offset);
    }}
    if (tgx == 0) atomicAdd(energy_buffer, energy);
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
    const float* positions,
    float* forces,
    float* energy_buffer,
    const int* block_atoms,
    const int* self_tile_indices,
    const unsigned int* self_exclusion_masks,
    const unsigned int* self_scaling_masks,
    float cutoff_sq{param_decls}
) {{
    int tile_idx = blockIdx.x;
    int tgx = threadIdx.x & 31;
    if (threadIdx.x >= 32) return;

    int block_k = self_tile_indices[tile_idx];
    float energy = 0.0f;

    int gi = block_atoms[block_k * 32 + tgx];
    float px_i = 0.0f, py_i = 0.0f, pz_i = 0.0f;
    if (gi >= 0) {{
        px_i = positions[gi * 3 + 0];
        py_i = positions[gi * 3 + 1];
        pz_i = positions[gi * 3 + 2];
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

            {broadcast_j}
            {param_select}

            {expression_fragment}

            float inv_dist_force = force_magnitude * inv_dist;
            force_x -= dx * inv_dist_force;
            force_y -= dy * inv_dist_force;
            force_z -= dz * inv_dist_force;
            energy += 0.5f * energy_val;

            {param_restore}
        }}

        excl >>= 1;
        scale >>= 1;
    }}

    if (gi >= 0) {{
        atomicAdd(&forces[gi * 3 + 0], force_x);
        atomicAdd(&forces[gi * 3 + 1], force_y);
        atomicAdd(&forces[gi * 3 + 2], force_z);
    }}

    for (int offset = 16; offset > 0; offset >>= 1) {{
        energy += __shfl_down_sync(0xffffffff, energy, offset);
    }}
    if (tgx == 0) atomicAdd(energy_buffer, energy);
}}'''
    return kernel


_FORCE_ACCUMULATE = '''
        float inv_dist_force = force_magnitude * inv_dist;
        float fx = dx * inv_dist_force;
        float fy = dy * inv_dist_force;
        float fz = dz * inv_dist_force;

        atomicAdd(&forces[gi*3+0],  fx);
        atomicAdd(&forces[gi*3+1],  fy);
        atomicAdd(&forces[gi*3+2],  fz);
        atomicAdd(&forces[gj*3+0], -fx);
        atomicAdd(&forces[gj*3+1], -fy);
        atomicAdd(&forces[gj*3+2], -fz);
        atomicAdd(energy_buffer, energy_val);
'''


def _assemble_self_tile_kernel(parameter_names, expression_fragment):
    param_decls = _generate_param_decls(parameter_names)
    pre_fetch_code = _generate_pre_fetch(parameter_names)

    kernel = f'''extern "C" __global__
void self_tile_kernel(
    const float* positions,
    float* forces,
    float* energy_buffer,
    const int* block_atoms,
    const int* self_tile_indices,
    const unsigned int* self_exclusion_masks,
    const unsigned int* self_scaling_masks,
    float cutoff_sq{param_decls}
) {{
    __shared__ float smem_pos[32*3];

    int tile_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_k = self_tile_indices[tile_idx];

    if (tid < 32) {{
        int gi_load = block_atoms[block_k * 32 + tid];
        if (gi_load >= 0) {{
            smem_pos[tid*3+0] = positions[gi_load*3+0];
            smem_pos[tid*3+1] = positions[gi_load*3+1];
            smem_pos[tid*3+2] = positions[gi_load*3+2];
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
    const float* positions,
    float* forces,
    float* energy_buffer,
    const int* block_atoms,
    const int* cross_tiles_i,
    const int* cross_tiles_j,
    const float* cross_tiles_shift,
    const unsigned int* cross_exclusion_masks,
    const unsigned int* cross_scaling_masks,
    float cutoff_sq,
    int num_cross,
    int* tile_counter{param_decls}
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
                smem_pos_i[tid * 3 + 0] = positions[gi_load * 3 + 0];
                smem_pos_i[tid * 3 + 1] = positions[gi_load * 3 + 1];
                smem_pos_i[tid * 3 + 2] = positions[gi_load * 3 + 2];
            }} else {{
                smem_pos_i[tid * 3 + 0] = 0.0f;
                smem_pos_i[tid * 3 + 1] = 0.0f;
                smem_pos_i[tid * 3 + 2] = 0.0f;
            }}
        }} else if (tid < 2 * W) {{
            int local = tid - W;
            int gj_load = block_atoms[bj * W + local];
            if (gj_load >= 0) {{
                smem_pos_j[local * 3 + 0] = positions[gj_load * 3 + 0] + shift_x;
                smem_pos_j[local * 3 + 1] = positions[gj_load * 3 + 1] + shift_y;
                smem_pos_j[local * 3 + 2] = positions[gj_load * 3 + 2] + shift_z;
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

            atomicAdd(&forces[gi * 3 + 0],  fx);
            atomicAdd(&forces[gi * 3 + 1],  fy);
            atomicAdd(&forces[gi * 3 + 2],  fz);
            atomicAdd(&forces[gj * 3 + 0], -fx);
            atomicAdd(&forces[gj * 3 + 1], -fy);
            atomicAdd(&forces[gj * 3 + 2], -fz);
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
        self._self_kernel = None
        self._cross_kernel = None
        self._self_kernel_source = None
        self._cross_kernel_source = None
        self._d_parameter_arrays = {}
        self._parameter_arrays = {}
        self._cutoff = None
        self._cutoff_sq = None

    def bind(self, topology, parameter_table, cutoff):
        self._cutoff = cutoff
        self._cutoff_sq = cutoff * cutoff
        self._num_sm = cp.cuda.runtime.getDeviceProperties(0)['multiProcessorCount']
        particle_types = topology.particle_types

        for param_name in self.expression.parameter_names:
            per_atom = parameter_table.expand_to_per_atom(param_name, particle_types)
            self._parameter_arrays[param_name] = per_atom.astype(np.float32)

            name_14 = param_name + '_14'
            if name_14 in parameter_table.per_type or name_14 in parameter_table.per_atom:
                per_atom_14 = parameter_table.expand_to_per_atom(name_14, particle_types)
            else:
                per_atom_14 = per_atom
            self._parameter_arrays[name_14] = per_atom_14.astype(np.float32)

        self._self_kernel_source = self.expression.assemble_self_tile_kernel()
        self._cross_kernel_source = self.expression.assemble_cross_tile_kernel()

    def _ensure_compiled(self):
        if self._self_kernel is not None:
            return
        self._self_kernel = cp.RawKernel(self._self_kernel_source, 'self_tile_kernel')
        self._cross_kernel = cp.RawKernel(self._cross_kernel_source, 'cross_tile_kernel')
        for param_name in self._parameter_arrays:
            self._d_parameter_arrays[param_name] = cp.asarray(
                self._parameter_arrays[param_name]
            )

    def _param_args(self):
        args = []
        for param_name in self.expression.parameter_names:
            args.append(self._d_parameter_arrays[param_name])
            args.append(self._d_parameter_arrays[param_name + '_14'])
        return args

    def compute(self, gpu_context, tile_list=None):
        self._ensure_compiled()

        if tile_list is None:
            return 0.0

        if tile_list.num_self > 0:
            self_args = [
                gpu_context.d_positions,
                gpu_context.d_forces,
                gpu_context.d_energy,
                tile_list.d_block_atoms,
                tile_list.d_self_tile_indices,
                tile_list.d_self_exclusion_masks,
                tile_list.d_self_scaling_masks,
                np.float32(self._cutoff_sq),
            ] + self._param_args()
            self._self_kernel(
                (tile_list.num_self,), (256,),
                tuple(self_args),
            )

        if tile_list.num_cross > 0:
            cross_args = [
                gpu_context.d_positions,
                gpu_context.d_forces,
                gpu_context.d_energy,
                tile_list.d_block_atoms,
                tile_list.d_cross_tiles_i,
                tile_list.d_cross_tiles_j,
                tile_list.d_cross_tiles_shift,
                tile_list.d_cross_exclusion_masks,
                tile_list.d_cross_scaling_masks,
                np.float32(self._cutoff_sq),
                np.int32(tile_list.num_cross),
            ] + self._param_args()
            num_sm = self._num_sm
            cross_grid = 4 * num_sm
            self._cross_kernel(
                (cross_grid,), (256,),
                tuple(cross_args),
            )

        if tile_list.num_interactions == 0:
            return 0.0
        return None
