import ast
import inspect
import textwrap

from mdpy.force.primitives import param as _param_marker, scalar as _scalar_marker, point as _point_marker
from mdpy.force._utils import ExprInfo, _strip_trailing_digits, _MATH_FUNCTIONS, _numeric_literal
from mdpy.force.ad_engine import TapeEntry, ForwardADEngine, _HELPER_OPERATIONS

_DISTANCE_FORWARD = r'''
        float3 _delta_{rn} = pbc_wrap_vec(sub_f3(load_pos(pos_x,pos_y,pos_z,{atom_b}), load_pos(pos_x,pos_y,pos_z,{atom_a})), pbc_inv, pbc_matrix);
        float {rn} = len_f3(_delta_{rn});
        float _inv_r_{rn} = 0.0f;
        if ({rn} >= 1e-12f) {{
            _inv_r_{rn} = 1.0f / {rn};
        }}
        float3 _partial_{rn}_0 = scale_f3(_delta_{rn}, -_inv_r_{rn});
        float3 _partial_{rn}_1 = scale_f3(_delta_{rn}, _inv_r_{rn});
'''

_ANGLE_FORWARD = r'''
        float3 _r1_{rn} = pbc_wrap_vec(sub_f3(load_pos(pos_x,pos_y,pos_z,{arm1}), load_pos(pos_x,pos_y,pos_z,{vertex})), pbc_inv, pbc_matrix);
        float3 _r2_{rn} = pbc_wrap_vec(sub_f3(load_pos(pos_x,pos_y,pos_z,{arm2}), load_pos(pos_x,pos_y,pos_z,{vertex})), pbc_inv, pbc_matrix);
        float _l1_{rn} = len_f3(_r1_{rn});
        float _l2_{rn} = len_f3(_r2_{rn});
        if (_l1_{rn} < 1e-12f || _l2_{rn} < 1e-12f) continue;
        float _inv_l1_{rn} = 1.0f / _l1_{rn};
        float _inv_l2_{rn} = 1.0f / _l2_{rn};
        float _ct_{rn} = dot_f3(_r1_{rn}, _r2_{rn}) * _inv_l1_{rn} * _inv_l2_{rn};
        _ct_{rn} = fmaxf(-1.0f, fminf(1.0f, _ct_{rn}));
        float {rn} = acosf(_ct_{rn});
        float3 _n_{rn} = cross_f3(_r1_{rn}, _r2_{rn});
        float3 _c1_{rn} = cross_f3(_r1_{rn}, _n_{rn});
        float _lc1_{rn} = len_f3(_c1_{rn});
        float3 _partial_{rn}_0 = make_f3(0.0f, 0.0f, 0.0f);
        if (_lc1_{rn} > 1e-12f) _partial_{rn}_0 = scale_f3(_c1_{rn}, 1.0f / (_lc1_{rn} * _l1_{rn}));
        float3 _c3_{rn} = cross_f3(scale_f3(_r2_{rn}, -1.0f), _n_{rn});
        float _lc3_{rn} = len_f3(_c3_{rn});
        float3 _partial_{rn}_2 = make_f3(0.0f, 0.0f, 0.0f);
        if (_lc3_{rn} > 1e-12f) _partial_{rn}_2 = scale_f3(_c3_{rn}, 1.0f / (_lc3_{rn} * _l2_{rn}));
        float3 _partial_{rn}_1 = scale_f3(add_f3(_partial_{rn}_0, _partial_{rn}_2), -1.0f);
'''

_DIHEDRAL_FORWARD = r'''
        float3 _rab_{rn} = pbc_wrap_vec(sub_f3(load_pos(pos_x,pos_y,pos_z,{b}), load_pos(pos_x,pos_y,pos_z,{a})), pbc_inv, pbc_matrix);
        float3 _rbc_{rn} = pbc_wrap_vec(sub_f3(load_pos(pos_x,pos_y,pos_z,{c}), load_pos(pos_x,pos_y,pos_z,{b})), pbc_inv, pbc_matrix);
        float3 _rcd_{rn} = pbc_wrap_vec(sub_f3(load_pos(pos_x,pos_y,pos_z,{d}), load_pos(pos_x,pos_y,pos_z,{c})), pbc_inv, pbc_matrix);
        float _lab_{rn} = len_f3(_rab_{rn}), _lbc_{rn} = len_f3(_rbc_{rn}), _lcd_{rn} = len_f3(_rcd_{rn});
        if (_lab_{rn} < 1e-12f || _lbc_{rn} < 1e-12f || _lcd_{rn} < 1e-12f) continue;
        float3 _n1_{rn} = cross_f3(_rab_{rn}, _rbc_{rn});
        float3 _n2_{rn} = cross_f3(_rbc_{rn}, _rcd_{rn});
        float _dn_{rn} = dot_f3(_n1_{rn}, _n2_{rn});
        float _drn_{rn} = dot_f3(_rab_{rn}, _n2_{rn});
        float {rn} = atan2f(_lbc_{rn} * _drn_{rn}, _dn_{rn});
        float _n1s_{rn} = dot_f3(_n1_{rn}, _n1_{rn});
        float _n2s_{rn} = dot_f3(_n2_{rn}, _n2_{rn});
        if (_n1s_{rn} < 1e-12f || _n2s_{rn} < 1e-12f) continue;
        float3 _partial_{rn}_0 = scale_f3(_n1_{rn}, -_lbc_{rn} / _n1s_{rn});
        float3 _partial_{rn}_3 = scale_f3(_n2_{rn}, _lbc_{rn} / _n2s_{rn});
        float3 _voc_{rn} = scale_f3(_rbc_{rn}, 0.5f);
        float _loc_{rn} = _lbc_{rn} * 0.5f;
        float _ils_{rn} = 1.0f / (_loc_{rn} * _loc_{rn});
        float3 _t1_{rn} = cross_f3(_voc_{rn}, _partial_{rn}_3);
        float3 _t2_{rn} = scale_f3(cross_f3(_rcd_{rn}, _partial_{rn}_3), 0.5f);
        float3 _t3_{rn} = scale_f3(cross_f3(scale_f3(_rab_{rn}, -1.0f), _partial_{rn}_0), 0.5f);
        float3 _st_{rn} = scale_f3(add_f3(_t1_{rn}, add_f3(_t2_{rn}, _t3_{rn})), -1.0f);
        float3 _partial_{rn}_2 = scale_f3(cross_f3(_st_{rn}, _voc_{rn}), _ils_{rn});
        float3 _partial_{rn}_1 = scale_f3(add_f3(_partial_{rn}_0, add_f3(_partial_{rn}_2, _partial_{rn}_3)), -1.0f);
'''

HELPER_REGISTRY = {
    'distance': {
        'position_args': ['atom_a', 'atom_b'],
        'forward': _DISTANCE_FORWARD,
    },
    'angle': {
        'position_args': ['arm1', 'vertex', 'arm2'],
        'forward': _ANGLE_FORWARD,
    },
    'dihedral': {
        'position_args': ['a', 'b', 'c', 'd'],
        'forward': _DIHEDRAL_FORWARD,
    },
}


def _classify_for_bonded(func, body):
    sig = inspect.signature(func)
    positions = []
    per_particle = {}
    params = []
    scalars = []
    point_params = []
    for i, (name, p) in enumerate(sig.parameters.items()):
        if i < body:
            positions.append(name)
        elif p.default is _param_marker:
            params.append(name)
        elif p.default is _point_marker:
            point_params.extend([f'{name}_x', f'{name}_y', f'{name}_z'])
        elif p.default is _scalar_marker:
            scalars.append(name)
        elif p.default is inspect.Parameter.empty:
            prop_name = _strip_trailing_digits(name)
            per_particle[name] = prop_name
        elif isinstance(p.default, (int, float)):
            params.append(name)
    params.extend(point_params)
    return ExprInfo(positions, per_particle, params, scalars, body)


def _build_projection(result_name, position_args, arg_indices, grad_expr):
    """Emit the generic force-projection CUDA for one helper call.

    For each atom the geometry reads (slot i maps to atom a{arg_indices[i]+1}),
    apply F_i = -(dE/dq) * partial_i.
    """
    lines = ['        {', f'            float _neg_grad_{result_name} = -({grad_expr});']
    for i in range(len(position_args)):
        atom = f'a{arg_indices[i] + 1}'
        lines.append(
            f'            add_force(f_x,f_y,f_z, {atom}, '
            f'scale_f3(_partial_{result_name}_{i}, _neg_grad_{result_name}));'
        )
    lines.append('        }')
    return '\n'.join(lines)


class _BondedASTWalker:
    def __init__(self, position_names, module_globals=None):
        self._position_index = {name: i for i, name in enumerate(position_names)}
        self._name_map = {}
        self._module_globals = module_globals if module_globals is not None else {}
        self.tape = []
        self.forward_lines = []
        self._helper_calls = []
        self._counter = 0

    def _fresh_name(self, prefix='_t'):
        self._counter += 1
        return f'{prefix}{self._counter}'

    def _expr(self, node):
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, float):
                return f'{v}f'
            if isinstance(v, int) and not isinstance(v, bool):
                return str(v)
            return str(v)

        if isinstance(node, ast.Name):
            name = node.id
            if name in self._name_map:
                return self._name_map[name]
            if name in self._module_globals:
                literal = _numeric_literal(self._module_globals[name])
                if literal is not None:
                    return literal
            return name

        if isinstance(node, ast.BinOp):
            left = self._expr(node.left)
            right = self._expr(node.right)
            op_map = {
                ast.Add: '+', ast.Sub: '-', ast.Mult: '*',
                ast.Div: '/', ast.Mod: '%', ast.Pow: '**',
            }
            op_str = op_map.get(type(node.op))
            if op_str is None:
                raise ValueError(f"Unsupported binary op: {type(node.op).__name__}")
            result = self._fresh_name()
            if op_str == '**':
                cuda_val = f'powf({left}, {right})'
                self.tape.append(TapeEntry(result, 'pow', [left, right]))
            else:
                cuda_val = f'({left} {op_str} {right})'
                op_name = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div', '%': 'mod'}[op_str]
                self.tape.append(TapeEntry(result, op_name, [left, right]))
            self.forward_lines.append(f'float {result} = {cuda_val};')
            return result

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            operand = self._expr(node.operand)
            result = self._fresh_name()
            cuda_val = f'(0.0f - {operand})'
            self.tape.append(TapeEntry(result, 'sub', ['0.0f', operand]))
            self.forward_lines.append(f'float {result} = {cuda_val};')
            return result

        if isinstance(node, ast.Call):
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id

            if func_name in ('distance', 'angle', 'dihedral'):
                result = self._fresh_name('_h')
                arg_indices = []
                for arg in node.args:
                    if isinstance(arg, ast.Name) and arg.id in self._position_index:
                        arg_indices.append(self._position_index[arg.id])
                # helper_type is just func_name now — no lookup table needed
                self.tape.append(TapeEntry(result, func_name, []))
                self._helper_calls.append((func_name, result, arg_indices))
                self.forward_lines.append(f'// {func_name} computed by helper')
                return result

            if func_name in _MATH_FUNCTIONS:
                arg = self._expr(node.args[0])
                cuda_name = _MATH_FUNCTIONS[func_name]
                result = self._fresh_name()
                cuda_val = f'{cuda_name}({arg})'
                self.tape.append(TapeEntry(result, func_name, [arg]))
                self.forward_lines.append(f'float {result} = {cuda_val};')
                return result

            raise ValueError(f"Unsupported function call: {func_name}")

        raise ValueError(f"Unsupported AST node: {ast.dump(node)}")


class _BondedExpression:
    def __init__(self, func, body):
        self._func = func
        self._expr_info = _classify_for_bonded(func, body)
        self.body = body
        self.parameter_names = self._expr_info.params
        self.cuda_fragment = ''
        self._compile()

    @property
    def per_particle(self):
        return self._expr_info.per_particle

    def _compile(self):
        source = textwrap.dedent(inspect.getsource(self._func))
        tree = ast.parse(source)

        func_def = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                func_def = node
                break
        if func_def is None:
            return

        walker = _BondedASTWalker(self._expr_info.positions, self._func.__globals__)

        energy_var = None
        for stmt in func_def.body:
            if isinstance(stmt, ast.Pass):
                continue
            if isinstance(stmt, ast.Assign):
                result = walker._expr(stmt.value)
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        walker._name_map[target.id] = result
            elif isinstance(stmt, ast.Return) and stmt.value is not None:
                energy_var = walker._expr(stmt.value)

        if energy_var is None:
            return

        fwd_ad = ForwardADEngine()

        parts = []

        for helper_type, result_name, arg_indices in walker._helper_calls:
            entry = HELPER_REGISTRY[helper_type]
            template = entry['forward']
            position_args = entry['position_args']
            fmt = {'rn': result_name}
            for ph, idx in zip(position_args, arg_indices):
                fmt[ph] = f'a{idx + 1}'
            parts.append(template.format(**fmt))

        for line in walker.forward_lines:
            parts.append(f'        {line}')

        parts.append(f'        float _result_energy = {energy_var};')

        num_helpers = len(walker._helper_calls)
        if num_helpers <= 1:
            shared_grad_lines, shared_derivs = fwd_ad.differentiate(walker.tape)
            for line in shared_grad_lines:
                parts.append(f'        {line}')

        for idx, (helper_type, result_name, arg_indices) in enumerate(walker._helper_calls):
            entry = HELPER_REGISTRY[helper_type]
            position_args = entry['position_args']
            if num_helpers > 1:
                prefix = f'_g{idx}_'
                grad_lines, derivs = fwd_ad.differentiate(
                    walker.tape, seed_vars={result_name: '1.0f'}, prefix=prefix,
                )
                for line in grad_lines:
                    parts.append(f'        {line}')
            else:
                derivs = shared_derivs

            grad_expr = derivs.get(energy_var, '0.0f')
            parts.append(_build_projection(result_name, position_args, arg_indices, grad_expr))

        self.cuda_fragment = '\n'.join(parts)


def bonded_expression(body):
    def decorator(func):
        return _BondedExpression(func, body)
    return decorator
