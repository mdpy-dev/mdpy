from __future__ import annotations

import ast
import inspect
import textwrap

_MATH_FUNCTIONS = {
    'sqrt': 'sqrtf', 'sin': 'sinf', 'cos': 'cosf',
    'tan': 'tanf', 'acos': 'acosf', 'asin': 'asinf',
    'atan': 'atanf', 'atan2': 'atan2f', 'exp': 'expf',
    'log': 'logf', 'abs': 'fabsf', 'floor': 'floorf',
    'ceil': 'ceilf', 'min': 'fminf', 'max': 'fmaxf',
}


class BondedExpression:
    def __init__(self, body, param_names, geometric_names, cuda_fragment, local_variables):
        self.body = body
        self.param_names = param_names
        self.geometric_names = geometric_names
        self.cuda_fragment = cuda_fragment
        self.local_variables = local_variables


def bonded_expression(body):
    def decorator(func):
        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
        func_def = tree.body[0]
        func_name = func_def.name
        args = func_def.args.args

        geometric_names = []
        param_names = []
        for i, arg in enumerate(args):
            if arg.arg == 'self':
                continue
            if i < len(args) and not _has_parameter_default(func_def, arg.arg):
                geometric_names.append(arg.arg)
            else:
                param_names.append(arg.arg)

        transpiler = _BondedTranspiler(geometric_names, param_names)
        cuda_fragment = transpiler.transpile(func_def)
        local_variables = transpiler.local_variables

        return BondedExpression(
            body=body,
            param_names=param_names,
            geometric_names=geometric_names,
            cuda_fragment=cuda_fragment,
            local_variables=local_variables,
        )
    return decorator


def _has_parameter_default(func_def, arg_name):
    defaults = func_def.args.defaults
    args = func_def.args.args
    num_defaults = len(defaults)
    num_args = len(args)
    for i, arg in enumerate(args):
        if arg.arg == arg_name:
            return i >= (num_args - num_defaults)
    return False


class _BondedTranspiler(ast.NodeVisitor):
    def __init__(self, geometric_names, param_names):
        self.geometric_names = geometric_names
        self.param_names = param_names
        self.lines = []
        self.local_variables = set()

    def transpile(self, func_def):
        for stmt in func_def.body:
            self.visit(stmt)
        return '\n'.join(self.lines)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.local_variables.add(target.id)
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        value = self._expr(node.value)
        for name in names:
            self.lines.append(f'float {name} = {value};')

    def visit_Return(self, node):
        if isinstance(node.value, ast.Tuple):
            elements = node.value.elts
            if len(elements) == 2:
                self.lines.append(f'float _result_energy = {self._expr(elements[0])};')
                self.lines.append(f'float _result_force = {self._expr(elements[1])};')
            elif len(elements) == 3:
                self.lines.append(f'float _result_energy = {self._expr(elements[0])};')
                self.lines.append(f'float _result_force_0 = {self._expr(elements[1])};')
                self.lines.append(f'float _result_force_1 = {self._expr(elements[2])};')
        else:
            self.lines.append(f'float _result_energy = {self._expr(node.value)};')

    def _expr(self, node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, float):
                return f'{node.value}f'
            return str(float(node.value)) + 'f'
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.BinOp):
            left = self._expr(node.left)
            right = self._expr(node.right)
            op = self._binop(node.op)
            return f'({left} {op} {right})'
        if isinstance(node, ast.UnaryOp):
            operand = self._expr(node.operand)
            if isinstance(node.op, ast.USub):
                return f'(-{operand})'
            if isinstance(node.op, ast.UAdd):
                return f'(+{operand})'
        if isinstance(node, ast.Call):
            func_name = self._call_name(node.func)
            if func_name in _MATH_FUNCTIONS:
                cuda_name = _MATH_FUNCTIONS[func_name]
                args = ', '.join(self._expr(a) for a in node.args)
                return f'{cuda_name}({args})'
            if func_name == 'pow':
                base = self._expr(node.args[0])
                exp_node = node.args[1]
                if isinstance(exp_node, ast.Constant):
                    exp_val = exp_node.value
                    if exp_val == 0:
                        return '1.0f'
                    if exp_val == 1:
                        return base
                    if exp_val == 2:
                        return f'({base} * {base})'
                args = ', '.join(self._expr(a) for a in node.args)
                return f'powf({args})'
            args = ', '.join(self._expr(a) for a in node.args)
            return f'{func_name}({args})'
        if isinstance(node, ast.BoolOp):
            op = ' && ' if isinstance(node.op, ast.And) else ' || '
            return op.join(self._expr(v) for v in node.values)
        if isinstance(node, ast.Compare):
            left = self._expr(node.left)
            parts = []
            for op, comparator in zip(node.ops, node.comparators):
                right = self._expr(comparator)
                parts.append(f'({left} {self._cmpop(op)} {right})')
            return ' && '.join(parts)
        return '0.0f'

    def _binop(self, op):
        ops = {
            ast.Add: '+', ast.Sub: '-', ast.Mult: '*',
            ast.Div: '/', ast.Mod: '%',
        }
        return ops.get(type(op), '?')

    def _cmpop(self, op):
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
