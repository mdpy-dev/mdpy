import ast
import inspect
import textwrap

from mdpy.force.markers import param as _param_marker, scalar as _scalar_marker
from mdpy.force.expr_info import ExprInfo, _strip_trailing_digits
from mdpy.force.ad_engine import TapeEntry, ScalarADEngine, _HELPER_OPERATIONS
from mdpy.force.helper_registry import HELPER_REGISTRY

_MATH_FUNCTIONS = {
    'sqrt': 'sqrtf', 'sin': 'sinf', 'cos': 'cosf',
    'exp': 'expf', 'log': 'logf', 'abs': 'fabsf',
    'erf': 'erff',
}

_ARG_INDEX_TO_HELPER = {}
for _htype, _info in HELPER_REGISTRY.items():
    _ARG_INDEX_TO_HELPER[tuple(_info['arg_indices'])] = _htype


def _classify_for_bonded(func, body):
    sig = inspect.signature(func)
    positions = []
    per_particle = {}
    params = []
    scalars = []
    for i, (name, p) in enumerate(sig.parameters.items()):
        if i < body:
            positions.append(name)
        elif p.default is _param_marker:
            params.append(name)
        elif p.default is _scalar_marker:
            scalars.append(name)
        elif p.default is inspect.Parameter.empty:
            prop_name = _strip_trailing_digits(name)
            per_particle[name] = prop_name
        elif isinstance(p.default, (int, float)):
            params.append(name)
    return ExprInfo(positions, per_particle, params, scalars, body)


class _BondedASTWalker:
    def __init__(self, position_names):
        self._position_index = {name: i for i, name in enumerate(position_names)}
        self._name_map = {}
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
                self.tape.append(TapeEntry(result, 'pow', [left, right], cuda_val))
            else:
                cuda_val = f'({left} {op_str} {right})'
                op_name = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div', '%': 'mod'}[op_str]
                self.tape.append(TapeEntry(result, op_name, [left, right], cuda_val))
            self.forward_lines.append(f'float {result} = {cuda_val};')
            return result

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            operand = self._expr(node.operand)
            result = self._fresh_name()
            cuda_val = f'(0.0f - {operand})'
            self.tape.append(TapeEntry(result, 'sub', ['0.0f', operand], cuda_val))
            self.forward_lines.append(f'float {result} = {cuda_val};')
            return result

        if isinstance(node, ast.Call):
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id

            if func_name in ('distance', 'angle', 'dihedral'):
                result = self._fresh_name(func_name[0])
                arg_indices = []
                for arg in node.args:
                    if isinstance(arg, ast.Name) and arg.id in self._position_index:
                        arg_indices.append(self._position_index[arg.id])
                helper_type = _ARG_INDEX_TO_HELPER.get(tuple(arg_indices), func_name)
                self.tape.append(TapeEntry(result, func_name, [], result))
                self._helper_calls.append((helper_type, result, arg_indices))
                self.forward_lines.append(f'// {func_name} computed by helper')
                return result

            if func_name in _MATH_FUNCTIONS:
                arg = self._expr(node.args[0])
                cuda_name = _MATH_FUNCTIONS[func_name]
                result = self._fresh_name()
                cuda_val = f'{cuda_name}({arg})'
                self.tape.append(TapeEntry(result, func_name, [arg], cuda_val))
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
        self.helper_calls = []
        self._compile()

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

        walker = _BondedASTWalker(self._expr_info.positions)

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

        ad = ScalarADEngine()
        ad.differentiate(walker.tape, energy_var)

        parts = []

        for helper_type, result_name, _ in walker._helper_calls:
            template = HELPER_REGISTRY[helper_type]['forward']
            parts.append(template.format(result_name=result_name))

        for line in walker.forward_lines:
            if not line.startswith('//'):
                parts.append(f'        {line}')
            else:
                parts.append(f'        {line}')

        parts.append(f'        float _result_energy = {energy_var};')

        for helper_type, result_name, _ in walker._helper_calls:
            for entry in walker.tape:
                if entry.var_name == result_name and entry.d_output is not None:
                    template = HELPER_REGISTRY[helper_type]['force']
                    parts.append(template.format(
                        result_name=result_name,
                        grad_expr=entry.d_output,
                    ))
                    self.helper_calls.append((helper_type, result_name, entry.d_output))
                    break

        self.cuda_fragment = '\n'.join(parts)


def bonded_expression(body):
    def decorator(func):
        return _BondedExpression(func, body)
    return decorator
