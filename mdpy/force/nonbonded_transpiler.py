import ast
import inspect
import textwrap

from mdpy.force.primitives import param as _param_marker, scalar as _scalar_marker
from mdpy.force._utils import ExprInfo, _strip_trailing_digits, _MATH_FUNCTIONS, _numeric_literal
from mdpy.force.ad_engine import TapeEntry, ForwardADEngine


def _classify_for_nonbonded(func):
    sig = inspect.signature(func)
    body = 2
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
    return ExprInfo(positions, per_particle, params, scalars, body)


class _NonbondedASTWalker:
    def __init__(self, position_names, module_globals=None):
        self._position_index = {name: i for i, name in enumerate(position_names)}
        self._name_map = {}
        self._module_globals = module_globals if module_globals is not None else {}
        self.tape = []
        self.forward_lines = []
        self._counter = 0

    def _fresh_name(self, prefix='_t'):
        self._counter += 1
        return f'{prefix}{self._counter}'

    def _emit_mul(self, a, b):
        result = self._fresh_name()
        cuda_val = f'({a} * {b})'
        self.tape.append(TapeEntry(result, 'mul', [a, b]))
        self.forward_lines.append(f'float {result} = {cuda_val};')
        return result

    def _expand_pow(self, base_str, exp_str):
        try:
            n = int(exp_str)
        except (ValueError, TypeError):
            n = None
        if n is None or n < 0:
            result = self._fresh_name()
            cuda_val = f'powf({base_str}, {exp_str})'
            self.tape.append(TapeEntry(result, 'pow', [base_str, exp_str]))
            self.forward_lines.append(f'float {result} = {cuda_val};')
            return result
        if n == 0:
            return '1.0f'
        if n == 1:
            return base_str
        if n == 2:
            return self._emit_mul(base_str, base_str)
        if n == 3:
            return self._emit_mul(self._emit_mul(base_str, base_str), base_str)
        if n == 6:
            cube = self._emit_mul(base_str, self._emit_mul(base_str, base_str))
            return self._emit_mul(cube, cube)
        if n == 12:
            cube = self._emit_mul(base_str, self._emit_mul(base_str, base_str))
            six = self._emit_mul(cube, cube)
            return self._emit_mul(six, six)
        sq = self._emit_mul(base_str, base_str)
        cur = sq
        for _ in range(n // 2 - 1):
            cur = self._emit_mul(cur, sq)
        if n % 2 == 1:
            cur = self._emit_mul(cur, base_str)
        return cur

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
            if op_str == '**':
                return self._expand_pow(left, right)
            if op_str == '/' and right == 'r':
                result = self._fresh_name()
                cuda_val = f'({left} * inv_dist)'
                self.tape.append(TapeEntry(result, 'div', [left, 'r']))
                self.forward_lines.append(f'float {result} = {cuda_val};')
                return result
            result = self._fresh_name()
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

            if func_name == 'distance':
                self.tape.append(TapeEntry('r', 'distance', []))
                self.forward_lines.append('// distance computed externally')
                return 'r'

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


class _NonbondedExpression:
    def __init__(self, func):
        self._func = func
        self._expr_info = _classify_for_nonbonded(func)
        self.energy_cuda = ''
        self.dEdr_cuda = None
        self.grad_cuda = None
        self._local_vars = set()
        self._compile()

    @property
    def expr_info(self):
        return self._expr_info

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

        walker = _NonbondedASTWalker(self._expr_info.positions, self._func.__globals__)

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

        parts = []
        for line in walker.forward_lines:
            parts.append(f'        {line}')
        parts.append(f'        float _result_energy = {energy_var};')
        self.energy_cuda = '\n'.join(parts)
        self._local_vars = {line.split()[1].split('=')[0].strip()
                            for line in walker.forward_lines
                            if not line.startswith('//') and '=' in line}
        self._local_vars.add('_result_energy')

        fwd_ad = ForwardADEngine()
        grad_lines, derivs = fwd_ad.differentiate(walker.tape)
        if grad_lines:
            indented = '\n'.join(f'        {line}' for line in grad_lines)
            self.grad_cuda = indented
            for line in grad_lines:
                var_name = line.split()[1].split('=')[0].strip()
                self._local_vars.add(var_name)
            self.dEdr_cuda = derivs.get(energy_var, '0.0f')
        else:
            self.grad_cuda = None
            self.dEdr_cuda = '0.0f'

    def __add__(self, other):
        if not isinstance(other, _NonbondedExpression):
            return NotImplemented

        merged = _NonbondedExpression.__new__(_NonbondedExpression)
        merged._func = None

        merged_params = list(dict.fromkeys(
            self._expr_info.params + other._expr_info.params
        ))
        merged_per_particle = {}
        merged_per_particle.update(self._expr_info.per_particle)
        merged_per_particle.update(other._expr_info.per_particle)
        merged_scalars = list(dict.fromkeys(
            self._expr_info.scalars + other._expr_info.scalars
        ))

        merged._expr_info = ExprInfo(
            self._expr_info.positions,
            merged_per_particle,
            merged_params,
            merged_scalars,
            2,
        )

        suffix = "_2"
        self_locals = self._local_vars | set(self._expr_info.params)
        other_locals = set()
        for var in other._local_vars:
            if var in self_locals:
                other_locals.add(var + suffix)
            else:
                other_locals.add(var)

        import re as _re

        def _rename_vars(text):
            if text is None:
                return None
            for var in sorted(other._local_vars, key=len, reverse=True):
                if var in self_locals:
                    text = _re.sub(
                        r'\b' + _re.escape(var) + r'\b', var + suffix, text
                    )
            return text

        renamed_energy_2 = _rename_vars(other.energy_cuda)
        renamed_grad_2 = _rename_vars(other.grad_cuda)

        merged.energy_cuda = self.energy_cuda + '\n' + renamed_energy_2

        if self.grad_cuda and renamed_grad_2:
            merged.grad_cuda = self.grad_cuda + '\n' + renamed_grad_2
        elif self.grad_cuda:
            merged.grad_cuda = self.grad_cuda
        elif renamed_grad_2:
            merged.grad_cuda = renamed_grad_2
        else:
            merged.grad_cuda = None

        if self.dEdr_cuda is not None and other.dEdr_cuda is not None:
            dEdr_2 = _rename_vars(other.dEdr_cuda)
            merged.dEdr_cuda = f'({self.dEdr_cuda} + {dEdr_2})'
        elif self.dEdr_cuda is not None:
            merged.dEdr_cuda = self.dEdr_cuda
        else:
            merged.dEdr_cuda = other.dEdr_cuda

        merged._local_vars = self._local_vars | other_locals
        return merged


def nonbonded_expression(func):
    return _NonbondedExpression(func)
