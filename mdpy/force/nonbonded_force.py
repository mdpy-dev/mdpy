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
    "sqrt": "sqrtf",
    "exp": "expf",
    "log": "logf",
    "abs": "fabsf",
    "sin": "sinf",
    "cos": "cosf",
    "tan": "tanf",
}

_PACKED_PARAMS = {
    "sigma_half": ("sigma_epsilon", "x"),
    "sqrt_epsilon": ("sigma_epsilon", "y"),
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


_PACK_POSQ_KERNEL = r"""
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
"""

_PACK_SORTED_POSQ_KERNEL = r"""
extern "C" __global__
void pack_sorted_posq_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    const float* __restrict__ charge,
    const int* __restrict__ block_atoms,
    int num_particles,
    int total_slots,
    float* __restrict__ posq,
    float* __restrict__ sorted_posq
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_particles) {
        posq[idx * 4 + 0] = pos_x[idx];
        posq[idx * 4 + 1] = pos_y[idx];
        posq[idx * 4 + 2] = pos_z[idx];
        posq[idx * 4 + 3] = charge[idx];
    }
    if (idx < total_slots) {
        int atom_id = block_atoms[idx];
        if (atom_id >= 0 && atom_id < num_particles) {
            sorted_posq[idx * 4 + 0] = pos_x[atom_id];
            sorted_posq[idx * 4 + 1] = pos_y[atom_id];
            sorted_posq[idx * 4 + 2] = pos_z[atom_id];
            sorted_posq[idx * 4 + 3] = charge[atom_id];
        } else {
            sorted_posq[idx * 4 + 0] = 0.0f;
            sorted_posq[idx * 4 + 1] = 0.0f;
            sorted_posq[idx * 4 + 2] = 0.0f;
            sorted_posq[idx * 4 + 3] = 0.0f;
        }
    }
}
"""


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
        return "\n".join(self.lines)

    def visit_Assign(self, node):
        if len(node.targets) != 1:
            raise NotImplementedError("Multiple assignment targets not supported")
        target_name = node.targets[0].id
        self.local_variables.add(target_name)
        value = self._translate_expr(node.value)
        self._emit(f"float {target_name} = {value};")

    def visit_Return(self, node):
        if not isinstance(node.value, ast.Tuple) or len(node.value.elts) != 2:
            raise NotImplementedError(
                "Return must be a 2-tuple (energy, force_magnitude)"
            )
        energy_expr = self._translate_expr(node.value.elts[0])
        force_expr = self._translate_expr(node.value.elts[1])
        self._emit(f"float _result_energy = {energy_expr};")
        self._emit(f"float _result_force = {force_expr};")

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
            raise NotImplementedError(
                f"Unsupported AST node type: {type(node).__name__}"
            )

    def _translate_constant(self, node):
        if isinstance(node.value, float):
            s = repr(node.value)
            if "." not in s and "e" not in s and "E" not in s:
                s += ".0"
            return s + "f"
        elif isinstance(node.value, int):
            return repr(float(node.value)) + "f"
        return repr(node.value)

    def _translate_binop(self, node):
        if isinstance(node.op, ast.Pow):
            return self._translate_pow(node)
        left = self._translate_expr(node.left)
        right = self._translate_expr(node.right)
        op_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
        }
        op_type = type(node.op)
        if op_type in op_map:
            return f"({left} {op_map[op_type]} {right})"
        raise NotImplementedError(f"Unsupported binary op: {op_type.__name__}")

    def _translate_unaryop(self, node):
        operand = self._translate_expr(node.operand)
        if isinstance(node.op, ast.USub):
            return f"(-{operand})"
        elif isinstance(node.op, ast.UAdd):
            return f"(+{operand})"
        raise NotImplementedError(f"Unsupported unary op: {type(node.op).__name__}")

    def _translate_subscript(self, node):
        if not isinstance(node.value, ast.Name):
            raise NotImplementedError("Only simple name subscripts supported")
        param_name = node.value.id
        if param_name not in self.parameter_names:
            raise ValueError(f"{param_name} is not a declared Parameter")
        if not isinstance(node.slice, ast.Name):
            raise NotImplementedError("Parameter index must be a variable name")
        index_name = node.slice.id
        if index_name == self.index_names[0]:
            return f"{param_name}_i"
        elif index_name == self.index_names[1]:
            return f"{param_name}_j"
        raise ValueError(
            f"Index variable {index_name} is not a recognized particle index"
        )

    def _translate_call(self, node):
        if not isinstance(node.func, ast.Name):
            raise NotImplementedError("Only simple function calls supported")
        func_name = node.func.id
        if func_name in _MATH_FUNCTIONS:
            if len(node.args) != 1:
                raise NotImplementedError(f"{func_name} expects exactly 1 argument")
            arg = self._translate_expr(node.args[0])
            return f"{_MATH_FUNCTIONS[func_name]}({arg})"
        raise NotImplementedError(f"Unsupported function: {func_name}")

    def _translate_pow(self, node):
        base = self._translate_expr(node.left)
        if not isinstance(node.right, ast.Constant) or not isinstance(
            node.right.value, int
        ):
            raise NotImplementedError("Only integer constant powers are supported")
        exponent = node.right.value
        if exponent < 0:
            raise NotImplementedError("Negative powers not supported")
        if exponent == 0:
            return "1.0f"
        if exponent == 1:
            return base
        if exponent == 2:
            return f"({base} * {base})"
        if exponent == 3:
            return f"({base} * {base} * {base})"
        if exponent == 6:
            temp = f"_pow6_{abs(hash(node)) % 10000}"
            self.local_variables.add(temp)
            self._emit(f"float {temp} = ({base} * {base} * {base});")
            return f"({temp} * {temp})"
        if exponent == 12:
            temp = f"_pow12_{abs(hash(node)) % 10000}"
            half = self._translate_pow_node_6(base, node)
            self._emit(f"float {temp} = {half};")
            return f"({temp} * {temp})"
        return self._inline_power_chain(base, exponent)

    def _translate_pow_node_6(self, base, node):
        temp = f"_pow6_{abs(hash(node)) % 10000}"
        self.local_variables.add(temp)
        self._emit(f"float {temp} = ({base} * {base} * {base});")
        return f"({temp} * {temp})"

    def _inline_power_chain(self, base, exponent):
        parts = [base] * exponent
        return "(" + " * ".join(parts) + ")"


class NonbondedExpression:
    def __init__(
        self,
        func,
        source,
        ast_tree,
        index_names,
        parameter_names,
        distance_name,
        cuda_fragment,
        local_variables,
    ):
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
        merged_params = list(
            dict.fromkeys(self.parameter_names + other.parameter_names)
        )
        suffix = "_2"
        other_locals = set()
        for var in other.local_variables:
            if var in self.local_variables or var in set(self.parameter_names):
                other_locals.add(var + suffix)
            else:
                other_locals.add(var)

        renamed_fragment_1 = _rename_output_vars(
            _rename_locals_in_cuda(
                self.cuda_fragment, {"force_magnitude"}, {"force_magnitude"}, "_1"
            ),
            "_1",
        )
        renamed_fragment_2 = _rename_output_vars(
            _rename_locals_in_cuda(
                other.cuda_fragment,
                other.local_variables,
                self.local_variables
                | set(self.parameter_names)
                | {"force_magnitude", "energy_val"},
                suffix,
            ),
            suffix,
        )
        combined_fragment = (
            renamed_fragment_1
            + "\n"
            + renamed_fragment_2
            + "\n"
            + "float energy_val = _result_energy_1 + _result_energy"
            + suffix
            + ";\n"
            + "float force_magnitude = _result_force_1 + _result_force"
            + suffix
            + ";"
        )
        combined_locals = self.local_variables | other_locals
        return NonbondedExpression(
            func=None,
            source=self.source + "\n--- combined ---\n" + other.source,
            ast_tree=None,
            index_names=self.index_names,
            parameter_names=merged_params,
            distance_name=self.distance_name,
            cuda_fragment=combined_fragment,
            local_variables=combined_locals,
        )

    def assemble_tile_kernel(self):
        fragment = self.cuda_fragment
        if "_result_energy_1" not in fragment:
            fragment += "\nfloat energy_val = _result_energy;"
            fragment += "\nfloat force_magnitude = _result_force;"
        return _assemble_exclusion_tile_kernel(self.parameter_names, fragment)

    def assemble_main_tile_kernel(self):
        fragment = self.cuda_fragment
        if "_result_energy_1" not in fragment:
            fragment += "\nfloat energy_val = _result_energy;"
            fragment += "\nfloat force_magnitude = _result_force;"
        return _assemble_main_tile_kernel(self.parameter_names, fragment)


def _rename_output_vars(cuda_fragment, tag):
    lines = cuda_fragment.split("\n")
    result = []
    for line in lines:
        new_line = line
        stripped = line.strip()
        if stripped.startswith("float _result_energy ="):
            new_line = line.replace(
                "float _result_energy =", f"float _result_energy{tag} =", 1
            )
        if stripped.startswith("float _result_force ="):
            new_line = line.replace(
                "float _result_force =", f"float _result_force{tag} =", 1
            )
        result.append(new_line)
    return "\n".join(result)


def _rename_locals_in_cuda(cuda_fragment, local_variables, conflict_set, suffix):
    lines = cuda_fragment.split("\n")
    result = []
    for line in lines:
        new_line = line
        for var in sorted(local_variables, key=len, reverse=True):
            if var in conflict_set:
                new_line = re.sub(
                    r"\b" + re.escape(var) + r"\b", var + suffix, new_line
                )
        result.append(new_line)
    return "\n".join(result)


def _is_parameter_call(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Parameter"
    )


def _generate_parameter_declarations_exclusion(parameter_names):
    decls = ""
    for arr in _unique_gpu_arrays(parameter_names):
        if arr in {v[0] for v in _PACKED_PARAMS.values()}:
            decls += f",\n    const float2* __restrict__ {arr}"
            decls += f",\n    const float2* __restrict__ {arr}_14"
        else:
            decls += f",\n    const float* __restrict__ {arr}"
            decls += f",\n    const float* __restrict__ {arr}_14"
    return decls


def _generate_sorted_parameter_declarations_exclusion(parameter_names):
    decls = ""
    for arr in _unique_gpu_arrays(parameter_names):
        if arr in {v[0] for v in _PACKED_PARAMS.values()}:
            decls += f",\n    const float2* __restrict__ sorted_{arr}"
            decls += f",\n    const float2* __restrict__ sorted_{arr}_14"
        else:
            decls += f",\n    const float* __restrict__ sorted_{arr}"
            decls += f",\n    const float* __restrict__ sorted_{arr}_14"
    return decls


def _generate_sorted_parameter_load_i_exclusion(parameter_names):
    lines = []
    loaded = set()
    for name in parameter_names:
        if name in _PACKED_PARAMS:
            arr, comp = _PACKED_PARAMS[name]
            if arr not in loaded:
                lines.append(f"float2 {arr}_i_v = sorted_{arr}[block_x * 32 + tgx];")
                lines.append(
                    f"float2 {arr}_i_14_v = sorted_{arr}_14[block_x * 32 + tgx];"
                )
                loaded.add(arr)
            lines.append(f"float {name}_i = {arr}_i_v.{comp};")
            lines.append(f"float {name}_i_14 = {arr}_i_14_v.{comp};")
        else:
            lines.append(f"float {name}_i = sorted_{name}[block_x * 32 + tgx];")
            lines.append(f"float {name}_i_14 = sorted_{name}_14[block_x * 32 + tgx];")
    return "\n        ".join(lines)


def _generate_parameter_load_j_tile_exclusion(parameter_names):
    lines = []
    loaded_j = set()
    for name in parameter_names:
        if name in _PACKED_PARAMS:
            arr, comp = _PACKED_PARAMS[name]
            lines.append(f"float {name}_j = 0.0f;")
            lines.append(f"float {name}_j_14 = 0.0f;")
        else:
            lines.append(f"float {name}_j = 0.0f, {name}_j_14 = 0.0f;")
            lines.append(
                f"if (gj >= 0 && gj < num_particles) {{ {name}_j = {name}[gj]; {name}_j_14 = {name}_14[gj]; }}"
            )
    for name in parameter_names:
        if name in _PACKED_PARAMS:
            arr, comp = _PACKED_PARAMS[name]
            if arr not in loaded_j:
                lines.append(
                    f"if (gj >= 0 && gj < num_particles) {{ float2 _{arr}_j_v = {arr}[gj]; float2 _{arr}_j_14_v = {arr}_14[gj];"
                )
                loaded_j.add(arr)
    if loaded_j:
        for name in parameter_names:
            if name in _PACKED_PARAMS:
                arr, comp = _PACKED_PARAMS[name]
                lines.append(f"{name}_j = _{arr}_j_v.{comp};")
                lines.append(f"{name}_j_14 = _{arr}_j_14_v.{comp};")
        lines.append("}")
    return "\n        ".join(lines)


def _generate_shuffle_warp_data_exclusion(parameter_names):
    lines = [
        "shfl_px = __shfl_sync(0xffffffff, shfl_px, (tgx + 1) & 31);",
        "shfl_py = __shfl_sync(0xffffffff, shfl_py, (tgx + 1) & 31);",
        "shfl_pz = __shfl_sync(0xffffffff, shfl_pz, (tgx + 1) & 31);",
        "shfl_fx = __shfl_sync(0xffffffff, shfl_fx, (tgx + 1) & 31);",
        "shfl_fy = __shfl_sync(0xffffffff, shfl_fy, (tgx + 1) & 31);",
        "shfl_fz = __shfl_sync(0xffffffff, shfl_fz, (tgx + 1) & 31);",
    ]
    for name in parameter_names:
        lines.append(f"{name}_j = __shfl_sync(0xffffffff, {name}_j, (tgx + 1) & 31);")
        lines.append(
            f"{name}_j_14 = __shfl_sync(0xffffffff, {name}_j_14, (tgx + 1) & 31);"
        )
    return "\n        ".join(lines)


def _generate_parameter_select_exclusion(parameter_names):
    lines = []
    for name in parameter_names:
        lines.append(
            f"float {name}_i_saved = {name}_i; if (is_14) {name}_i = {name}_i_14;"
        )
        lines.append(
            f"float {name}_j_saved = {name}_j; if (is_14) {name}_j = {name}_j_14;"
        )
    return "\n            ".join(lines)


def _generate_parameter_restore_exclusion(parameter_names):
    lines = []
    for name in parameter_names:
        lines.append(f"if (is_14) {name}_i = {name}_i_saved;")
        lines.append(f"if (is_14) {name}_j = {name}_j_saved;")
    return "\n            ".join(lines)


def _generate_parameter_declarations_main_posq(parameter_names):
    decls = ""
    for arr in _unique_gpu_arrays(parameter_names):
        if arr == "charge":
            pass
        else:
            if arr in {v[0] for v in _PACKED_PARAMS.values()}:
                decls += f",\n    const float2* __restrict__ {arr}"
            else:
                decls += f",\n    const float* __restrict__ {arr}"
    return decls


def _generate_sorted_parameter_declarations_main_posq(parameter_names):
    decls = ""
    for arr in _unique_gpu_arrays(parameter_names):
        if arr == "charge":
            pass
        else:
            if arr in {v[0] for v in _PACKED_PARAMS.values()}:
                decls += f",\n    const float2* __restrict__ sorted_{arr}"
            else:
                decls += f",\n    const float* __restrict__ sorted_{arr}"
    return decls


def _generate_sorted_parameter_load_i_main_posq(parameter_names):
    lines = []
    loaded = set()
    for name in parameter_names:
        if name == "charge":
            lines.append("float charge_i = posq_i.w;")
        elif name in _PACKED_PARAMS:
            arr, comp = _PACKED_PARAMS[name]
            if arr not in loaded:
                lines.append(f"float2 {arr}_i_v = sorted_{arr}[block_x * 32 + tgx];")
                loaded.add(arr)
            lines.append(f"float {name}_i = {arr}_i_v.{comp};")
        else:
            lines.append(f"float {name}_i = sorted_{name}[block_x * 32 + tgx];")
    return "\n        ".join(lines)


def _generate_parameter_load_j_tile_main_posq(parameter_names):
    lines = []
    loaded_j = set()
    for name in parameter_names:
        if name == "charge":
            lines.append("float charge_j = _charge_j_posq;")
        elif name in _PACKED_PARAMS:
            lines.append(f"float {name}_j = 0.0f;")
        else:
            lines.append(f"float {name}_j = 0.0f;")
            lines.append(
                f"if (gj >= 0 && gj < num_particles) {{ {name}_j = {name}[gj]; }}"
            )
    for name in parameter_names:
        if name in _PACKED_PARAMS:
            arr, comp = _PACKED_PARAMS[name]
            if arr not in loaded_j:
                lines.append(
                    f"if (gj >= 0 && gj < num_particles) {{ float2 _{arr}_j_v = {arr}[gj];"
                )
                loaded_j.add(arr)
    if loaded_j:
        for name in parameter_names:
            if name in _PACKED_PARAMS:
                arr, comp = _PACKED_PARAMS[name]
                lines.append(f"{name}_j = _{arr}_j_v.{comp};")
        lines.append("}")
    return "\n        ".join(lines)


def _generate_shuffle_warp_data_main(parameter_names):
    lines = [
        "shfl_px = __shfl_sync(0xffffffff, shfl_px, (tgx + 1) & 31);",
        "shfl_py = __shfl_sync(0xffffffff, shfl_py, (tgx + 1) & 31);",
        "shfl_pz = __shfl_sync(0xffffffff, shfl_pz, (tgx + 1) & 31);",
        "shfl_fx = __shfl_sync(0xffffffff, shfl_fx, (tgx + 1) & 31);",
        "shfl_fy = __shfl_sync(0xffffffff, shfl_fy, (tgx + 1) & 31);",
        "shfl_fz = __shfl_sync(0xffffffff, shfl_fz, (tgx + 1) & 31);",
    ]
    for name in parameter_names:
        if name == "charge":
            lines.append(
                "charge_j = __shfl_sync(0xffffffff, charge_j, (tgx + 1) & 31);"
            )
        else:
            lines.append(
                f"{name}_j = __shfl_sync(0xffffffff, {name}_j, (tgx + 1) & 31);"
            )
    return "\n        ".join(lines)


def _generate_parameter_declarations_main(parameter_names):
    decls = ""
    for arr in _unique_gpu_arrays(parameter_names):
        if arr in {v[0] for v in _PACKED_PARAMS.values()}:
            decls += f",\n    const float2* __restrict__ {arr}"
        else:
            decls += f",\n    const float* __restrict__ {arr}"
    return decls


def _generate_sorted_parameter_declarations_main(parameter_names):
    decls = ""
    for arr in _unique_gpu_arrays(parameter_names):
        if arr in {v[0] for v in _PACKED_PARAMS.values()}:
            decls += f",\n    const float2* __restrict__ sorted_{arr}"
        else:
            decls += f",\n    const float* __restrict__ sorted_{arr}"
    return decls


def _generate_sorted_parameter_load_i_main(parameter_names):
    lines = []
    loaded = set()
    for name in parameter_names:
        if name in _PACKED_PARAMS:
            arr, comp = _PACKED_PARAMS[name]
            if arr not in loaded:
                lines.append(f"float2 {arr}_i_v = sorted_{arr}[block_x * 32 + tgx];")
                loaded.add(arr)
            lines.append(f"float {name}_i = {arr}_i_v.{comp};")
        else:
            lines.append(f"float {name}_i = sorted_{name}[block_x * 32 + tgx];")
    return "\n        ".join(lines)


def _generate_parameter_load_j_tile_main(parameter_names):
    lines = []
    loaded_j = set()
    for name in parameter_names:
        if name in _PACKED_PARAMS:
            lines.append(f"float {name}_j = 0.0f;")
        else:
            lines.append(f"float {name}_j = 0.0f;")
            lines.append(
                f"if (gj >= 0 && gj < num_particles) {{ {name}_j = {name}[gj]; }}"
            )
    for name in parameter_names:
        if name in _PACKED_PARAMS:
            arr, comp = _PACKED_PARAMS[name]
            if arr not in loaded_j:
                lines.append(
                    f"if (gj >= 0 && gj < num_particles) {{ float2 _{arr}_j_v = {arr}[gj];"
                )
                loaded_j.add(arr)
    if loaded_j:
        for name in parameter_names:
            if name in _PACKED_PARAMS:
                arr, comp = _PACKED_PARAMS[name]
                lines.append(f"{name}_j = _{arr}_j_v.{comp};")
        lines.append("}")
    return "\n        ".join(lines)


def _generate_parameter_declarations_exclusion_posq(parameter_names):
    decls = ""
    for arr in _unique_gpu_arrays(parameter_names):
        if arr == "charge":
            decls += ",\n    const float* __restrict__ charge_14"
        else:
            if arr in {v[0] for v in _PACKED_PARAMS.values()}:
                decls += f",\n    const float2* __restrict__ {arr}"
                decls += f",\n    const float2* __restrict__ {arr}_14"
            else:
                decls += f",\n    const float* __restrict__ {arr}"
                decls += f",\n    const float* __restrict__ {arr}_14"
    return decls


def _generate_sorted_parameter_declarations_exclusion_posq(parameter_names):
    decls = ""
    for arr in _unique_gpu_arrays(parameter_names):
        if arr == "charge":
            decls += ",\n    const float* __restrict__ sorted_charge_14"
        else:
            if arr in {v[0] for v in _PACKED_PARAMS.values()}:
                decls += f",\n    const float2* __restrict__ sorted_{arr}"
                decls += f",\n    const float2* __restrict__ sorted_{arr}_14"
            else:
                decls += f",\n    const float* __restrict__ sorted_{arr}"
                decls += f",\n    const float* __restrict__ sorted_{arr}_14"
    return decls


def _generate_sorted_parameter_load_i_exclusion_posq(parameter_names):
    lines = []
    loaded = set()
    for name in parameter_names:
        if name == "charge":
            lines.append("float charge_i = posq_i.w;")
            lines.append("float charge_i_14 = sorted_charge_14[block_x * 32 + tgx];")
        elif name in _PACKED_PARAMS:
            arr, comp = _PACKED_PARAMS[name]
            if arr not in loaded:
                lines.append(f"float2 {arr}_i_v = sorted_{arr}[block_x * 32 + tgx];")
                lines.append(
                    f"float2 {arr}_i_14_v = sorted_{arr}_14[block_x * 32 + tgx];"
                )
                loaded.add(arr)
            lines.append(f"float {name}_i = {arr}_i_v.{comp};")
            lines.append(f"float {name}_i_14 = {arr}_i_14_v.{comp};")
        else:
            lines.append(f"float {name}_i = sorted_{name}[block_x * 32 + tgx];")
            lines.append(f"float {name}_i_14 = sorted_{name}_14[block_x * 32 + tgx];")
    return "\n        ".join(lines)


def _generate_parameter_load_j_tile_exclusion_posq(parameter_names):
    lines = []
    loaded_j = set()
    for name in parameter_names:
        if name == "charge":
            lines.append("float charge_j = _charge_j_posq;")
            lines.append("float charge_j_14 = 0.0f;")
        elif name in _PACKED_PARAMS:
            lines.append(f"float {name}_j = 0.0f;")
            lines.append(f"float {name}_j_14 = 0.0f;")
        else:
            lines.append(f"float {name}_j = 0.0f, {name}_j_14 = 0.0f;")
            lines.append(
                f"if (gj >= 0 && gj < num_particles) {{ {name}_j = {name}[gj]; {name}_j_14 = {name}_14[gj]; }}"
            )
    for name in parameter_names:
        if name in _PACKED_PARAMS:
            arr, comp = _PACKED_PARAMS[name]
            if arr not in loaded_j:
                lines.append(
                    f"if (gj >= 0 && gj < num_particles) {{ float2 _{arr}_j_v = {arr}[gj]; float2 _{arr}_j_14_v = {arr}_14[gj];"
                )
                loaded_j.add(arr)
    if loaded_j:
        for name in parameter_names:
            if name in _PACKED_PARAMS:
                arr, comp = _PACKED_PARAMS[name]
                lines.append(f"{name}_j = _{arr}_j_v.{comp};")
                lines.append(f"{name}_j_14 = _{arr}_j_14_v.{comp};")
        lines.append("}")
    if "charge" in parameter_names:
        lines.append(
            "if (gj >= 0 && gj < num_particles) { charge_j_14 = charge_14[gj]; }"
        )
    return "\n        ".join(lines)


def _assemble_exclusion_tile_kernel(parameter_names, expression_fragment):
    use_posq = "charge" in parameter_names

    if use_posq:
        param_decls = _generate_parameter_declarations_exclusion_posq(parameter_names)
        sorted_param_decls = _generate_sorted_parameter_declarations_exclusion_posq(
            parameter_names
        )
        sorted_param_load_i = _generate_sorted_parameter_load_i_exclusion_posq(
            parameter_names
        )
        param_load_j = _generate_parameter_load_j_tile_exclusion_posq(parameter_names)
        pos_args_decl = (
            "    const float4* __restrict__ sorted_posq,\n"
            "    const float4* __restrict__ posq,"
        )
        i_pos_load = (
            "        float4 posq_i = sorted_posq[block_x * 32 + tgx];\n"
            "        float px_i = posq_i.x;\n"
            "        float py_i = posq_i.y;\n"
            "        float pz_i = posq_i.z;"
        )
        j_pos_load = (
            "        float shfl_px = 0.0f, shfl_py = 0.0f, shfl_pz = 0.0f, _charge_j_posq = 0.0f;\n"
            "        if (gj >= 0 && gj < num_particles) {\n"
            "            float4 _pj = posq[gj];\n"
            "            shfl_px = _pj.x;\n"
            "            shfl_py = _pj.y;\n"
            "            shfl_pz = _pj.z;\n"
            "            _charge_j_posq = _pj.w;\n"
            "        }"
        )
    else:
        param_decls = _generate_parameter_declarations_exclusion(parameter_names)
        sorted_param_decls = _generate_sorted_parameter_declarations_exclusion(
            parameter_names
        )
        sorted_param_load_i = _generate_sorted_parameter_load_i_exclusion(
            parameter_names
        )
        param_load_j = _generate_parameter_load_j_tile_exclusion(parameter_names)
        pos_args_decl = (
            "    const float* __restrict__ sorted_pos_x,\n"
            "    const float* __restrict__ sorted_pos_y,\n"
            "    const float* __restrict__ sorted_pos_z,\n"
            "    const float* __restrict__ pos_x,\n"
            "    const float* __restrict__ pos_y,\n"
            "    const float* __restrict__ pos_z,"
        )
        i_pos_load = (
            "        float px_i = sorted_pos_x[block_x * 32 + tgx];\n"
            "        float py_i = sorted_pos_y[block_x * 32 + tgx];\n"
            "        float pz_i = sorted_pos_z[block_x * 32 + tgx];"
        )
        j_pos_load = (
            "        float shfl_px = 0.0f, shfl_py = 0.0f, shfl_pz = 0.0f;\n"
            "        if (gj >= 0 && gj < num_particles) {\n"
            "            shfl_px = pos_x[gj];\n"
            "            shfl_py = pos_y[gj];\n"
            "            shfl_pz = pos_z[gj];\n"
            "        }"
        )

    shuffle_code = _generate_shuffle_warp_data_exclusion(parameter_names)
    param_select = _generate_parameter_select_exclusion(parameter_names)
    param_restore = _generate_parameter_restore_exclusion(parameter_names)

    kernel = f"""extern "C" __global__
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
}}"""
    return kernel


def _assemble_main_tile_kernel(parameter_names, expression_fragment):
    use_posq = "charge" in parameter_names

    if use_posq:
        param_decls = _generate_parameter_declarations_main_posq(parameter_names)
        sorted_param_decls = _generate_sorted_parameter_declarations_main_posq(
            parameter_names
        )
        sorted_param_load_i = _generate_sorted_parameter_load_i_main_posq(
            parameter_names
        )
        param_load_j = _generate_parameter_load_j_tile_main_posq(parameter_names)
        pos_args_decl = (
            "    const float4* __restrict__ sorted_posq,\n"
            "    const float4* __restrict__ posq,"
        )
        i_pos_load = (
            "        float4 posq_i = sorted_posq[block_x * 32 + tgx];\n"
            "        float px_i = posq_i.x;\n"
            "        float py_i = posq_i.y;\n"
            "        float pz_i = posq_i.z;"
        )
        j_pos_load = (
            "        float shfl_px = 0.0f, shfl_py = 0.0f, shfl_pz = 0.0f, _charge_j_posq = 0.0f;\n"
            "        if (gj >= 0 && gj < num_particles) {\n"
            "            float4 _pj = posq[gj];\n"
            "            shfl_px = _pj.x;\n"
            "            shfl_py = _pj.y;\n"
            "            shfl_pz = _pj.z;\n"
            "            _charge_j_posq = _pj.w;\n"
            "        }"
        )
    else:
        param_decls = _generate_parameter_declarations_main(parameter_names)
        sorted_param_decls = _generate_sorted_parameter_declarations_main(
            parameter_names
        )
        sorted_param_load_i = _generate_sorted_parameter_load_i_main(parameter_names)
        param_load_j = _generate_parameter_load_j_tile_main(parameter_names)
        pos_args_decl = (
            "    const float* __restrict__ sorted_pos_x,\n"
            "    const float* __restrict__ sorted_pos_y,\n"
            "    const float* __restrict__ sorted_pos_z,\n"
            "    const float* __restrict__ pos_x,\n"
            "    const float* __restrict__ pos_y,\n"
            "    const float* __restrict__ pos_z,"
        )
        i_pos_load = (
            "        float px_i = sorted_pos_x[block_x * 32 + tgx];\n"
            "        float py_i = sorted_pos_y[block_x * 32 + tgx];\n"
            "        float pz_i = sorted_pos_z[block_x * 32 + tgx];"
        )
        j_pos_load = (
            "        float shfl_px = 0.0f, shfl_py = 0.0f, shfl_pz = 0.0f;\n"
            "        if (gj >= 0 && gj < num_particles) {\n"
            "            shfl_px = pos_x[gj];\n"
            "            shfl_py = pos_y[gj];\n"
            "            shfl_pz = pos_z[gj];\n"
            "        }"
        )

    shuffle_code = _generate_shuffle_warp_data_main(parameter_names)

    kernel = f"""extern "C" __global__
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
}}"""
    return kernel


def nonbonded_expression(func):
    try:
        source = inspect.getsource(func)
    except OSError as exc:
        raise OSError(
            f"Cannot read source for {func.__name__}. "
            "The @nonbonded_expression decorator requires access to the "
            "function source code. Make sure the function is defined in a "
            ".py file (not in an interactive session)."
        ) from exc
    source = textwrap.dedent(source)
    tree = ast.parse(source)

    func_def = tree.body[0]
    if not isinstance(func_def, ast.FunctionDef):
        raise TypeError("Decorator must be applied to a function definition")

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

        if arg_name == "r":
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
            f"Expected exactly 2 particle index arguments, got {len(index_names)}: {index_names}"
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


_GATHER_SORTED_KERNEL_SRC = r"""
extern "C" __global__
void gather_sorted_kernel(
    const float* __restrict__ src,
    const int* __restrict__ block_atoms,
    int total_slots,
    int num_particles,
    float* __restrict__ dst
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_slots) return;
    int atom_id = block_atoms[idx];
    float val = 0.0f;
    if (atom_id >= 0 && atom_id < num_particles) {
        val = src[atom_id];
    }
    dst[idx] = val;
}
"""

_GATHER_SORTED_2COMP_KERNEL_SRC = r"""
extern "C" __global__
void gather_sorted_kernel_2comp(
    const float* __restrict__ src,
    const int* __restrict__ block_atoms,
    int total_slots,
    int num_particles,
    float* __restrict__ dst
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_slots) return;
    int atom_id = block_atoms[idx];
    if (atom_id >= 0 && atom_id < num_particles) {
        dst[idx * 2 + 0] = src[atom_id * 2 + 0];
        dst[idx * 2 + 1] = src[atom_id * 2 + 1];
    } else {
        dst[idx * 2 + 0] = 0.0f;
        dst[idx * 2 + 1] = 0.0f;
    }
}
"""


class NonbondedForce(ForceTerm):
    name = "nonbonded"

    def __init__(self, expression):
        self.expression = expression
        self._kernel = None
        self._exclusion_kernel = None
        self._kernel_source = None
        self._exclusion_kernel_source = None
        self._d_parameter_arrays = {}
        self._parameter_arrays = {}
        self._d_cached_params = {}
        self._parameter_table = None
        self._cutoff = None
        self._cutoff_sq = None
        self._d_posq = None
        self._d_sorted_posq = None
        self._pack_sorted_posq_kernel = None
        self._gather_kernels = None
        self._d_sorted_params = {}
        self._d_sorted_pos_x = None
        self._d_sorted_pos_y = None
        self._d_sorted_pos_z = None

    def bind(self, topology, parameter_table, cutoff):
        self._cutoff = cutoff
        self._cutoff_sq = cutoff * cutoff
        self._num_sm = cp.cuda.runtime.getDeviceProperties(0)["multiProcessorCount"]
        self._parameter_table = parameter_table
        self._rebuild_parameter_arrays(topology.particle_types)
        self._upload_parameter_arrays()
        self._d_cached_params = dict(self._d_parameter_arrays)
        self._kernel_source = self.expression.assemble_main_tile_kernel()
        self._exclusion_kernel_source = self.expression.assemble_tile_kernel()

    def _rebuild_parameter_arrays(self, particle_types):
        _TABLE_PARAM_MAP = {
            "sigma_half": "sigma",
            "sqrt_epsilon": "epsilon",
        }
        pt = self._parameter_table
        for param_name in self.expression.parameter_names:
            table_name = _TABLE_PARAM_MAP.get(param_name, param_name)
            particle = pt.expand_to_particle(table_name, particle_types)
            if param_name == "sigma_half":
                particle = 0.5 * particle
            elif param_name == "sqrt_epsilon":
                particle = np.sqrt(np.maximum(particle, 0.0))
            self._parameter_arrays[param_name] = particle.astype(np.float32)

            name_14 = param_name + "_14"
            table_name_14 = _TABLE_PARAM_MAP.get(param_name, param_name) + "_14"
            has_14 = table_name_14 in pt.type_parameters or table_name_14 in pt.particle_parameters
            if has_14:
                particle_14 = pt.expand_to_particle(table_name_14, particle_types)
                if param_name == "sigma_half":
                    particle_14 = 0.5 * particle_14
                elif param_name == "sqrt_epsilon":
                    particle_14 = np.sqrt(np.maximum(particle_14, 0.0))
            else:
                particle_14 = particle
            self._parameter_arrays[name_14] = particle_14.astype(np.float32)

        if (
            "sigma_half" in self.expression.parameter_names
            and "sqrt_epsilon" in self.expression.parameter_names
        ):
            N = len(particle_types)
            se = np.empty(N * 2, dtype=np.float32)
            se[0::2] = self._parameter_arrays["sigma_half"]
            se[1::2] = self._parameter_arrays["sqrt_epsilon"]
            self._parameter_arrays["sigma_epsilon"] = se

            se_14 = np.empty(N * 2, dtype=np.float32)
            se_14[0::2] = self._parameter_arrays["sigma_half_14"]
            se_14[1::2] = self._parameter_arrays["sqrt_epsilon_14"]
            self._parameter_arrays["sigma_epsilon_14"] = se_14

    def _upload_parameter_arrays(self):
        for name, arr in self._parameter_arrays.items():
            self._d_parameter_arrays[name] = cp.asarray(arr)

    def _ensure_compiled(self):
        if self._kernel is not None:
            return
        self._kernel = cp.RawKernel(self._kernel_source, "main_tile_kernel")
        self._exclusion_kernel = cp.RawKernel(
            self._exclusion_kernel_source.replace(
                "void tile_kernel(", "void exclusion_tile_kernel("
            ),
            "exclusion_tile_kernel",
        )
        if self._use_posq():
            self._pack_sorted_posq_kernel = cp.RawKernel(
                _PACK_SORTED_POSQ_KERNEL, "pack_sorted_posq_kernel"
            )
            N = self._parameter_arrays["charge"].shape[0]
            self._d_posq = cp.zeros(N * 4, dtype=np.float32)
        if not self._d_cached_params:
            self._upload_parameter_arrays()
            self._d_cached_params = dict(self._d_parameter_arrays)

    def _use_posq(self):
        return "charge" in self.expression.parameter_names

    def _ensure_gather_kernels(self):
        if self._gather_kernels is not None:
            return
        self._gather_kernels = {
            "gather_sorted": cp.RawKernel(
                _GATHER_SORTED_KERNEL_SRC, "gather_sorted_kernel"
            ),
            "gather_sorted_2comp": cp.RawKernel(
                _GATHER_SORTED_2COMP_KERNEL_SRC, "gather_sorted_kernel_2comp"
            ),
        }

    def _gather_all_params(self, tile_list):
        if tile_list.num_blocks == 0:
            return
        self._ensure_gather_kernels()
        total_slots = tile_list.num_blocks * 32
        tpb = 256
        grid = ((total_slots + tpb - 1) // tpb,)
        for arr_name in _unique_gpu_arrays(self.expression.parameter_names):
            if self._use_posq() and arr_name == "charge":
                self._gather_one_param(
                    arr_name + "_14", tile_list, total_slots, tpb, grid
                )
            else:
                self._gather_one_param(arr_name, tile_list, total_slots, tpb, grid)
                self._gather_one_param(
                    arr_name + "_14", tile_list, total_slots, tpb, grid
                )

    def _gather_one_param(self, param_name, tile_list, total_slots, tpb, grid):
        d_arr = self._d_parameter_arrays[param_name]
        n_elem = d_arr.shape[0]
        N = tile_list.num_particles
        num_components = n_elem // N if N > 0 else 1
        if num_components == 2:
            sorted_arr = cp.empty(total_slots * 2, dtype=np.float32)
            self._gather_kernels["gather_sorted_2comp"](
                grid,
                (tpb,),
                (d_arr, tile_list.d_block_atoms, np.int32(total_slots),
                 np.int32(N), sorted_arr),
            )
        else:
            sorted_arr = cp.empty(total_slots, dtype=np.float32)
            self._gather_kernels["gather_sorted"](
                grid,
                (tpb,),
                (d_arr, tile_list.d_block_atoms, np.int32(total_slots),
                 np.int32(N), sorted_arr),
            )
        self._d_sorted_params[param_name] = sorted_arr

    def bind_sorted(self, topology, tile_list, gpu_context):
        self._ensure_compiled()
        if not self._d_cached_params:
            self._rebuild_parameter_arrays(topology.particle_types)
            self._upload_parameter_arrays()
            self._d_cached_params = dict(self._d_parameter_arrays)

        permutation = tile_list.d_raw_order
        N = tile_list.num_particles

        arrays_float = {}
        arrays_2comp = {}
        for name, src_arr in self._d_cached_params.items():
            n_elem = src_arr.shape[0]
            if n_elem == N * 2:
                arrays_2comp[name] = src_arr
            else:
                arrays_float[name] = src_arr

        gpu_context.permute_to_sorted(
            permutation, arrays_float, arrays_2comp=arrays_2comp
        )

        for name, arr in arrays_float.items():
            self._d_parameter_arrays[name] = arr
        for name, arr in arrays_2comp.items():
            self._d_parameter_arrays[name] = arr

        self._d_cached_params = dict(self._d_parameter_arrays)

        self._gather_all_params(tile_list)
        if self._use_posq():
            N = gpu_context.number_particles
            total_slots = tile_list.num_blocks * 32
            tpb = 256
            grid = ((total_slots + tpb - 1) // tpb,)
            self._d_sorted_posq = cp.zeros(total_slots * 4, dtype=np.float32)
            self._pack_sorted_posq_kernel(
                grid,
                (tpb,),
                (
                    gpu_context.d_wrapped_positions_x,
                    gpu_context.d_wrapped_positions_y,
                    gpu_context.d_wrapped_positions_z,
                    self._d_parameter_arrays["charge"],
                    tile_list.d_block_atoms,
                    np.int32(N),
                    np.int32(total_slots),
                    self._d_posq,
                    self._d_sorted_posq,
                ),
            )

    def _parameter_arguments(self):
        args = []
        for arr in _unique_gpu_arrays(self.expression.parameter_names):
            if self._use_posq() and arr == "charge":
                args.append(self._d_parameter_arrays[arr + "_14"])
            else:
                args.append(self._d_parameter_arrays[arr])
                args.append(self._d_parameter_arrays[arr + "_14"])
        return args

    def _main_parameter_arguments(self):
        args = []
        for arr in _unique_gpu_arrays(self.expression.parameter_names):
            if arr == "charge":
                continue
            if arr in {v[0] for v in _PACKED_PARAMS.values()}:
                args.append(self._d_parameter_arrays[arr])
            else:
                args.append(self._d_parameter_arrays[arr])
        return args

    def _main_sorted_parameter_arguments(self):
        args = []
        for arr in _unique_gpu_arrays(self.expression.parameter_names):
            if arr == "charge":
                continue
            args.append(self._d_sorted_params[arr])
        return args

    def _sorted_parameter_arguments(self):
        args = []
        for arr in _unique_gpu_arrays(self.expression.parameter_names):
            if self._use_posq() and arr == "charge":
                args.append(self._d_sorted_params["charge_14"])
            else:
                args.append(self._d_sorted_params[arr])
                args.append(self._d_sorted_params[f"{arr}_14"])
        return args

    def _refresh_posq(self, gpu_context, tile_list):
        N = gpu_context.number_particles
        total_slots = tile_list.num_blocks * 32
        tpb = 256
        grid = ((total_slots + tpb - 1) // tpb,)
        if self._d_sorted_posq.size != total_slots * 4:
            self._d_sorted_posq = cp.zeros(total_slots * 4, dtype=np.float32)
        self._pack_sorted_posq_kernel(
            grid,
            (tpb,),
            (
                gpu_context.d_wrapped_positions_x,
                gpu_context.d_wrapped_positions_y,
                gpu_context.d_wrapped_positions_z,
                self._d_parameter_arrays["charge"],
                tile_list.d_block_atoms,
                np.int32(N),
                np.int32(total_slots),
                self._d_posq,
                self._d_sorted_posq,
            ),
        )

    def _refresh_sorted_data(self, tile_list, gpu_context):
        if tile_list.num_blocks == 0:
            return
        if self._use_posq():
            self._refresh_posq(gpu_context, tile_list)
        else:
            self._ensure_gather_kernels()
            N = gpu_context.number_particles
            total_slots = tile_list.num_blocks * 32
            tpb = 256
            grid = ((total_slots + tpb - 1) // tpb,)
            for src, attr in [
                (gpu_context.d_wrapped_positions_x, "_d_sorted_pos_x"),
                (gpu_context.d_wrapped_positions_y, "_d_sorted_pos_y"),
                (gpu_context.d_wrapped_positions_z, "_d_sorted_pos_z"),
            ]:
                dst = getattr(self, attr)
                if dst is None or dst.size != total_slots:
                    dst = cp.empty(total_slots, dtype=np.float32)
                    setattr(self, attr, dst)
                self._gather_kernels["gather_sorted"](
                    grid,
                    (tpb,),
                    (src, tile_list.d_block_atoms, np.int32(total_slots),
                     np.int32(N), dst),
                )

    def compute(self, gpu_context, tile_list=None):
        self._ensure_compiled()

        if tile_list is None or tile_list.num_tiles == 0:
            return

        self._refresh_sorted_data(tile_list, gpu_context)

        num_sm = self._num_sm
        grid_size = 16 * num_sm

        if self._use_posq():
            num_main = getattr(tile_list, "num_main_tiles", 0)
            if num_main > 0:
                main_args = (
                    [
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
                    ]
                    + self._main_parameter_arguments()
                    + self._main_sorted_parameter_arguments()
                )
                self._kernel((grid_size,), (256,), tuple(main_args))

            num_excl = getattr(tile_list, "num_exclusion_tiles", 0)
            if num_excl > 0:
                excl_grid_size = max(grid_size, (num_excl + 7) // 8)
                excl_args = (
                    [
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
                    ]
                    + self._parameter_arguments()
                    + self._sorted_parameter_arguments()
                )
                self._exclusion_kernel((excl_grid_size,), (256,), tuple(excl_args))
        else:
            num_main = getattr(tile_list, "num_main_tiles", 0)
            if num_main > 0:
                main_args = (
                    [
                        self._d_sorted_pos_x,
                        self._d_sorted_pos_y,
                        self._d_sorted_pos_z,
                        gpu_context.d_wrapped_positions_x,
                        gpu_context.d_wrapped_positions_y,
                        gpu_context.d_wrapped_positions_z,
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
                    ]
                    + self._main_parameter_arguments()
                    + self._main_sorted_parameter_arguments()
                )
                self._kernel((grid_size,), (256,), tuple(main_args))

            num_excl = getattr(tile_list, "num_exclusion_tiles", 0)
            if num_excl > 0:
                excl_grid_size = max(grid_size, (num_excl + 7) // 8)
                excl_args = (
                    [
                        self._d_sorted_pos_x,
                        self._d_sorted_pos_y,
                        self._d_sorted_pos_z,
                        gpu_context.d_wrapped_positions_x,
                        gpu_context.d_wrapped_positions_y,
                        gpu_context.d_wrapped_positions_z,
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
                    ]
                    + self._parameter_arguments()
                    + self._sorted_parameter_arguments()
                )
                self._exclusion_kernel((excl_grid_size,), (256,), tuple(excl_args))
