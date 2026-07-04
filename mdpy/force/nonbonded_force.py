from __future__ import annotations

import re

import cupy as cp
import numpy as np

from mdpy.force.force_term import ForceTerm

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


def _prepare_energy_expression(energy_cuda):
    if not energy_cuda:
        return "", "0.0f"
    lines = energy_cuda.split("\n")
    result_vars = []
    new_lines = []
    for line in lines:
        m = re.match(r"^(\s*float\s+)(_result_energy(?:_\d+)?)(\s*=.*)$", line)
        if m:
            result_vars.append(m.group(2))
        new_lines.append(line)
    total_expr = " + ".join(result_vars) if result_vars else "0.0f"
    return "\n".join(new_lines), total_expr


def _split_per_particle(per_particle):
    i_props = {}
    j_props = {}
    for arg_name, base_name in per_particle.items():
        if arg_name.endswith("2"):
            j_props[arg_name] = base_name
        else:
            i_props[arg_name] = base_name
    return i_props, j_props


def _unique_prop_bases(per_particle):
    return list(dict.fromkeys(per_particle.values()))


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


def _assemble_exclusion_kernel(
    expr_info, energy_cuda, grad_cuda, dEdr_cuda, total_energy_expr, compute_energy=True
):
    i_props, j_props = _split_per_particle(expr_info.per_particle)
    bases = _unique_prop_bases(expr_info.per_particle)
    non_charge_bases = [b for b in bases if b != "charge"]

    sorted_decls = ""
    for base in non_charge_bases:
        sorted_decls += f",\n    const float* __restrict__ sorted_{base}"

    unsorted_decls = ""
    for base in non_charge_bases:
        unsorted_decls += f",\n    const float* __restrict__ d_{base}"

    pair_decls = ""
    for name in expr_info.params:
        pair_decls += f",\n    const float* __restrict__ d_{name}_matrix"

    scalar_decls = ""
    for name in expr_info.scalars:
        scalar_decls += f",\n    float {name}"

    load_i = ""
    for arg_name, base_name in i_props.items():
        if base_name == "charge":
            load_i += f"\n        float {arg_name} = posq_i.w;"
        else:
            load_i += (
                f"\n        float {arg_name} = sorted_{base_name}[block_x * 32 + tgx];"
            )

    load_j_init = ""
    for arg_name in j_props:
        load_j_init += f"\n        float {arg_name} = 0.0f;"

    load_j_from_array = ""
    for arg_name, base_name in j_props.items():
        if base_name == "charge":
            load_j_from_array += f"\n            {arg_name} = _pj.w;"
        else:
            load_j_from_array += f"\n            {arg_name} = d_{base_name}[gj];"

    shuffle_j = ""
    for arg_name in j_props:
        shuffle_j += f"\n            {arg_name} = __shfl_sync(0xffffffff, {arg_name}, (tgx + 1) & 31);"

    load_pair = ""
    for name in expr_info.params:
        load_pair += f"\n                float {name} = d_{name}_matrix[pair_idx];"

    if compute_energy:
        energy_buffer_arg = "    float* __restrict__ energy_buffer,"
        energy_init = "    float total_energy = 0.0f;"
        energy_accum = "                total_energy += energy_val;"
        energy_reduce = """
    for (int offset = 16; offset > 0; offset >>= 1) {{
        total_energy += __shfl_down_sync(0xffffffff, total_energy, offset);
    }}
    if (tgx == 0) atomicAdd(energy_buffer, total_energy);
"""
    else:
        energy_buffer_arg = ""
        energy_init = ""
        energy_accum = ""
        energy_reduce = ""

    kernel = f"""extern "C" __global__
void exclusion_block_pair_kernel(
    const float4* __restrict__ sorted_posq,
    const float4* __restrict__ posq,
    const float* __restrict__ shift_x,
    const float* __restrict__ shift_y,
    const float* __restrict__ shift_z,
    float* __restrict__ f_x,
    float* __restrict__ f_y,
    float* __restrict__ f_z,
{energy_buffer_arg}
    const int* __restrict__ block_atoms,
    const int* __restrict__ block_pairs,
    const int* __restrict__ interacting_atoms,
    const unsigned int* __restrict__ exclusion_masks,
    float cutoff_sq,
    const int* __restrict__ d_block_pair_count,
    int num_particles
    {sorted_decls}{unsorted_decls}{pair_decls},
    const int* __restrict__ d_types,
    int n_types
    {scalar_decls}
) {{
    __shared__ int s_num_pairs;
    if (threadIdx.x == 0) s_num_pairs = d_block_pair_count[0];
    __syncthreads();
    int num_block_pairs = s_num_pairs;

    int total_warps = (blockDim.x * gridDim.x) / 32;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int tgx = threadIdx.x & 31;
    int tbx = threadIdx.x - tgx;
    int pos = (int)((long long)warp_id * num_block_pairs / total_warps);
    int end = (int)((long long)(warp_id + 1) * num_block_pairs / total_warps);
{energy_init}
    __shared__ int atom_indices_shared[256];
    __shared__ unsigned int excl_shared[256];
    for (; pos < end; pos++) {{
        int block_x = block_pairs[pos];
        int gi = block_atoms[block_x * 32 + tgx];
        float4 posq_i = sorted_posq[block_x * 32 + tgx];
        float px_i = posq_i.x;
        float py_i = posq_i.y;
        float pz_i = posq_i.z;
        float shfl_sx = shift_x[pos * 32 + tgx];
        float shfl_sy = shift_y[pos * 32 + tgx];
        float shfl_sz = shift_z[pos * 32 + tgx];
{load_i}
        int type_i = 0;
        if (gi >= 0 && gi < num_particles) {{
            type_i = __ldg(&d_types[gi]);
        }}
        int gj = interacting_atoms[pos * 32 + tgx];
        float shfl_px = 0.0f, shfl_py = 0.0f, shfl_pz = 0.0f;
{load_j_init}
        if (gj >= 0 && gj < num_particles) {{
            float4 _pj = posq[gj];
            shfl_px = _pj.x;
            shfl_py = _pj.y;
            shfl_pz = _pj.z;
{load_j_from_array}
        }}
        atom_indices_shared[threadIdx.x] = gj;
        excl_shared[threadIdx.x] = exclusion_masks[pos * 32 + tgx];
        float force_x = 0.0f, force_y = 0.0f, force_z = 0.0f;
        float shfl_fx = 0.0f, shfl_fy = 0.0f, shfl_fz = 0.0f;
        int tj = tgx;
        for (int j = 0; j < 32; j++) {{
            unsigned int excl_j = excl_shared[tbx + tj];
            int atom2 = atom_indices_shared[tbx + tj];
            float dx = shfl_px - px_i + shfl_sx;
            float dy = shfl_py - py_i + shfl_sy;
            float dz = shfl_pz - pz_i + shfl_sz;
            float dist_sq = dx * dx + dy * dy + dz * dz;
            bool excluded = (atom2 < 0 || atom2 >= num_particles)
                         || ((excl_j >> tgx) & 1);
            if (!excluded && dist_sq > 1.0e-12f && dist_sq <= cutoff_sq && gi >= 0 && gi < num_particles) {{
                float inv_dist = rsqrtf(dist_sq);
                float r = dist_sq * inv_dist;
                int type_j = __ldg(&d_types[atom2]);
                int pair_idx = type_i * n_types + type_j;
{load_pair}
{energy_cuda}
{grad_cuda if grad_cuda else ''}
                float force_magnitude = ({dEdr_cuda});
                float energy_val = {total_energy_expr};
                float inv_dist_force = force_magnitude * inv_dist;
                float fx = dx * inv_dist_force;
                float fy = dy * inv_dist_force;
                float fz = dz * inv_dist_force;
                force_x += fx; force_y += fy; force_z += fz;
                shfl_fx -= fx; shfl_fy -= fy; shfl_fz -= fz;
{energy_accum}
            }}
            shfl_px = __shfl_sync(0xffffffff, shfl_px, (tgx + 1) & 31);
            shfl_py = __shfl_sync(0xffffffff, shfl_py, (tgx + 1) & 31);
            shfl_pz = __shfl_sync(0xffffffff, shfl_pz, (tgx + 1) & 31);
            shfl_fx = __shfl_sync(0xffffffff, shfl_fx, (tgx + 1) & 31);
            shfl_fy = __shfl_sync(0xffffffff, shfl_fy, (tgx + 1) & 31);
            shfl_fz = __shfl_sync(0xffffffff, shfl_fz, (tgx + 1) & 31);
            shfl_sx = __shfl_sync(0xffffffff, shfl_sx, (tgx + 1) & 31);
            shfl_sy = __shfl_sync(0xffffffff, shfl_sy, (tgx + 1) & 31);
            shfl_sz = __shfl_sync(0xffffffff, shfl_sz, (tgx + 1) & 31);
{shuffle_j}
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
{energy_reduce}
}}"""
    return kernel


class NonbondedForce(ForceTerm):
    name = "nonbonded"

    def __init__(self, expression, cutoff=12.0):
        self._expression = expression
        self._expr_info = expression.expr_info
        self._dEdr_cuda = expression.dEdr_cuda
        self._grad_cuda = getattr(expression, "grad_cuda", None)
        self._energy_cuda_raw = expression.energy_cuda

        self._i_props, self._j_props = _split_per_particle(self._expr_info.per_particle)
        self._prop_bases = _unique_prop_bases(self._expr_info.per_particle)

        self._per_particle_data = {}
        self._pair_param_data = {}
        self._scalar_data = {}

        self._d_per_particle = {}
        self._d_sorted_per_particle = {}
        self._d_pair_params = {}

        self._excl_kernel = None
        self._excl_kernel_fo = None
        self._pack_posq_kernel = None
        self._gather_kernels = None

        self._d_posq = None
        self._d_sorted_posq = None
        self._d_types = None
        self._n_types = 0

        self._cutoff = cutoff
        self._cutoff_sq = cutoff * cutoff
        self._num_sm = None
        self._compiled = False

        self._energy_cuda = None
        self._total_energy_expr = None

    def set_pair_parameter(self, name, matrix):
        self._pair_param_data[name] = np.asarray(matrix, dtype=np.float32)

    def set_scalar(self, name, value):
        self._scalar_data[name] = float(value)

    def _lazy_compile(self, gpu_context):
        if self._compiled:
            return
        self._num_sm = cp.cuda.runtime.getDeviceProperties(0)["multiProcessorCount"]

        if self._pair_param_data:
            first_matrix = next(iter(self._pair_param_data.values()))
            self._n_types = int(np.sqrt(first_matrix.shape[0]))
        else:
            self._n_types = int(cp.max(gpu_context.d_types).get()) + 1

        for name, mat in self._pair_param_data.items():
            self._d_pair_params[name] = cp.asarray(mat)

        self._energy_cuda, self._total_energy_expr = _prepare_energy_expression(
            self._energy_cuda_raw
        )

        excl_src = _assemble_exclusion_kernel(
            self._expr_info,
            self._energy_cuda,
            self._grad_cuda,
            self._dEdr_cuda,
            self._total_energy_expr,
            compute_energy=True,
        )
        excl_src_fo = _assemble_exclusion_kernel(
            self._expr_info,
            self._energy_cuda,
            self._grad_cuda,
            self._dEdr_cuda,
            self._total_energy_expr,
            compute_energy=False,
        )

        self._excl_kernel = cp.RawKernel(excl_src, "exclusion_block_pair_kernel")
        self._excl_kernel_fo = cp.RawKernel(excl_src_fo, "exclusion_block_pair_kernel")
        self._pack_posq_kernel = cp.RawKernel(
            _PACK_SORTED_POSQ_KERNEL, "pack_sorted_posq_kernel"
        )

        N = gpu_context.number_particles
        self._d_posq = cp.zeros(N * 4, dtype=np.float32)

        self._d_types = gpu_context.d_types

        self._compiled = True

    def _ensure_gather_kernels(self):
        if self._gather_kernels is not None:
            return
        self._gather_kernels = {
            "gather_sorted": cp.RawKernel(
                _GATHER_SORTED_KERNEL_SRC, "gather_sorted_kernel"
            ),
        }

    def _resolve_per_particle(self, gpu_context):
        for base_name in self._prop_bases:
            if base_name == "charge":
                self._d_per_particle[base_name] = gpu_context.d_charges

    def _gather_per_particle(self, block_list):
        if block_list.num_blocks == 0:
            return
        non_charge_bases = [b for b in self._prop_bases if b != "charge"]
        if not non_charge_bases:
            return
        self._ensure_gather_kernels()
        total_slots = block_list.num_blocks * 32
        tpb = 256
        grid = ((total_slots + tpb - 1) // tpb,)
        for base_name in non_charge_bases:
            d_arr = self._d_per_particle[base_name]
            sorted_arr = cp.empty(total_slots, dtype=np.float32)
            self._gather_kernels["gather_sorted"](
                grid,
                (tpb,),
                (
                    d_arr,
                    block_list.d_block_atoms,
                    np.int32(total_slots),
                    np.int32(block_list.num_particles),
                    sorted_arr,
                ),
            )
            self._d_sorted_per_particle[base_name] = sorted_arr

    def bind_sorted(self, topology, block_list, gpu_context):
        if not self._compiled:
            self._lazy_compile(gpu_context)
        self._resolve_per_particle(gpu_context)
        permutation = block_list.d_raw_order

        borrowed = {k for k in self._prop_bases if k == "charge"}
        arrays_float = {
            k: v for k, v in self._d_per_particle.items() if k not in borrowed
        }
        if arrays_float:
            gpu_context.permute_to_sorted(permutation, arrays_float)
            for base_name, arr in arrays_float.items():
                self._d_per_particle[base_name] = arr

        if self._expr_info.params:
            arrays_int = {"_types": gpu_context.d_types}
            gpu_context.permute_to_sorted(
                block_list.d_sorted_to_pdb,
                {},
                arrays_int=arrays_int,
            )
            self._d_types = arrays_int["_types"]
        else:
            self._d_types = gpu_context.d_types

        self._gather_per_particle(block_list)

        N = gpu_context.number_particles
        total_slots = block_list.num_blocks * 32
        tpb = 256
        grid = ((total_slots + tpb - 1) // tpb,)
        self._d_sorted_posq = cp.zeros(total_slots * 4, dtype=np.float32)
        d_charges = gpu_context.d_charges
        self._pack_posq_kernel(
            grid,
            (tpb,),
            (
                gpu_context.d_positions_x,
                gpu_context.d_positions_y,
                gpu_context.d_positions_z,
                d_charges,
                block_list.d_block_atoms,
                np.int32(N),
                np.int32(total_slots),
                self._d_posq,
                self._d_sorted_posq,
            ),
        )

    def _refresh_posq(self, gpu_context, block_list):
        N = gpu_context.number_particles
        total_slots = block_list.num_blocks * 32
        tpb = 256
        grid = ((total_slots + tpb - 1) // tpb,)
        if self._d_sorted_posq.size != total_slots * 4:
            self._d_sorted_posq = cp.zeros(total_slots * 4, dtype=np.float32)
        d_charges = gpu_context.d_charges
        self._pack_posq_kernel(
            grid,
            (tpb,),
            (
                gpu_context.d_positions_x,
                gpu_context.d_positions_y,
                gpu_context.d_positions_z,
                d_charges,
                block_list.d_block_atoms,
                np.int32(N),
                np.int32(total_slots),
                self._d_posq,
                self._d_sorted_posq,
            ),
        )

    def _build_excl_args(self, gpu_context, block_list, compute_energy):
        args = [
            self._d_sorted_posq,
            self._d_posq,
            block_list.d_excl_shift_x,
            block_list.d_excl_shift_y,
            block_list.d_excl_shift_z,
            gpu_context.d_forces_x,
            gpu_context.d_forces_y,
            gpu_context.d_forces_z,
        ]
        if compute_energy:
            args.append(gpu_context.d_energy)
        args.extend(
            [
                block_list.d_block_atoms,
                block_list.d_excl_block_pairs,
                block_list.d_excl_interacting_atoms,
                block_list.d_excl_exclusion_masks,
                np.float32(self._cutoff_sq),
                block_list._d_counters,
                np.int32(gpu_context.number_particles),
            ]
        )
        for base in self._prop_bases:
            if base == "charge":
                continue
            args.append(self._d_sorted_per_particle[base])
        for base in self._prop_bases:
            if base == "charge":
                continue
            args.append(self._d_per_particle[base])
        for name in self._expr_info.params:
            args.append(self._d_pair_params[name])
        args.append(self._d_types)
        args.append(np.int32(self._n_types))
        for name in self._expr_info.scalars:
            args.append(np.float32(self._scalar_data.get(name, 0.0)))
        return tuple(args)

    def compute(self, gpu_context, block_list=None, compute_energy=True):
        if not self._compiled:
            self._lazy_compile(gpu_context)

        self._resolve_per_particle(gpu_context)

        if block_list is None:
            return

        self._refresh_posq(gpu_context, block_list)

        num_sm = self._num_sm
        grid_size = 16 * num_sm

        if compute_energy:
            excl_kernel = self._excl_kernel
        else:
            excl_kernel = self._excl_kernel_fo

        excl_args = self._build_excl_args(gpu_context, block_list, compute_energy)
        excl_kernel((grid_size,), (256,), excl_args)
