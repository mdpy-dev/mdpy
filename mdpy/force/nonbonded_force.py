from __future__ import annotations

import re

import cupy as cp
import numpy as np

from mdpy.force.force_term import ForceTerm
from mdpy import precision


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


_ADD_SORTED_FORCES_KERNEL_SRC = r"""
extern "C" __global__
void add_sorted_forces_kernel(
    const float* __restrict__ sorted_fx,
    const float* __restrict__ sorted_fy,
    const float* __restrict__ sorted_fz,
    const int* __restrict__ block_atoms,
    int total_slots,
    float* __restrict__ pdb_fx,
    float* __restrict__ pdb_fy,
    float* __restrict__ pdb_fz
) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= total_slots) return;
    int pdb_id = block_atoms[slot];
    if (pdb_id >= 0) {
        atomicAdd(&pdb_fx[pdb_id], sorted_fx[slot]);
        atomicAdd(&pdb_fy[pdb_id], sorted_fy[slot]);
        atomicAdd(&pdb_fz[pdb_id], sorted_fz[slot]);
    }
}
"""


def _assemble_exclusion_kernel(
    expr_info, energy_cuda, grad_cuda, radial_force_cuda, total_energy_expr, compute_energy=True
):
    i_props, j_props = _split_per_particle(expr_info.per_particle)
    for base in expr_info.per_particle.values():
        if base != "charge":
            raise NotImplementedError(
                f"Non-charge per-particle property '{base}' is not supported. "
                f"See AGENTS.md (_d_sorted_per_particle stub)."
            )

    pair_decls = ""
    for name in expr_info.params:
        pair_decls += f",\n    const float* __restrict__ d_{name}_matrix"

    scalar_decls = ""
    for name in expr_info.scalars:
        scalar_decls += f",\n    float {name}"

    load_i = ""
    for arg_name in i_props:
        load_i += f"\n        float {arg_name} = position_charge_i.w;"

    load_j_init = ""
    for arg_name in j_props:
        load_j_init += f"\n        float {arg_name} = 0.0f;"

    load_j_from_array = ""
    for arg_name in j_props:
        load_j_from_array += f"\n            {arg_name} = jdata.w;"

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
    const float4* __restrict__ sorted_data,
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
    {pair_decls},
    const int* __restrict__ d_sorted_type_indices,
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
        float4 position_charge_i = sorted_data[block_x * 32 + tgx];
        float px_i = position_charge_i.x;
        float py_i = position_charge_i.y;
        float pz_i = position_charge_i.z;
        float sx = shift_x[pos];
        float sy = shift_y[pos];
        float sz = shift_z[pos];
{load_i}
        int type_i = d_sorted_type_indices[block_x * 32 + tgx];
        int j_slot = interacting_atoms[pos * 32 + tgx];
        int gj = (j_slot >= 0) ? block_atoms[j_slot] : -1;
        float shfl_px = 0.0f, shfl_py = 0.0f, shfl_pz = 0.0f;
{load_j_init}
        if (gj >= 0 && gj < num_particles) {{
            float4 jdata = sorted_data[j_slot];
            shfl_px = jdata.x;
            shfl_py = jdata.y;
            shfl_pz = jdata.z;
        {load_j_from_array}
        }}
        atom_indices_shared[threadIdx.x] = j_slot;
        excl_shared[threadIdx.x] = exclusion_masks[pos * 32 + tgx];
        float force_x = 0.0f, force_y = 0.0f, force_z = 0.0f;
        float shfl_fx = 0.0f, shfl_fy = 0.0f, shfl_fz = 0.0f;
        int tj = tgx;
        for (int j = 0; j < 32; j++) {{
            unsigned int excl_j = excl_shared[tbx + tj];
            int slot2 = atom_indices_shared[tbx + tj];
            float dx = shfl_px - px_i + sx;
            float dy = shfl_py - py_i + sy;
            float dz = shfl_pz - pz_i + sz;
            float dist_sq = dx * dx + dy * dy + dz * dz;
            bool excluded = (slot2 < 0)
                         || ((excl_j >> tgx) & 1);
            if (!excluded && dist_sq > 1.0e-12f && dist_sq <= cutoff_sq && gi >= 0 && gi < num_particles) {{
                float inv_dist = rsqrtf(dist_sq);
                float r = dist_sq * inv_dist;
                int type_j = d_sorted_type_indices[slot2];
                int pair_idx = type_i * n_types + type_j;
{load_pair}
{energy_cuda}
{grad_cuda if grad_cuda else ''}
                float force_magnitude = ({radial_force_cuda});
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
{shuffle_j}
            tj = (tj + 1) & 31;
        }}
        if (gi >= 0 && gi < num_particles) {{
            int slot_i = block_x * 32 + tgx;
            atomicAdd(&f_x[slot_i], force_x);
            atomicAdd(&f_y[slot_i], force_y);
            atomicAdd(&f_z[slot_i], force_z);
        }}
        if (j_slot >= 0) {{
            atomicAdd(&f_x[j_slot], shfl_fx);
            atomicAdd(&f_y[j_slot], shfl_fy);
            atomicAdd(&f_z[j_slot], shfl_fz);
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
        self._radial_force_cuda = expression.radial_force_cuda
        self._grad_cuda = getattr(expression, "grad_cuda", None)
        self._energy_cuda_raw = expression.energy_cuda

        self._pair_param_data = {}
        self._scalar_data = {}

        self._d_pair_params = {}

        self._pair_kernel = None
        self._pair_kernel_fo = None

        self._n_types = 0

        self._cutoff = cutoff
        self._cutoff_sq = cutoff * cutoff
        self._num_sm = None
        self._compiled = False

        self._d_sorted_fx = None       # slot-indexed force buffer, 3 separate arrays
        self._d_sorted_fy = None       # matching State's d_forces_x/y/z convention
        self._d_sorted_fz = None
        self._sorted_force_slots = 0   # current allocated size (total_slots)

        self._energy_cuda = None
        self._total_energy_expr = None

    def set_pair_parameter(self, name, matrix):
        self._pair_param_data[name] = np.asarray(matrix, dtype=np.float32)

    def set_scalar(self, name, value):
        self._scalar_data[name] = float(value)

    def _lazy_compile(self, state):
        if self._compiled:
            return
        self._num_sm = cp.cuda.runtime.getDeviceProperties(0)["multiProcessorCount"]

        if self._pair_param_data:
            first_matrix = next(iter(self._pair_param_data.values()))
            self._n_types = int(np.sqrt(first_matrix.shape[0]))
        else:
            self._n_types = int(cp.max(state.d_particle_type_indices).get()) + 1

        for name, mat in self._pair_param_data.items():
            self._d_pair_params[name] = cp.asarray(mat)

        self._energy_cuda, self._total_energy_expr = _prepare_energy_expression(
            self._energy_cuda_raw
        )

        excl_src = _assemble_exclusion_kernel(
            self._expr_info,
            self._energy_cuda,
            self._grad_cuda,
            self._radial_force_cuda,
            self._total_energy_expr,
            compute_energy=True,
        )
        excl_src_fo = _assemble_exclusion_kernel(
            self._expr_info,
            self._energy_cuda,
            self._grad_cuda,
            self._radial_force_cuda,
            self._total_energy_expr,
            compute_energy=False,
        )

        self._pair_kernel = cp.RawKernel(excl_src, "exclusion_block_pair_kernel")
        self._pair_kernel_fo = cp.RawKernel(excl_src_fo, "exclusion_block_pair_kernel")

        self._add_forces_kernel = cp.RawKernel(
            _ADD_SORTED_FORCES_KERNEL_SRC, "add_sorted_forces_kernel"
        )

        self._compiled = True

    def _ensure_sorted_force_buffer(self, block_list):
        """Ensure slot-indexed force buffers are allocated for current block count.

        Uses grow-only allocation: buffers grow when max_blocks increases,
        never shrinks. Reused across steps to avoid per-step allocation.
        """
        total_slots = block_list.max_blocks * 32
        if total_slots == 0:
            return
        if self._d_sorted_fx is None or self._sorted_force_slots < total_slots:
            self._d_sorted_fx = cp.empty(total_slots, dtype=precision.FLOAT)
            self._d_sorted_fy = cp.empty(total_slots, dtype=precision.FLOAT)
            self._d_sorted_fz = cp.empty(total_slots, dtype=precision.FLOAT)
            self._sorted_force_slots = total_slots

    def _zero_sorted_forces(self, total_slots):
        """Zero the slot-indexed force buffers via async memset."""
        if self._d_sorted_fx is None or total_slots == 0:
            return
        nbytes = total_slots * 4  # 4 bytes per float32
        stream_ptr = cp.cuda.Stream.null.ptr
        for buf in (self._d_sorted_fx, self._d_sorted_fy, self._d_sorted_fz):
            cp.cuda.runtime.memsetAsync(buf.data.ptr, 0, nbytes, stream_ptr)

    def _add_sorted_forces(self, state, block_list):
        """Add slot-indexed forces into PDB-order force array via one-pass kernel.

        atomicAdd(&pdb_fx[block_atoms[slot]], sorted_fx[slot]) — accumulates
        the sorted buffer's forces into the shared PDB-order force pool.
        """
        total_slots = block_list.max_blocks * 32
        if total_slots == 0 or self._d_sorted_fx is None:
            return
        threads = 256
        grid = ((total_slots + threads - 1) // threads,)
        self._add_forces_kernel(
            grid, (threads,),
            (
                self._d_sorted_fx, self._d_sorted_fy, self._d_sorted_fz,
                block_list.d_block_atoms,
                np.int32(total_slots),
                state.d_forces_x,
                state.d_forces_y,
                state.d_forces_z,
            ),
        )

    def compute(self, state, block_list=None, compute_energy=True, compute_virial=False):
        if not self._compiled:
            self._lazy_compile(state)

        if block_list is None:
            return

        total_slots = block_list.max_blocks * 32
        if total_slots == 0:
            return

        # Ensure slot-indexed force buffers exist and are zeroed
        self._ensure_sorted_force_buffer(block_list)
        self._zero_sorted_forces(total_slots)

        num_sm = self._num_sm
        grid_size = 16 * num_sm

        if compute_energy:
            pair_kernel = self._pair_kernel
        else:
            pair_kernel = self._pair_kernel_fo

        # Assemble kernel args inline (replaces deleted _build_excl_args)
        args = [
            block_list.d_sorted_posq,
            block_list.d_block_pair_shift_x,
            block_list.d_block_pair_shift_y,
            block_list.d_block_pair_shift_z,
            self._d_sorted_fx,
            self._d_sorted_fy,
            self._d_sorted_fz,
        ]
        if compute_energy:
            args.append(state.d_energy)
        args.extend(
            [
                block_list.d_block_atoms,
                block_list.d_block_pairs,
                block_list.d_interacting_atoms,
                block_list.d_exclusion_masks,
                np.float32(self._cutoff_sq),
                block_list.d_num_block_pairs,
                np.int32(state.num_particles),
            ]
        )
        for name in self._expr_info.params:
            args.append(self._d_pair_params[name])
        args.append(block_list.d_sorted_type_indices)
        args.append(np.int32(self._n_types))
        for name in self._expr_info.scalars:
            args.append(np.float32(self._scalar_data.get(name, 0.0)))

        pair_kernel((grid_size,), (256,), tuple(args))

        # Add slot-indexed forces into PDB-order force array
        self._add_sorted_forces(state, block_list)
