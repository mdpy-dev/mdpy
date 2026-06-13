from __future__ import annotations

import cupy as cp
import numpy as np
import re as _re

from mdpy import env
from mdpy.force.force_term import ForceTerm

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


class _BondedForceAggregator:
    name = 'bonded'

    def __init__(self, sub_forces):
        self._sub_forces = sub_forces

    def compute(self, gpu_context, block_list=None, compute_energy=True):
        for force in self._sub_forces:
            force.compute(gpu_context, block_list, compute_energy)

    def remap_indices_gpu(self, d_remap):
        for force in self._sub_forces:
            force.remap_indices_gpu(d_remap)

    def bind_sorted(self, topology, block_list, gpu_context):
        for force in self._sub_forces:
            if hasattr(force, 'bind_sorted'):
                force.bind_sorted(topology, block_list, gpu_context)


_BODY_TEMPLATES = {
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
        body_template = _BODY_TEMPLATES[self._body]
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
        self._kernel_source = _PREAMBLE + _MAIN_TEMPLATE.format(
            body=body, extra_params=extra_params,
        )

    def remap_indices_gpu(self, d_remap):
        if self._count == 0 or self._d_indices is None:
            return
        kernel = self._get_remap_kernel()
        indices = self._d_indices.ravel()
        n = indices.size
        tpb = 256
        grid = ((n + tpb - 1) // tpb,)
        kernel(grid, (tpb,), (d_remap, indices, np.int32(n)))

    def bind_sorted(self, topology, block_list, gpu_context):
        sort_order = block_list.d_raw_order
        for prop_name in self._per_particle_properties:
            d_arr = self._per_particle_gpu[prop_name]
            arrays = {prop_name: d_arr}
            gpu_context.permute_to_sorted(sort_order, arrays)
            self._per_particle_gpu[prop_name] = arrays[prop_name]

    def _ensure_compiled(self):
        if self._kernel is not None:
            return
        self._kernel = cp.RawKernel(self._kernel_source, 'compute_bonded')
        if self._num_sm is None:
            self._num_sm = cp.cuda.runtime.getDeviceProperties(0)['multiProcessorCount']

    @classmethod
    def charmm(cls, topology, parameter_table):
        from mdpy.forcefield.charmm_forces import (
            _create_bond_force, _create_angle_force,
            _create_dihedral_force, _create_improper_force,
        )
        sub_forces = [
            _create_bond_force(topology, parameter_table),
            _create_angle_force(topology, parameter_table),
            _create_dihedral_force(topology, parameter_table),
            _create_improper_force(topology, parameter_table),
        ]
        return _BondedForceAggregator(sub_forces)

    def compute(self, gpu_context, block_list=None, compute_energy=True):
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
            if prop_name == 'charge' and gpu_context.d_charges is not None:
                args.append(gpu_context.d_charges)
            elif prop_name in self._per_particle_gpu:
                args.append(self._per_particle_gpu[prop_name])
        self._kernel((grid_size,), (block_size,), tuple(args))
