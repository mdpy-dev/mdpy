"""Shared CUDA kernel source snippets reused across mdpy.

These strings are prepended to (or imported by) cupy.RawKernel source strings
in multiple modules. Centralizing them prevents drift: a bug fix or
optimization applies everywhere at once.

Consumers:
    PBC_MIN_IMAGE_DEVICE_FN   — constraint/lincs.py, constraint/settle.py
    REMAP_INDICES_KERNEL_SRC  — force/_utils.py, constraint/settle.py,
                                constraint/lincs.py
"""

PBC_MIN_IMAGE_DEVICE_FN = r"""
__device__ __forceinline__ void pbc_min_image(
    float& dx, float& dy, float& dz,
    const float* __restrict__ pbc_inv,
    const float* __restrict__ pbc_matrix
) {
    float fx = dx*pbc_inv[0] + dy*pbc_inv[3] + dz*pbc_inv[6];
    float fy = dx*pbc_inv[1] + dy*pbc_inv[4] + dz*pbc_inv[7];
    float fz = dx*pbc_inv[2] + dy*pbc_inv[5] + dz*pbc_inv[8];
    fx -= roundf(fx); fy -= roundf(fy); fz -= roundf(fz);
    dx = fx*pbc_matrix[0] + fy*pbc_matrix[3] + fz*pbc_matrix[6];
    dy = fx*pbc_matrix[1] + fy*pbc_matrix[4] + fz*pbc_matrix[7];
    dz = fx*pbc_matrix[2] + fy*pbc_matrix[5] + fz*pbc_matrix[8];
}
"""

REMAP_INDICES_KERNEL_SRC = r"""
extern "C" __global__
void remap_indices_kernel(
    const int* __restrict__ d_remap,
    int* __restrict__ d_indices,
    int num_indices
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_indices) return;
    int val = d_indices[i];
    if (val >= 0) {
        d_indices[i] = d_remap[val];
    }
}
"""

FILL_CONSTANT_INT32_KERNEL_SRC = r"""
extern "C" __global__
void fill_constant_int32_kernel(
    int* __restrict__ out,
    int number_elements,
    int value
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= number_elements) return;
    out[i] = value;
}
"""
