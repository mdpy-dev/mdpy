"""Custom RawKernels that replace cupy library operations in the rebuild path.
All compiled once at import; no JIT in the hot path."""
from __future__ import annotations
import numpy as np
import cupy as cp

_FILL_INDEX_KERNEL = r"""
extern "C" __global__
void fill_index_kernel(int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = i;
}
"""

_COMPOSE_PERM_KERNEL = r"""
extern "C" __global__
void compose_perm_kernel(int* out, const int* __restrict__ perm, int N) {
    /* Replaces: out[perm] = arange(N)  =>  out[perm[i]] = i */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[perm[i]] = i;
}
"""

_COPY_INT32_KERNEL = r"""
extern "C" __global__
void copy_int32_kernel(const int* __restrict__ src, int* __restrict__ dst, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = src[i];
}
"""

_KERNELS = None

def compile_rebuild_kernels():
    global _KERNELS
    if _KERNELS is None:
        _KERNELS = {
            "fill_index": cp.RawKernel(_FILL_INDEX_KERNEL, "fill_index_kernel"),
            "compose_perm": cp.RawKernel(_COMPOSE_PERM_KERNEL, "compose_perm_kernel"),
            "copy_int32": cp.RawKernel(_COPY_INT32_KERNEL, "copy_int32_kernel"),
        }
    return _KERNELS
