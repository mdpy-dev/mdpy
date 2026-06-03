from __future__ import annotations

import numpy as np
import cupy as cp

_FILL_CONSTANT_KERNEL = r"""
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

_fill_kernel = cp.RawKernel(_FILL_CONSTANT_KERNEL, "fill_constant_int32_kernel")


def fill_constant(arr: cp.ndarray, value: int) -> None:
    N = arr.shape[0]
    tpb = 256
    grid = ((N + tpb - 1) // tpb,)
    _fill_kernel(grid, (tpb,), (arr, np.int32(N), np.int32(value)))
