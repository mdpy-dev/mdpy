from __future__ import annotations

import numpy as np
import cupy as cp

_RADIX_HISTOGRAM_KERNEL = r"""
extern "C" __global__
void radix_histogram_kernel(
    const unsigned long long* __restrict__ keys,
    int number_elements,
    int pass_idx,
    int* __restrict__ block_hist
) {
    __shared__ int shared_hist[16];
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (tid < 16) shared_hist[tid] = 0;
    __syncthreads();

    for (int i = blockIdx.x * block_size + tid;
         i < number_elements;
         i += block_size * gridDim.x) {
        int digit = (keys[i] >> (pass_idx * 4)) & 0xF;
        atomicAdd(&shared_hist[digit], 1);
    }
    __syncthreads();

    if (tid < 16)
        block_hist[blockIdx.x * 16 + tid] = shared_hist[tid];
}
"""

_RADIX_PREFIX_KERNEL = r"""
extern "C" __global__
void radix_prefix_kernel(
    const int* __restrict__ block_hist,
    int number_blocks,
    int* __restrict__ digit_prefix
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int running = 0;
    for (int d = 0; d < 16; d++) {
        for (int b = 0; b < number_blocks; b++) {
            int idx = b * 16 + d;
            digit_prefix[idx] = running;
            running += block_hist[idx];
        }
    }
}
"""

_RADIX_SCATTER_KERNEL = r"""
extern "C" __global__
void radix_scatter_kernel(
    const unsigned long long* __restrict__ keys_in,
    unsigned long long* __restrict__ keys_out,
    const int* __restrict__ vals_in,
    int* __restrict__ vals_out,
    int number_elements,
    int pass_idx,
    int number_blocks,
    const int* __restrict__ digit_prefix,
    int* __restrict__ block_counters
) {
    __shared__ int shared_counters[16];
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int bid = blockIdx.x;

    if (tid < 16) shared_counters[tid] = 0;
    __syncthreads();

    for (int i = bid * block_size + tid;
         i < number_elements;
         i += block_size * gridDim.x) {
        int digit = (keys_in[i] >> (pass_idx * 4)) & 0xF;
        int local_pos = atomicAdd(&shared_counters[digit], 1);
        int global_base = digit_prefix[bid * 16 + digit];
        int pos = global_base + local_pos;
        keys_out[pos] = keys_in[i];
        vals_out[pos] = vals_in[i];
    }
}
"""

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


class RadixSorter:
    def __init__(self, max_elements: int, num_bits: int = 40):
        self._max_elements = max_elements
        self._num_bits = num_bits
        self._num_passes = (num_bits + 3) // 4
        self._tpb = 256

        self._hist_kernel = cp.RawKernel(_RADIX_HISTOGRAM_KERNEL, "radix_histogram_kernel")
        self._prefix_kernel = cp.RawKernel(_RADIX_PREFIX_KERNEL, "radix_prefix_kernel")
        self._scatter_kernel = cp.RawKernel(_RADIX_SCATTER_KERNEL, "radix_scatter_kernel")

        num_blocks = (max_elements + self._tpb - 1) // self._tpb

        self._d_keys_buf = [cp.empty(max_elements, dtype=np.uint64),
                            cp.empty(max_elements, dtype=np.uint64)]
        self._d_vals_buf = [cp.empty(max_elements, dtype=np.int32),
                            cp.empty(max_elements, dtype=np.int32)]
        self._d_block_hist = cp.empty(num_blocks * 16, dtype=np.int32)
        self._d_digit_prefix = cp.empty(num_blocks * 16, dtype=np.int32)

    def argsort(self, keys: cp.ndarray) -> cp.ndarray:
        N = keys.shape[0]
        assert N <= self._max_elements
        assert keys.dtype == np.uint64

        num_blocks = (N + self._tpb - 1) // self._tpb
        grid = (num_blocks,)

        self._d_keys_buf[0][:N] = keys
        self._d_vals_buf[0][:N] = cp.arange(N, dtype=np.int32)

        src = 0
        for p in range(self._num_passes):
            self._hist_kernel(
                grid, (self._tpb,),
                (self._d_keys_buf[src], np.int32(N), np.int32(p),
                 self._d_block_hist),
            )
            self._prefix_kernel(
                (1,), (1,),
                (self._d_block_hist, np.int32(num_blocks),
                 self._d_digit_prefix),
            )
            self._scatter_kernel(
                grid, (self._tpb,),
                (self._d_keys_buf[src], self._d_keys_buf[1 - src],
                 self._d_vals_buf[src], self._d_vals_buf[1 - src],
                 np.int32(N), np.int32(p), np.int32(num_blocks),
                 self._d_digit_prefix, self._d_block_hist),
            )
            src = 1 - src

        return self._d_vals_buf[src][:N].copy()


_fill_kernel = cp.RawKernel(_FILL_CONSTANT_KERNEL, "fill_constant_int32_kernel")


def fill_constant(arr: cp.ndarray, value: int) -> None:
    N = arr.shape[0]
    tpb = 256
    grid = ((N + tpb - 1) // tpb,)
    _fill_kernel(grid, (tpb,), (arr, np.int32(N), np.int32(value)))
