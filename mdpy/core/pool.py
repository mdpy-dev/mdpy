"""Shared grow-on-demand GPU buffer pool helper.

Used by core/block_list.py (_pool_get) and core/topology.py (_excl_get) to
avoid re-allocating work buffers of stable dtype across rebuilds. Buffers are
keyed by (name, dtype); they grow when a larger size is requested and are
sliced to the requested size on return.
"""
import numpy as np
import cupy as cp
from mdpy.core.kernel_preambles import FILL_CONSTANT_INT32_KERNEL_SRC

_fill_constant_kernel = cp.RawKernel(FILL_CONSTANT_INT32_KERNEL_SRC, "fill_constant_int32_kernel")


def _fill_constant_int32(arr, value):
    n = arr.shape[0]
    threads_per_block = 256
    grid = ((n + threads_per_block - 1) // threads_per_block,)
    _fill_constant_kernel(grid, (threads_per_block,), (arr, np.int32(n), np.int32(value)))


def pool_get(pool, name, size, dtype, fill=None):
    """Return a reusable buffer of the given size from ``pool``.

    Allocates on first call or when the requested size grows; otherwise
    returns the cached array sliced to ``size``. When ``fill`` is not None the
    buffer is filled: memsetAsync(0) for fill == 0, otherwise an int32 fill
    kernel (current callers only exercise fill in {-1, 0}).
    """
    key = (name, dtype)
    arr = pool.get(key)
    if arr is None or arr.size < size:
        arr = cp.empty(size, dtype=dtype)
        pool[key] = arr
    arr = arr[:size]
    if fill is not None:
        if fill == 0:
            cp.cuda.runtime.memsetAsync(
                arr.data.ptr, 0, size * arr.itemsize, cp.cuda.Stream.null.ptr
            )
        else:
            _fill_constant_int32(arr, fill)
    return arr
