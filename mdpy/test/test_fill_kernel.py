import numpy as np
import cupy as cp
from mdpy.core.block_list import _fill_constant as fill_constant


class TestFillConstant:

    def test_fill_negative_one(self):
        arr = cp.empty(1000, dtype=np.int32)
        fill_constant(arr, -1)
        np.testing.assert_array_equal(cp.asnumpy(arr), np.full(1000, -1, dtype=np.int32))

    def test_fill_zero(self):
        arr = cp.empty(500, dtype=np.int32)
        fill_constant(arr, 0)
        np.testing.assert_array_equal(cp.asnumpy(arr), np.zeros(500, dtype=np.int32))

    def test_fill_large(self):
        arr = cp.empty(100000, dtype=np.int32)
        fill_constant(arr, 42)
        np.testing.assert_array_equal(cp.asnumpy(arr), np.full(100000, 42, dtype=np.int32))
