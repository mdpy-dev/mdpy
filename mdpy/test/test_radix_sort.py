import numpy as np
import cupy as cp
import pytest
from mdpy.core.radix_sort import RadixSorter, fill_constant


class TestRadixSorterBasic:

    def test_sort_empty(self):
        sorter = RadixSorter(max_elements=100)
        keys = cp.array([], dtype=np.uint64)
        result = sorter.argsort(keys)
        assert result.size == 0

    def test_sort_single_element(self):
        sorter = RadixSorter(max_elements=100)
        keys = cp.array([42], dtype=np.uint64)
        result = sorter.argsort(keys)
        expected = cp.argsort(keys).astype(np.int32)
        np.testing.assert_array_equal(cp.asnumpy(result), cp.asnumpy(expected))

    def test_sort_already_sorted(self):
        sorter = RadixSorter(max_elements=100)
        keys = cp.array([1, 2, 3, 4, 5], dtype=np.uint64)
        result = sorter.argsort(keys)
        expected = cp.argsort(keys).astype(np.int32)
        np.testing.assert_array_equal(cp.asnumpy(result), cp.asnumpy(expected))

    def test_sort_reverse_sorted(self):
        sorter = RadixSorter(max_elements=100)
        keys = cp.array([5, 4, 3, 2, 1], dtype=np.uint64)
        result = sorter.argsort(keys)
        expected = cp.argsort(keys).astype(np.int32)
        np.testing.assert_array_equal(cp.asnumpy(result), cp.asnumpy(expected))

    def test_sort_duplicates(self):
        sorter = RadixSorter(max_elements=100)
        keys = cp.array([3, 1, 3, 1, 2], dtype=np.uint64)
        result = sorter.argsort(keys)
        result_np = cp.asnumpy(result)
        sorted_keys = cp.asnumpy(keys[result_np])
        assert list(sorted_keys) == [1, 1, 2, 3, 3]

    def test_sort_stability(self):
        """Elements with equal keys must maintain their original relative order."""
        sorter = RadixSorter(max_elements=100)
        keys = cp.array([10, 10, 10, 10], dtype=np.uint64)
        result = sorter.argsort(keys)
        np.testing.assert_array_equal(cp.asnumpy(result), [0, 1, 2, 3])

    def test_sort_matches_cupy_small(self):
        rng = np.random.RandomState(42)
        for size in [10, 50, 100, 255, 256, 257, 500, 512, 1000]:
            sorter = RadixSorter(max_elements=max(1001, size))
            data = rng.randint(0, 2**20, size=size, dtype=np.int64)
            keys = cp.asarray(data.astype(np.uint64))
            result = sorter.argsort(keys)
            expected = cp.argsort(keys).astype(np.int32)
            np.testing.assert_array_equal(
                cp.asnumpy(result), cp.asnumpy(expected),
                err_msg=f"Mismatch at size={size}"
            )

    def test_sort_large_random(self):
        sorter = RadixSorter(max_elements=100000)
        rng = np.random.RandomState(123)
        data = rng.randint(0, 2**40, size=95567, dtype=np.int64)
        keys = cp.asarray(data.astype(np.uint64))
        result = sorter.argsort(keys)
        expected = cp.argsort(keys).astype(np.int32)
        np.testing.assert_array_equal(cp.asnumpy(result), cp.asnumpy(expected))

    def test_sort_cell_morton_keys(self):
        """Keys with structure (cell_index << 30) | morton_code, matching block_list usage."""
        sorter = RadixSorter(max_elements=100000)
        rng = np.random.RandomState(456)
        cell_indices = rng.randint(0, 729, size=95567)
        morton = rng.randint(0, 2**30, size=95567)
        keys_np = (cell_indices.astype(np.uint64) << 30) | morton.astype(np.uint64)
        keys = cp.asarray(keys_np)
        result = sorter.argsort(keys)
        expected = cp.argsort(keys).astype(np.int32)
        np.testing.assert_array_equal(cp.asnumpy(result), cp.asnumpy(expected))


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
