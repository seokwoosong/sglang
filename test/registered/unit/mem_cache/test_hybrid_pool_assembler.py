"""Unit test for hybrid HiCache fixed-size budget splitting."""

import unittest

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _split_hicache_size,
    _use_unified_typed_l2,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Pool:
    def __init__(self, kv_bytes):
        self._kv_bytes = kv_bytes

    def get_kv_size_bytes(self):
        return self._kv_bytes


class _UnifiedSpec:
    def __init__(self, entry_bytes):
        self._entry_bytes = entry_bytes

    def entry_bytes(self):
        return self._entry_bytes


class _UnifiedBuffer:
    def __init__(self, capacities):
        self._capacities = capacities

    def spec(self, name):
        return _UnifiedSpec(self._capacities[name][1])

    def max_slots(self, name):
        return self._capacities[name][0]


class _UnifiedViewPool(_Pool):
    def __init__(self, unified_buffer, name):
        super().__init__((0, 0))
        self._unified_buffer = unified_buffer
        self._sub_pool_name = name


class TestSplitHicacheSize(CustomTestCase):
    def test_unified_typed_l2_ablation_switch(self):
        with envs.SGLANG_HICACHE_UNIFIED_TYPED_L2.override(True):
            self.assertTrue(_use_unified_typed_l2(is_unified_mamba=True))
            self.assertFalse(_use_unified_typed_l2(is_unified_mamba=False))
        with envs.SGLANG_HICACHE_UNIFIED_TYPED_L2.override(False):
            self.assertFalse(_use_unified_typed_l2(is_unified_mamba=True))

    def test_splits_total_budget_by_device_bytes(self):
        # scalar and (k, v) tuple return shapes both supported
        shares = _split_hicache_size(
            100, (_Pool(75 * 10**9), _Pool((15 * 10**9, 10 * 10**9)))
        )
        self.assertEqual(shares, (75.0, 25.0))  # proportional to device KV bytes
        self.assertEqual(sum(shares), 100)  # total budget preserved, not doubled

    def test_splits_total_budget_by_device_bytes_three_pools(self):
        # scalar and (k, v) tuple return shapes both supported
        shares = _split_hicache_size(
            100, (_Pool(55 * 10**9), _Pool((15 * 10**9, 10 * 10**9)), _Pool(20 * 10**9))
        )
        self.assertEqual(shares, (55.0, 25.0, 20.0))  # proportional to device KV bytes
        self.assertEqual(sum(shares), 100)  # total budget preserved, not doubled

    def test_unified_views_use_logical_sub_pool_capacity(self):
        # Unified views report zero physical bytes so the shared GPU allocation
        # is logged only once. Fixed-size L2 splitting must still assign both
        # typed pools a nonzero share instead of triggering ratio-based sizing.
        unified = _UnifiedBuffer(
            {
                "full": (200, 10),
                "mamba": (20, 100),
            }
        )
        shares = _split_hicache_size(
            8,
            (
                _UnifiedViewPool(unified, "full"),
                _UnifiedViewPool(unified, "mamba"),
            ),
        )
        self.assertEqual(shares, (4.0, 4.0))
        self.assertEqual(sum(shares), 8)


if __name__ == "__main__":
    unittest.main()
