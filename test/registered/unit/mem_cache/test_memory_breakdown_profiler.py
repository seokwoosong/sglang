import json

import pytest
import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.memory_breakdown_profiler import (
    MemoryBreakdownProfiler,
    profile_cpu_scope,
    record_mamba_layout,
)
from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup, PoolEntry
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _find(items, category, pool, operation):
    return next(
        item
        for item in items
        if (item["category"], item["pool"], item["operation"])
        == (category, pool, operation)
    )


def test_aggregate_snapshot(tmp_path):
    profiler = MemoryBreakdownProfiler(
        str(tmp_path), flush_interval_s=60, enable_cuda_timing=False
    )
    try:
        with profile_cpu_scope(
            profiler, "allocator", "kv", "alloc", rows=3, num_bytes=96
        ):
            pass
        profiler.increment("compaction", "kv", "moves", rows=2, num_bytes=64)
        profiler.record_sample("mamba_batch", "mamba", "indices", 8)
        profiler.record_sample("mamba_batch", "mamba", "indices", 80)
        profiler.flush()

        payload = json.loads(profiler.output_path.read_text())
        alloc = _find(payload["metrics"], "allocator", "kv", "alloc")
        moves = _find(payload["metrics"], "compaction", "kv", "moves")
        batch = _find(payload["samples"], "mamba_batch", "mamba", "indices")
        assert alloc["calls"] == 1
        assert alloc["rows"] == 3
        assert alloc["bytes"] == 96
        assert moves["rows"] == 2
        assert moves["bytes"] == 64
        assert batch["count"] == 2
        assert batch["min"] == 8
        assert batch["max"] == 80
        assert batch["histogram"] == {"8": 1, "65+": 1}
    finally:
        profiler.close()


def test_scope_records_errors(tmp_path):
    profiler = MemoryBreakdownProfiler(
        str(tmp_path), flush_interval_s=60, enable_cuda_timing=False
    )
    try:
        with pytest.raises(RuntimeError, match="expected"):
            with profile_cpu_scope(profiler, "allocator", "kv", "free"):
                raise RuntimeError("expected")
        profiler.flush()
        payload = json.loads(profiler.output_path.read_text())
        metric = _find(payload["metrics"], "allocator", "kv", "free")
        assert metric["calls"] == 1
        assert metric["errors"] == 1
    finally:
        profiler.close()


def test_mamba_layout_records_actual_slot_stride(tmp_path):
    profiler = MemoryBreakdownProfiler(
        str(tmp_path), flush_interval_s=60, enable_cuda_timing=False
    )
    raw = torch.empty((5, 3, 2, 4))
    layer_first = raw.transpose(0, 1)
    try:
        record_mamba_layout(
            profiler,
            pool="mamba",
            layout_kind="page_first_envelope",
            conv=[layer_first],
            temporal=layer_first,
        )
        profiler.flush()
        payload = json.loads(profiler.output_path.read_text())
        layout = payload["layouts"][0]
        assert layout["temporal"]["row_bytes"] == 8 * raw.element_size()
        assert (
            layout["temporal"]["slot_stride_bytes"]
            == raw.stride(0) * raw.element_size()
        )
        assert layout["temporal"]["slot_stride_amplification"] == 3
    finally:
        profiler.close()


def test_host_pool_group_profiles_pool_transfer_components(tmp_path):
    class FakeHostPool:
        layout = "page_first"
        page_size = 2
        size_per_token = 100
        layer_num = 2
        device = "cpu"
        size = 32
        logical_size = 32
        can_use_write_back_jit = False

        def load_to_device_per_layer(self, *args):
            pass

        def backup_from_device_all_layer(self, *args):
            pass

    host_pool = FakeHostPool()
    entry = PoolEntry(
        name=PoolName.KV,
        host_pool=host_pool,
        device_pool=object(),
        layer_mapper=lambda layer_id: layer_id,
        is_primary_index_anchor=True,
    )
    group = HostPoolGroup([entry])
    profiler = MemoryBreakdownProfiler(
        str(tmp_path), flush_interval_s=60, enable_cuda_timing=False
    )
    group._memory_profiler = profiler
    indices = torch.arange(4)
    try:
        for layer_id in range(2):
            group.load_to_device_per_layer(None, indices, indices, layer_id, "kernel")
        group.backup_from_device_all_layer(None, indices, indices, "kernel")
        profiler.flush()

        payload = json.loads(profiler.output_path.read_text())
        h2d = _find(
            payload["metrics"],
            "hicache_transfer_dispatch",
            "kv",
            "h2d_per_layer",
        )
        d2h = _find(
            payload["metrics"],
            "hicache_transfer_dispatch",
            "kv",
            "d2h_all_layers",
        )
        h2d_batch = _find(
            payload["samples"],
            "hicache_transfer_batch",
            "kv",
            "h2d_per_layer",
        )
        assert h2d["calls"] == 2
        assert h2d["bytes"] == 400
        assert d2h["calls"] == 1
        assert d2h["bytes"] == 400
        assert h2d_batch["count"] == 1
        assert h2d_batch["sum"] == 4
    finally:
        profiler.close()
