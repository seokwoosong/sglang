"""Unit tests for the unified-memory profile comparison helper."""

import json
import sys
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[3]
HICACHE_DIR = REPO_ROOT / "benchmark" / "hicache"
if str(HICACHE_DIR) not in sys.path:
    sys.path.insert(0, str(HICACHE_DIR))

import summarize_memory_breakdown as summary_tool  # noqa: E402


def _metric(
    category,
    operation,
    *,
    pool="mamba",
    calls=1,
    cpu_time_ns=0,
    rows=0,
    num_bytes=0,
):
    return {
        "category": category,
        "pool": pool,
        "operation": operation,
        "calls": calls,
        "errors": 0,
        "cpu_time_ns": cpu_time_ns,
        "rows": rows,
        "bytes": num_bytes,
    }


def test_summarize_measured_memory_profile(tmp_path):
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "variant": "direct-u2",
                "server_info": {"page_size": 16},
                "validation": {"passed": True},
                "summary": {
                    "duration_s": 10,
                    "ttft_ms": {"mean": 20},
                    "tpot_ms": {"mean": 3},
                    "total_token_throughput": 100,
                },
                "measured_metric_delta": {
                    "sglang:forward_execution_seconds_total": 4,
                },
                "memory_profile_measured_delta": {
                    "metrics": [
                        _metric("allocator", "alloc", cpu_time_ns=1_000_000),
                        _metric("translation", "virtual", cpu_time_ns=2_000_000),
                        _metric(
                            "compaction",
                            "opportunistic_flush_moves",
                            num_bytes=1024**3,
                        ),
                        _metric(
                            "mamba_layout_access",
                            "extend_state_gather",
                            calls=2,
                            num_bytes=1024**3,
                        ),
                        _metric(
                            "mamba_layout_access",
                            "extend_state_scatter",
                            calls=2,
                            num_bytes=1024**3,
                        ),
                    ],
                    "cuda_metrics": [
                        _metric(
                            "compaction_gpu",
                            "lazy_relocation",
                            cpu_time_ns=3_000_000,
                        ),
                        _metric(
                            "hicache_transfer_gpu",
                            "h2d_total",
                            pool="all",
                            cpu_time_ns=10_000_000,
                            num_bytes=1024**3,
                        ),
                        _metric(
                            "mamba_layout_gpu",
                            "extend_state_gather",
                            cpu_time_ns=4_000_000,
                        ),
                        _metric(
                            "mamba_layout_gpu",
                            "extend_state_scatter",
                            cpu_time_ns=5_000_000,
                        ),
                        _metric(
                            "hicache_transfer_gpu",
                            "h2d_per_layer",
                            pool="kv",
                            cpu_time_ns=8_000_000,
                            num_bytes=1024**3,
                        ),
                    ],
                    "samples": [
                        {
                            "category": "mamba_batch",
                            "pool": "mamba",
                            "operation": "virtual_indices",
                            "count": 2,
                            "sum": 16,
                            "histogram": {"8": 2},
                        }
                    ],
                    "layouts": [
                        {
                            "pool": "mamba",
                            "layout_kind": "page_first_envelope",
                            "temporal": {
                                "row_bytes": 1024,
                                "slot_stride_bytes": 2 * 1024**2,
                                "slot_stride_amplification": 2048,
                            },
                        }
                    ],
                },
            }
        )
    )

    row = summary_tool.summarize(result_path)

    assert row["variant"] == "direct-u2"
    assert row["page_size"] == 16
    assert row["allocator_cpu_ms"] == 1
    assert row["translation_cpu_ms"] == 2
    assert row["compaction_gpu_ms"] == 3
    assert row["compaction_moved_gib"] == 1
    assert row["layout_copy_calls"] == 2
    assert row["layout_copy_gib"] == 2
    assert row["mamba_batch_mean"] == 8
    assert row["mamba_slot_stride_mib"] == 2
    assert row["h2d_total_gpu_ms"] == 10
    assert row["h2d_total_gib_s"] == 100
    assert row["kv_h2d_gpu_ms"] == 8
    assert row["kv_h2d_gib_s"] == 125
    assert row["layout_gather_gpu_ms"] == 4
    assert row["layout_scatter_gpu_ms"] == 5
    assert "HiCache transfer decomposition" in summary_tool.render([row])
    assert "CPU and CUDA intervals may overlap" in summary_tool.render([row])
