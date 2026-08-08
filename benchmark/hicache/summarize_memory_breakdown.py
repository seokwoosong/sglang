"""Compare serving and memory-path profiles from unified ablation results.

The reported CPU and CUDA intervals are independent observations and can
overlap.  They must not be added together and interpreted as an end-to-end
latency decomposition.  The tables are intended to identify which path differs
between otherwise matched Baseline/U1/U2 runs before deeper Nsight profiling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def discover_results(paths: Iterable[Path]) -> list[Path]:
    results: list[Path] = []
    seen: set[Path] = set()

    def add(item: Path) -> None:
        resolved = item.resolve()
        if resolved not in seen:
            seen.add(resolved)
            results.append(resolved)

    for path in paths:
        if path.is_file():
            add(path)
        elif (path / "result.json").is_file():
            add(path / "result.json")
        elif path.is_dir():
            for item in sorted(path.rglob("result.json")):
                add(item)
        else:
            raise FileNotFoundError(path)
    if not results:
        raise FileNotFoundError("No result.json files were found")
    return results


def sum_metric(
    profile: dict[str, Any],
    field: str,
    value: str,
    *,
    category: str | None = None,
    pool: str | None = None,
    operation: str | None = None,
    operation_suffix: str | None = None,
) -> int:
    total = 0
    for item in profile.get(field, []):
        if category is not None and item.get("category") != category:
            continue
        if pool is not None and item.get("pool") != pool:
            continue
        if operation is not None and item.get("operation") != operation:
            continue
        if operation_suffix is not None and not str(item.get("operation", "")).endswith(
            operation_suffix
        ):
            continue
        total += int(item.get(value, 0))
    return total


def transfer_metric(
    profile: dict[str, Any],
    value: str,
    *,
    pool: str,
    operation: str,
    cuda: bool = True,
) -> int:
    return sum_metric(
        profile,
        "cuda_metrics" if cuda else "metrics",
        value,
        category=("hicache_transfer_gpu" if cuda else "hicache_transfer_dispatch"),
        pool=pool,
        operation=operation,
    )


def gib_per_second(num_bytes: int, elapsed_ns: int) -> float:
    if elapsed_ns <= 0:
        return 0.0
    return num_bytes / (1024**3) / (elapsed_ns / 1e9)


def counter(result: dict[str, Any], name: str) -> float:
    source = result.get("measured_metric_delta") or result.get("total_metric_delta", {})
    return float(source.get(name, 0.0))


def average_duration(result: dict[str, Any], prefix: str) -> float:
    duration = counter(result, f"sglang:{prefix}_duration_seconds_sum")
    calls = counter(result, f"sglang:{prefix}_duration_seconds_count")
    return duration / calls if calls else 0.0


def mamba_batch(profile: dict[str, Any]) -> tuple[int, float]:
    samples = [
        item
        for item in profile.get("samples", [])
        if item.get("category") == "mamba_batch"
    ]
    count = sum(int(item.get("count", 0)) for item in samples)
    total = sum(int(item.get("sum", 0)) for item in samples)
    return count, total / count if count else 0.0


def mamba_layout(profile: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        item
        for item in profile.get("layouts", [])
        if isinstance(item.get("temporal"), dict)
    ]
    if not candidates:
        return {}
    # There is one active Mamba pool per rank. Prefer the explicit mamba pool if
    # another component ever starts publishing tensor layouts as well.
    return next(
        (item for item in candidates if item.get("pool") == "mamba"), candidates[0]
    )


def summarize(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    profile = result.get("memory_profile_measured_delta") or result.get(
        "memory_profile_total_delta", {}
    )
    summary = result.get("summary", {})
    layout = mamba_layout(profile)
    temporal = layout.get("temporal", {})
    batch_calls, batch_mean = mamba_batch(profile)
    compaction_bytes = sum_metric(
        profile,
        "metrics",
        "bytes",
        category="compaction",
        operation_suffix="_moves",
    )
    h2d_total_ns = transfer_metric(
        profile, "cpu_time_ns", pool="all", operation="h2d_total"
    )
    d2h_total_ns = transfer_metric(
        profile, "cpu_time_ns", pool="all", operation="d2h_total"
    )
    h2d_total_bytes = transfer_metric(
        profile, "bytes", pool="all", operation="h2d_total"
    )
    d2h_total_bytes = transfer_metric(
        profile, "bytes", pool="all", operation="d2h_total"
    )

    transfer_parts: dict[str, float] = {}
    for direction, operation in (
        ("h2d", "h2d_per_layer"),
        ("d2h", "d2h_all_layers"),
    ):
        for pool in ("kv", "mamba"):
            elapsed_ns = transfer_metric(
                profile, "cpu_time_ns", pool=pool, operation=operation
            )
            num_bytes = transfer_metric(
                profile, "bytes", pool=pool, operation=operation
            )
            transfer_parts[f"{pool}_{direction}_gpu_ms"] = elapsed_ns / 1e6
            transfer_parts[f"{pool}_{direction}_gib_s"] = gib_per_second(
                num_bytes, elapsed_ns
            )
    return {
        "path": str(path),
        "variant": result.get("variant", path.parent.parent.name),
        "page_size": int(result.get("server_info", {}).get("page_size", 0) or 0),
        "validation_passed": bool(result.get("validation", {}).get("passed", False)),
        "duration_s": float(summary.get("duration_s", 0.0) or 0.0),
        "ttft_ms": float(summary.get("ttft_ms", {}).get("mean", 0.0) or 0.0),
        "tpot_ms": float(summary.get("tpot_ms", {}).get("mean", 0.0) or 0.0),
        "tokens_per_s": float(summary.get("total_token_throughput", 0.0) or 0.0),
        "forward_s": counter(result, "sglang:forward_execution_seconds_total"),
        "backup_mean_ms": average_duration(result, "hicache_backup") * 1e3,
        "loadback_mean_ms": average_duration(result, "load_back") * 1e3,
        "h2d_total_gpu_ms": h2d_total_ns / 1e6,
        "h2d_total_gib_s": gib_per_second(h2d_total_bytes, h2d_total_ns),
        "d2h_total_gpu_ms": d2h_total_ns / 1e6,
        "d2h_total_gib_s": gib_per_second(d2h_total_bytes, d2h_total_ns),
        "h2d_enqueue_cpu_ms": sum_metric(
            profile,
            "metrics",
            "cpu_time_ns",
            category="hicache_transfer_control",
            pool="all",
            operation="h2d_enqueue",
        )
        / 1e6,
        "d2h_enqueue_cpu_ms": sum_metric(
            profile,
            "metrics",
            "cpu_time_ns",
            category="hicache_transfer_control",
            pool="all",
            operation="d2h_enqueue",
        )
        / 1e6,
        "allocator_cpu_ms": sum_metric(
            profile, "metrics", "cpu_time_ns", category="allocator"
        )
        / 1e6,
        "translation_cpu_ms": sum_metric(
            profile, "metrics", "cpu_time_ns", category="translation"
        )
        / 1e6,
        "translation_rows": sum_metric(
            profile, "metrics", "rows", category="translation"
        ),
        "compaction_cpu_ms": sum_metric(
            profile, "metrics", "cpu_time_ns", category="compaction"
        )
        / 1e6,
        "compaction_gpu_ms": sum_metric(
            profile, "cuda_metrics", "cpu_time_ns", category="compaction_gpu"
        )
        / 1e6,
        "row_fence_gpu_ms": sum_metric(
            profile, "cuda_metrics", "cpu_time_ns", category="row_fence_gpu"
        )
        / 1e6,
        "compaction_moved_gib": compaction_bytes / (1024**3),
        "compaction_deferred": sum_metric(
            profile,
            "metrics",
            "calls",
            category="row_fence",
            operation="opportunistic_compaction_deferred",
        ),
        "urgent_waits": sum_metric(
            profile,
            "metrics",
            "calls",
            category="row_fence",
            operation="urgent_wait_inserted",
        ),
        "unrelated_allowed": sum_metric(
            profile,
            "metrics",
            "calls",
            category="row_fence",
            operation="unrelated_compaction_allowed",
        ),
        "layout_copy_calls": sum_metric(
            profile,
            "metrics",
            "calls",
            category="mamba_layout_access",
            operation="extend_state_gather",
        ),
        "layout_copy_gib": (
            sum_metric(
                profile,
                "metrics",
                "bytes",
                category="mamba_layout_access",
                operation="extend_state_gather",
            )
            + sum_metric(
                profile,
                "metrics",
                "bytes",
                category="mamba_layout_access",
                operation="extend_state_scatter",
            )
        )
        / (1024**3),
        "layout_gather_gpu_ms": sum_metric(
            profile,
            "cuda_metrics",
            "cpu_time_ns",
            category="mamba_layout_gpu",
            operation="extend_state_gather",
        )
        / 1e6,
        "layout_scatter_gpu_ms": sum_metric(
            profile,
            "cuda_metrics",
            "cpu_time_ns",
            category="mamba_layout_gpu",
            operation="extend_state_scatter",
        )
        / 1e6,
        "decode_layer_accesses": sum_metric(
            profile,
            "metrics",
            "calls",
            category="mamba_layout_access",
            operation="decode_layer",
        ),
        "mamba_batch_calls": batch_calls,
        "mamba_batch_mean": batch_mean,
        "mamba_layout": layout.get("layout_kind", "n/a"),
        "mamba_slot_stride_mib": float(temporal.get("slot_stride_bytes", 0))
        / (1024**2),
        "mamba_row_kib": float(temporal.get("row_bytes", 0)) / 1024,
        "mamba_stride_amplification": float(
            temporal.get("slot_stride_amplification", 0.0)
        ),
        **transfer_parts,
    }


def fmt(value: Any, digits: int = 2) -> str:
    if isinstance(value, bool):
        return "yes" if value else "NO"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    output = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        output.append(
            "| " + " | ".join(fmt(row.get(key, "")) for key, _ in columns) + " |"
        )
    return "\n".join(output)


def render(rows: list[dict[str, Any]]) -> str:
    performance = table(
        rows,
        [
            ("variant", "Variant"),
            ("page_size", "Page"),
            ("validation_passed", "Valid"),
            ("duration_s", "Wall s"),
            ("ttft_ms", "TTFT ms"),
            ("tpot_ms", "TPOT ms"),
            ("tokens_per_s", "Tokens/s"),
            ("forward_s", "Forward s"),
            ("backup_mean_ms", "D2H mean ms"),
            ("loadback_mean_ms", "H2D mean ms"),
        ],
    )
    breakdown = table(
        rows,
        [
            ("variant", "Variant"),
            ("page_size", "Page"),
            ("allocator_cpu_ms", "Allocator CPU ms (incl.)"),
            ("translation_cpu_ms", "Translate CPU ms"),
            ("compaction_cpu_ms", "Compact CPU ms"),
            ("compaction_gpu_ms", "Relocate GPU ms"),
            ("row_fence_gpu_ms", "Fence GPU ms"),
            ("compaction_moved_gib", "Moved GiB"),
            ("compaction_deferred", "Deferred"),
            ("urgent_waits", "Urgent waits"),
        ],
    )
    locality = table(
        rows,
        [
            ("variant", "Variant"),
            ("page_size", "Page"),
            ("mamba_layout", "Mamba layout"),
            ("mamba_batch_calls", "Index batches"),
            ("mamba_batch_mean", "Mean batch"),
            ("mamba_row_kib", "Row KiB"),
            ("mamba_slot_stride_mib", "Slot stride MiB"),
            ("mamba_stride_amplification", "Stride/row"),
            ("layout_copy_calls", "Prefill copies"),
            ("layout_copy_gib", "Prefill copy GiB"),
            ("layout_gather_gpu_ms", "Gather GPU ms"),
            ("layout_scatter_gpu_ms", "Scatter GPU ms"),
            ("decode_layer_accesses", "Decode layers"),
            ("unrelated_allowed", "Fence-unrelated"),
        ],
    )
    transfer_total = table(
        rows,
        [
            ("variant", "Variant"),
            ("page_size", "Page"),
            ("d2h_total_gpu_ms", "D2H GPU ms"),
            ("d2h_total_gib_s", "D2H GiB/s"),
            ("d2h_enqueue_cpu_ms", "D2H enqueue CPU ms"),
            ("h2d_total_gpu_ms", "H2D GPU ms"),
            ("h2d_total_gib_s", "H2D GiB/s"),
            ("h2d_enqueue_cpu_ms", "H2D enqueue CPU ms"),
        ],
    )
    transfer_parts = table(
        rows,
        [
            ("variant", "Variant"),
            ("page_size", "Page"),
            ("kv_d2h_gpu_ms", "KV D2H ms"),
            ("kv_d2h_gib_s", "KV D2H GiB/s"),
            ("mamba_d2h_gpu_ms", "Mamba D2H ms"),
            ("mamba_d2h_gib_s", "Mamba D2H GiB/s"),
            ("kv_h2d_gpu_ms", "KV H2D ms"),
            ("kv_h2d_gib_s", "KV H2D GiB/s"),
            ("mamba_h2d_gpu_ms", "Mamba H2D ms"),
            ("mamba_h2d_gib_s", "Mamba H2D GiB/s"),
        ],
    )
    return (
        "# Memory-path comparison\n\n"
        "CPU and CUDA intervals may overlap; do not add them as an exact latency "
        "decomposition. Allocator CPU time is inclusive of compaction invoked "
        "inside allocation. All profile values use the measured phase when present.\n\n"
        "## Serving and HiCache transfer\n\n"
        f"{performance}\n\n"
        "## HiCache transfer decomposition\n\n"
        "The total interval is the transfer-stream envelope. Per-pool intervals "
        "are nested component measurements and should not be added to the total.\n\n"
        f"{transfer_total}\n\n"
        f"{transfer_parts}\n\n"
        "## Allocator, translation, compaction, and fence\n\n"
        f"{breakdown}\n\n"
        "## Mamba layout and batch locality\n\n"
        f"{locality}\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--json-output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = [summarize(path) for path in discover_results(args.results)]
    markdown = render(rows)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(markdown)


if __name__ == "__main__":
    main()
