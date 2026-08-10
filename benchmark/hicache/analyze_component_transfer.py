#!/usr/bin/env python3
"""Audit and summarize static versus unified typed HiCache transfers."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "artifacts/hicache_component_transfer"
COMPONENT_OPERATIONS = {
    ("kv", "d2h"): "d2h_all_layers",
    ("mamba", "d2h"): "d2h_all_layers",
    ("kv", "h2d"): "h2d_per_layer",
    ("mamba", "h2d"): "h2d_per_layer",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def median(values) -> float:
    return statistics.median(float(value) for value in values)


def gib_per_second(metric: dict[str, Any]) -> float:
    elapsed_ns = int(metric["cpu_time_ns"])
    return int(metric["bytes"]) / (1024**3) / (elapsed_ns / 1e9)


def profile_metric(profile, pool: str, operation: str) -> dict[str, Any]:
    matches = [
        metric
        for metric in profile["cuda_metrics"]
        if metric["category"] == "hicache_transfer_gpu"
        and metric["pool"] == pool
        and metric["operation"] == operation
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {pool}/{operation} metric, got {matches}")
    return matches[0]


def audit_server(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    errors = []
    for path in sorted(root.glob("component-transfer-r*/eval-*/*/result.json")):
        result = load_json(path)
        manifest = load_json(path.parent / "manifest.json")
        summary = result["summary"]
        metrics = result["measured_metric_delta"]
        run_name = path.parents[2].name
        repetition = int(run_name.split("-r", 1)[1].split("-", 1)[0])
        label = f"{run_name}/{result['variant']}"
        if manifest["status"] != "completed":
            errors.append(f"incomplete manifest: {label}")
        if not result["validation"]["passed"]:
            errors.append(f"validation failed: {label}")
        if summary["completed"] != 80 or summary["failed"] != 0:
            errors.append(f"request failure: {label}")
        for key in (
            "sglang:evicted_tokens_total",
            "sglang:hicache_backup_tokens_total",
            "sglang:load_back_tokens_total",
        ):
            if metrics[key] <= 0:
                errors.append(f"missing {key}: {label}")
        if summary["cached_tokens_host"] <= 0:
            errors.append(f"missing host hit: {label}")
        if metrics["sglang:hicache_dropped_tokens_total"] != 0:
            errors.append(f"dropped tokens: {label}")

        profile = result["memory_profile_measured_delta"]
        component_rates = {}
        for (pool, direction), operation in COMPONENT_OPERATIONS.items():
            component_rates[f"{pool}_{direction}_gib_s"] = gib_per_second(
                profile_metric(profile, pool, operation)
            )
        rows.append(
            {
                "repetition": repetition,
                "variant": result["variant"],
                "throughput_tokens_s": summary["total_token_throughput"],
                **component_rates,
                "result": str(path.relative_to(REPO_ROOT)),
            }
        )
    expected = {
        (repetition, variant)
        for repetition in (1, 2, 3)
        for variant in ("eval-s1", "eval-u3")
    }
    observed = {(row["repetition"], row["variant"]) for row in rows}
    if observed != expected:
        errors.append(
            f"server matrix mismatch: expected={expected}, observed={observed}"
        )
    return rows, errors


def aggregate_server(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    output = []
    for variant, group in sorted(grouped.items()):
        output.append(
            {
                "variant": variant,
                "runs": len(group),
                "throughput_tokens_s_median": median(
                    row["throughput_tokens_s"] for row in group
                ),
                **{
                    f"{pool}_{direction}_gib_s_median": median(
                        row[f"{pool}_{direction}_gib_s"] for row in group
                    )
                    for pool, direction in COMPONENT_OPERATIONS
                },
            }
        )
    return output


def summarize_mamba(path: Path) -> list[dict[str, Any]]:
    rows = load_json(path)["measurements"]
    output = []
    for batch in (1, 4, 8, 16):
        for direction in ("d2h", "h2d"):
            selected = [
                row
                for row in rows
                if row["batch_slots"] == batch
                and row["pattern"] == "contiguous"
                and row["direction"] == direction
            ]
            variants = (
                "baseline-static",
                "ours-component-direct",
                "ours-raw-slot",
            )
            rates = {
                variant: median(
                    row["gib_per_s"] for row in selected if row["path"] == variant
                )
                for variant in variants
            }
            enqueue = {
                f"{variant}_enqueue_us": 1000
                * median(
                    row["enqueue_ms"] for row in selected if row["path"] == variant
                )
                for variant in variants
            }
            output.append(
                {
                    "batch_slots": batch,
                    "direction": direction,
                    **rates,
                    **enqueue,
                    "raw_over_component": rates["ours-raw-slot"]
                    / rates["ours-component-direct"],
                }
            )
    return output


def summarize_kv(path: Path) -> list[dict[str, Any]]:
    rows = load_json(path)["measurements"]
    output = []
    for page_size in (1, 8, 32):
        for direction in ("d2h", "h2d"):
            selected = [
                row
                for row in rows
                if row["page_size"] == page_size
                and row["tokens"] == 4096
                and row["pattern"] == "contiguous"
                and row["direction"] == direction
            ]
            rates = {
                variant: median(
                    row["gib_per_s"] for row in selected if row["variant"] == variant
                )
                for variant in ("baseline-static", "ours-unified-typed")
            }
            output.append(
                {
                    "page_size": page_size,
                    "direction": direction,
                    **rates,
                    "ours_over_baseline": rates["ours-unified-typed"]
                    / rates["baseline-static"],
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    component_server_rows, component_errors = audit_server(root / "server")
    raw_server_rows, raw_errors = audit_server(root / "server_raw")
    component_medians = aggregate_server(component_server_rows)
    raw_medians = aggregate_server(raw_server_rows)
    selected_server_rows = [
        {
            "implementation": "baseline-static",
            **next(row for row in raw_medians if row["variant"] == "eval-s1"),
        },
        {
            "implementation": "ours-component-direct",
            **next(row for row in component_medians if row["variant"] == "eval-u3"),
        },
        {
            "implementation": "ours-raw-slot",
            **next(row for row in raw_medians if row["variant"] == "eval-u3"),
        },
    ]
    errors = component_errors + raw_errors
    payload = {
        "validation": {"passed": not errors, "errors": errors},
        "component_server_runs": component_server_rows,
        "raw_server_runs": raw_server_rows,
        "server_comparison": selected_server_rows,
        "mamba_microbenchmark": summarize_mamba(
            root / "mamba/raw-comparison/results.json"
        ),
        "kv_microbenchmark": summarize_kv(root / "kv/final/results.json"),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
