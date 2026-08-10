#!/usr/bin/env python3
"""Summarize the final Qwen3.5 unified-memory + HiCache matrix."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

CLEAN_RE = re.compile(
    r"clean-r(?P<repeat>\d+)-(?P<model>0\.8b|4b)-"
    r"(?P<workload>short-3k|middle-10k|long-50k)-"
    r"p(?P<page>\d+)-(?P<policy>none|write_back|write_through)"
    r"(?:-c(?P<concurrency>\d+))?"
)
PROFILE_RE = re.compile(
    r"profile-r(?P<repeat>\d+)-(?P<model>0\.8b|4b)-middle-10k-"
    r"p(?P<page>\d+)-(?P<policy>none|write_back|write_through)"
)
PARITY_RE = re.compile(
    r"parity-r(?P<repeat>\d+)-(?P<model>0\.8b|4b)-"
    r"p(?P<page>\d+)-(?P<policy>none|write_back|write_through)"
)
VARIANT_LABELS = {
    "eval-s0": "S0 static only",
    "eval-s1": "S1 static + HiCache",
    "eval-u0": "U0 unified only",
    "eval-u2": "U2 unified + split L2",
    "eval-u3": "U3 unified + typed L2",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Iterable[float]) -> float:
    return statistics.fmean(values)


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def nested(row: dict[str, Any], *keys: str) -> float:
    value: Any = row
    for key in keys:
        value = value[key]
    return float(value)


def latest_valid_results(
    root: Path, pattern: re.Pattern[str]
) -> list[tuple[re.Match[str], str, dict[str, Any], Path]]:
    latest: dict[tuple[str, ...], tuple[re.Match[str], str, dict[str, Any], Path]] = {}
    for path in sorted(root.glob("*/eval-*/*/result.json")):
        match = pattern.fullmatch(path.parents[2].name)
        if match is None:
            continue
        result = read_json(path)
        variant = str(result["variant"])
        identity = (*match.groups(), variant)
        latest[identity] = (match, variant, result, path)
    return list(latest.values())


def clean_summary(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    grouped: dict[tuple[str, int, str, str, int, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    errors: list[str] = []
    for match, variant, result, path in latest_valid_results(root, CLEAN_RE):
        validation = result.get("validation", {})
        if not validation.get("passed") or result["summary"].get("failed") != 0:
            errors.append(str(path))
            continue
        grouped[
            (
                match["model"],
                int(match["page"]),
                match["policy"],
                match["workload"],
                int(
                    match["concurrency"]
                    or (4 if match["workload"] == "long-50k" else 8)
                ),
                variant,
            )
        ].append(result)

    rows: list[dict[str, Any]] = []
    for (
        model,
        page,
        policy,
        workload,
        concurrency,
        variant,
    ), results in sorted(grouped.items()):
        summaries = [item["summary"] for item in results]
        metrics = [item["total_metric_delta"] for item in results]

        def values(*keys: str) -> list[float]:
            return [nested(item, *keys) for item in summaries]

        def metric(name: str) -> list[float]:
            return [float(item.get("sglang:" + name, 0.0)) for item in metrics]

        backup_bytes = metric("hicache_backup_bytes_total")
        backup_s = metric("hicache_backup_duration_seconds_sum")
        load_bytes = metric("load_back_bytes_total")
        load_s = metric("load_back_duration_seconds_sum")
        throughput = values("total_token_throughput")
        rows.append(
            {
                "model": model,
                "page_size": page,
                "policy": policy,
                "workload": workload,
                "concurrency": concurrency,
                "variant": variant,
                "configuration": VARIANT_LABELS[variant],
                "runs": len(results),
                "total_tok_s_mean": mean(throughput),
                "total_tok_s_std": sample_std(throughput),
                "input_tok_s_mean": mean(values("input_token_throughput")),
                "request_s_mean": mean(values("request_throughput")),
                "ttft_p50_ms_mean": mean(values("ttft_ms", "p50")),
                "ttft_p95_ms_mean": mean(values("ttft_ms", "p95")),
                "tpot_p50_ms_mean": mean(values("tpot_ms", "p50")),
                "tpot_p95_ms_mean": mean(values("tpot_ms", "p95")),
                "duration_s_mean": mean(values("duration_s")),
                "evicted_tokens_mean": mean(metric("evicted_tokens_total")),
                "backup_gib_s_mean": mean(
                    [
                        size / seconds / 2**30 if seconds else 0.0
                        for size, seconds in zip(backup_bytes, backup_s)
                    ]
                ),
                "load_gib_s_mean": mean(
                    [
                        size / seconds / 2**30 if seconds else 0.0
                        for size, seconds in zip(load_bytes, load_s)
                    ]
                ),
                "dropped_tokens_max": max(metric("hicache_dropped_tokens_total")),
            }
        )
    return rows, errors


def parity_summary(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    canonical: dict[tuple[str, int, int], tuple[tuple[Any, ...], tuple[Any, ...]]] = {}
    observed: list[
        tuple[
            tuple[str, int, int],
            tuple[Any, ...],
            tuple[Any, ...],
            dict[str, Any],
            Path,
        ]
    ] = []
    for match, variant, result, path in latest_valid_results(root, PARITY_RE):
        comparisons = [
            item["comparison"]
            for item in result.get("promotions", [])
            if "comparison" in item
        ]
        restored = result.get("restored")
        if restored is not None:
            comparisons.append(restored["comparison"])
        passed = bool(result.get("validation", {}).get("passed"))
        output_equal = all(item["output_ids_equal"] for item in comparisons)
        token_equal = all(item["logprob_token_ids_equal"] for item in comparisons)
        max_diff = max(
            (float(item["max_abs_logprob_diff"]) for item in comparisons),
            default=0.0,
        )
        if not passed or not output_equal or not token_equal:
            errors.append(str(path))
        key = (match["model"], int(match["page"]), int(match["repeat"]))
        promotions = sorted(
            result.get("promotions", []), key=lambda item: item["index"]
        )
        output_fingerprint = tuple(
            (int(item["index"]), tuple(item["result"]["output_ids"]))
            for item in promotions
        )
        logprob_token_fingerprint = tuple(
            (int(item["index"]), tuple(item["result"]["logprob_token_ids"]))
            for item in promotions
        )
        row = {
            "model": match["model"],
            "page_size": int(match["page"]),
            "policy": match["policy"],
            "variant": variant,
            "configuration": VARIANT_LABELS[variant],
            "comparisons": len(comparisons),
            "validation_passed": passed,
            "output_ids_equal": output_equal,
            "logprob_token_ids_equal": token_equal,
            "cross_variant_output_ids_equal": None,
            "cross_variant_logprob_token_ids_equal": None,
            "max_abs_logprob_diff": max_diff,
        }
        rows.append(row)
        observed.append((key, output_fingerprint, logprob_token_fingerprint, row, path))
        if variant == "eval-s0" and match["policy"] == "none":
            canonical[key] = (output_fingerprint, logprob_token_fingerprint)

    # Older artifact sets predate S0 and retain their original within-run parity
    # behavior. New four-way matrices include S0 and use it as the exact token-ID
    # reference for every policy and memory configuration at the same page size.
    if canonical:
        for key, output_fingerprint, token_fingerprint, row, path in observed:
            reference = canonical.get(key)
            if reference is None:
                errors.append(f"missing S0 parity reference for {path}")
                row["cross_variant_output_ids_equal"] = False
                row["cross_variant_logprob_token_ids_equal"] = False
                continue
            row["cross_variant_output_ids_equal"] = output_fingerprint == reference[0]
            row["cross_variant_logprob_token_ids_equal"] = (
                token_fingerprint == reference[1]
            )
            if not row["cross_variant_output_ids_equal"]:
                errors.append(f"cross-variant output IDs differ: {path}")
            if not row["cross_variant_logprob_token_ids_equal"]:
                errors.append(f"cross-variant logprob token IDs differ: {path}")
    return sorted(rows, key=lambda item: tuple(map(str, item.values()))), errors


def profile_summary(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    overviews: list[dict[str, Any]] = []
    errors: list[str] = []
    for match, variant, result, path in latest_valid_results(root, PROFILE_RE):
        profiles = result.get("memory_breakdown_profiles", [])
        if not result.get("validation", {}).get("passed") or not profiles:
            errors.append(str(path))
            continue
        combined: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(
            lambda: {"calls": 0.0, "rows": 0.0, "bytes": 0.0, "cpu_time_ns": 0.0}
        )
        for profile_entry in profiles:
            profile = profile_entry["profile"]
            for metric_kind, field_name in (
                ("cuda_interval", "cuda_metrics"),
                ("cpu_control", "metrics"),
            ):
                for item in profile.get(field_name, []):
                    key = (
                        metric_kind,
                        item["category"],
                        item["operation"],
                        item["pool"],
                    )
                    for field in combined[key]:
                        combined[key][field] += float(item.get(field, 0.0))
        for (
            metric_kind,
            category,
            operation,
            pool,
        ), totals in sorted(combined.items()):
            rows.append(
                {
                    "model": match["model"],
                    "page_size": int(match["page"]),
                    "policy": match["policy"],
                    "variant": variant,
                    "configuration": VARIANT_LABELS[variant],
                    "metric_kind": metric_kind,
                    "category": category,
                    "operation": operation,
                    "pool": pool,
                    "calls": int(totals["calls"]),
                    "rows": int(totals["rows"]),
                    "bytes": int(totals["bytes"]),
                    "cpu_time_ms": totals["cpu_time_ns"] / 1e6,
                    "gib_s": (
                        totals["bytes"] / totals["cpu_time_ns"] * 1e9 / 2**30
                        if totals["cpu_time_ns"]
                        else 0.0
                    ),
                }
            )

        def combined_totals(
            metric_kind: str,
            *,
            category: str | None = None,
            operation: str | None = None,
            pool: str | None = None,
        ) -> dict[str, float]:
            selected = [
                totals
                for (
                    kind,
                    item_category,
                    item_operation,
                    item_pool,
                ), totals in combined.items()
                if kind == metric_kind
                and (category is None or item_category == category)
                and (operation is None or item_operation == operation)
                and (pool is None or item_pool == pool)
            ]
            return {
                field: sum(item[field] for item in selected)
                for field in ("calls", "rows", "bytes", "cpu_time_ns")
            }

        def rate(operation: str, pool: str) -> float:
            totals = combined_totals("cuda_interval", operation=operation, pool=pool)
            return (
                totals["bytes"] / totals["cpu_time_ns"] * 1e9 / 2**30
                if totals["cpu_time_ns"]
                else 0.0
            )

        compaction = {field: 0.0 for field in ("calls", "rows", "bytes", "cpu_time_ns")}
        for operation in ("opportunistic_flush", "urgent_flush"):
            totals = combined_totals(
                "cpu_control", category="compaction", operation=operation
            )
            for field in compaction:
                compaction[field] += totals[field]
        allocator = combined_totals("cpu_control", category="allocator")
        transfer_control = combined_totals(
            "cpu_control", category="hicache_transfer_control"
        )
        translation = combined_totals("cpu_control", category="translation")
        row_fence = combined_totals("cpu_control", category="row_fence")
        metrics = result["measured_metric_delta"]
        overviews.append(
            {
                "model": match["model"],
                "page_size": int(match["page"]),
                "policy": match["policy"],
                "variant": variant,
                "configuration": VARIANT_LABELS[variant],
                "total_tok_s": float(result["summary"]["total_token_throughput"]),
                "forward_s": float(
                    metrics.get("sglang:forward_execution_seconds_total", 0.0)
                ),
                "eviction_s": float(
                    metrics.get("sglang:eviction_duration_seconds_sum", 0.0)
                ),
                "backup_s": float(
                    metrics.get("sglang:hicache_backup_duration_seconds_sum", 0.0)
                ),
                "loadback_s": float(
                    metrics.get("sglang:load_back_duration_seconds_sum", 0.0)
                ),
                "allocator_cpu_s": allocator["cpu_time_ns"] / 1e9,
                "compaction_cpu_s": compaction["cpu_time_ns"] / 1e9,
                "transfer_control_cpu_s": transfer_control["cpu_time_ns"] / 1e9,
                "translation_cpu_s": translation["cpu_time_ns"] / 1e9,
                "row_fence_cpu_s": row_fence["cpu_time_ns"] / 1e9,
                "row_fence_calls": int(row_fence["calls"]),
                "d2h_total_gib_s": rate("d2h_total", "all"),
                "h2d_total_gib_s": rate("h2d_total", "all"),
                "kv_d2h_gib_s": rate("d2h_all_layers", "kv"),
                "mamba_d2h_gib_s": rate("d2h_all_layers", "mamba"),
                "kv_h2d_gib_s": rate("h2d_per_layer", "kv"),
                "mamba_h2d_gib_s": rate("h2d_per_layer", "mamba"),
            }
        )
    return rows, overviews, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/qwen35_unified_hicache_4way"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/qwen35_unified_hicache_4way/summary"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clean, clean_errors = clean_summary(args.artifact_root)
    parity, parity_errors = parity_summary(args.artifact_root)
    profiles, profile_overviews, profile_errors = profile_summary(args.artifact_root)
    write_csv(args.output_dir / "clean_summary.csv", clean)
    write_csv(args.output_dir / "parity_summary.csv", parity)
    write_csv(args.output_dir / "profile_breakdown.csv", profiles)
    write_csv(args.output_dir / "profile_summary.csv", profile_overviews)
    errors = clean_errors + parity_errors + profile_errors
    print(
        f"clean_groups={len(clean)} parity_runs={len(parity)} "
        f"profile_operations={len(profiles)} errors={len(errors)}"
    )
    for path in errors:
        print(f"invalid: {path}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
