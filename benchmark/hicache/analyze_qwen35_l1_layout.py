#!/usr/bin/env python3
"""Audit and summarize the Qwen3.5 L1 layout-only evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_SHA = "743cae224c5bc28687457558a074736776350392"
VARIANTS = ("l1-lf", "l1-pf-static", "l1-pf-unified")
VARIANT_LABELS = {
    "l1-lf": "LF static allocator",
    "l1-pf-static": "PF static allocator",
    "l1-pf-unified": "PF unified allocator",
}
WORKLOADS = ("short-3k", "middle-10k", "long-50k")
GRAPH_MODES = ("enabled", "disabled")
PARITY_RE = re.compile(
    r"l1-parity-r(?P<repetition>\d+)-0\.8b-p(?P<page>\d+)-"
    r"cg-(?P<graph>enabled|disabled)"
)
RESIDENT_RE = re.compile(
    r"l1-resident-r(?P<repetition>\d+)-0\.8b-"
    r"(?P<workload>short-3k|middle-10k|long-50k)-p(?P<page>\d+)-"
    r"cg-(?P<graph>enabled|disabled)"
)
PRESSURE_RE = re.compile(
    r"l1-pressure-r(?P<repetition>\d+)-0\.8b-"
    r"(?P<workload>short-3k|middle-10k|long-50k)-p(?P<page>\d+)"
)
PROFILE_RE = re.compile(
    r"l1-profile-r(?P<repetition>\d+)-0\.8b-middle-10k-p(?P<page>\d+)"
)
NO_HICACHE_METRICS = (
    "hicache_backup_tokens_total",
    "load_back_tokens_total",
    "hicache_dropped_tokens_total",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Iterable[float]) -> float:
    return statistics.fmean(values)


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def latest_results(
    root: Path, pattern: re.Pattern[str]
) -> list[tuple[re.Match[str], str, dict[str, Any], dict[str, Any], Path]]:
    selected: dict[
        tuple[str, ...],
        tuple[re.Match[str], str, dict[str, Any], dict[str, Any], Path],
    ] = {}
    for path in sorted(root.glob("*/l1-*/*/result.json")):
        match = pattern.fullmatch(path.parents[2].name)
        if match is None:
            continue
        manifest_path = path.parent / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = load_json(manifest_path)
        result = load_json(path)
        variant = str(result["variant"])
        identity = (*match.groups(), variant)
        if manifest.get("status") == "completed":
            selected[identity] = (match, variant, result, manifest, path)
    return list(selected.values())


def validate_configuration(
    *, variant: str, result: dict[str, Any], manifest: dict[str, Any], label: str
) -> list[str]:
    errors: list[str] = []
    info = result["server_info"]
    expected_page_major = variant != "l1-lf"
    expected_unified = variant == "l1-pf-unified"
    if manifest.get("variant_definition", {}).get("sha") != SERVER_SHA:
        errors.append(f"wrong server SHA: {label}")
    if bool(info["enable_page_major_kv_layout"]) != expected_page_major:
        errors.append(f"wrong page-major setting: {label}")
    if bool(info["enable_unified_memory"]) != expected_unified:
        errors.append(f"wrong unified-memory setting: {label}")
    if info["enable_hierarchical_cache"]:
        errors.append(f"HiCache enabled in L1-only run: {label}")
    for name in ("attention_backend", "linear_attn_backend", "mamba_backend"):
        if info[name] != "triton":
            errors.append(f"{name} is not Triton: {label}")
    if int(info["max_total_num_tokens"]) != 120000:
        errors.append(f"wrong L1 token capacity: {label}")
    metrics = result.get("total_metric_delta", {})
    for name in NO_HICACHE_METRICS:
        if float(metrics.get("sglang:" + name, 0.0)) != 0.0:
            errors.append(f"unexpected {name}: {label}")
    if not result.get("validation", {}).get("passed"):
        errors.append(f"client validation failed: {label}")
    if result.get("summary", {}).get("failed", 0):
        errors.append(f"request failed: {label}")
    return errors


def performance_rows(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    specs = (
        ("resident", RESIDENT_RE, True),
        ("pressure", PRESSURE_RE, False),
    )
    for stage, pattern, radix_disabled in specs:
        for match, variant, result, manifest, path in latest_results(root, pattern):
            run_name = path.parents[2].name
            label = f"{run_name}/{variant}"
            errors.extend(
                validate_configuration(
                    variant=variant, result=result, manifest=manifest, label=label
                )
            )
            info = result["server_info"]
            if bool(info["disable_radix_cache"]) != radix_disabled:
                errors.append(f"wrong radix-cache mode: {label}")
            metrics = result["total_metric_delta"]
            evicted = float(metrics.get("sglang:evicted_tokens_total", 0.0))
            if stage == "pressure" and evicted <= 0:
                errors.append(f"pressure run did not evict: {label}")
            summary = result["summary"]
            graph_mode = match.groupdict().get("graph") or "enabled"
            rows.append(
                {
                    "stage": stage,
                    "repetition": int(match["repetition"]),
                    "page_size": int(match["page"]),
                    "workload": match["workload"],
                    "cuda_graph_mode": graph_mode,
                    "variant": variant,
                    "configuration": VARIANT_LABELS[variant],
                    "total_tok_s": float(summary["total_token_throughput"]),
                    "input_tok_s": float(summary["input_token_throughput"]),
                    "request_s": float(summary["request_throughput"]),
                    "ttft_p50_ms": float(summary["ttft_ms"]["p50"]),
                    "ttft_p95_ms": float(summary["ttft_ms"]["p95"]),
                    "tpot_p50_ms": float(summary["tpot_ms"]["p50"]),
                    "tpot_p95_ms": float(summary["tpot_ms"]["p95"]),
                    "duration_s": float(summary["duration_s"]),
                    "evicted_tokens": evicted,
                    "cached_tokens_device": int(summary["cached_tokens_device"]),
                    "result_path": str(path.relative_to(REPO_ROOT)),
                }
            )

    grouped: dict[tuple[str, int, str, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[
            (
                str(row["stage"]),
                int(row["page_size"]),
                str(row["workload"]),
                str(row["cuda_graph_mode"]),
                str(row["variant"]),
            )
        ].append(row)
    aggregates: list[dict[str, Any]] = []
    for (stage, page, workload, graph, variant), group in sorted(grouped.items()):
        throughput = [float(row["total_tok_s"]) for row in group]
        aggregates.append(
            {
                "stage": stage,
                "page_size": page,
                "workload": workload,
                "cuda_graph_mode": graph,
                "variant": variant,
                "configuration": VARIANT_LABELS[variant],
                "runs": len(group),
                "total_tok_s_mean": mean(throughput),
                "total_tok_s_std": sample_std(throughput),
                "total_tok_s_cv_percent": 100
                * sample_std(throughput)
                / mean(throughput),
                "ttft_p50_ms_mean": mean(float(row["ttft_p50_ms"]) for row in group),
                "tpot_p50_ms_mean": mean(float(row["tpot_p50_ms"]) for row in group),
                "evicted_tokens_mean": mean(
                    float(row["evicted_tokens"]) for row in group
                ),
            }
        )
    return rows, aggregates, errors


def paired_ratios(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    indexed = {
        (
            row["stage"],
            row["repetition"],
            row["page_size"],
            row["workload"],
            row["cuda_graph_mode"],
            row["variant"],
        ): row
        for row in rows
    }
    identities = sorted(
        {
            (
                str(row["stage"]),
                int(row["repetition"]),
                int(row["page_size"]),
                str(row["workload"]),
                str(row["cuda_graph_mode"]),
            )
            for row in rows
        }
    )
    paired: list[dict[str, Any]] = []
    errors: list[str] = []
    for stage, repetition, page, workload, graph in identities:
        key = (stage, repetition, page, workload, graph)
        variants = {variant: indexed.get((*key, variant)) for variant in VARIANTS}
        if any(value is None for value in variants.values()):
            errors.append(f"missing paired variant: {key}")
            continue
        lf = float(variants["l1-lf"]["total_tok_s"])
        pf_static = float(variants["l1-pf-static"]["total_tok_s"])
        pf_unified = float(variants["l1-pf-unified"]["total_tok_s"])
        paired.append(
            {
                "stage": stage,
                "repetition": repetition,
                "page_size": page,
                "workload": workload,
                "cuda_graph_mode": graph,
                "lf_total_tok_s": lf,
                "pf_static_total_tok_s": pf_static,
                "pf_unified_total_tok_s": pf_unified,
                "layout_only_pf_static_over_lf": pf_static / lf,
                "unified_over_pf_static": pf_unified / pf_static,
                "pf_unified_over_lf": pf_unified / lf,
            }
        )

    grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        grouped[
            (
                str(row["stage"]),
                int(row["page_size"]),
                str(row["workload"]),
                str(row["cuda_graph_mode"]),
            )
        ].append(row)
    summaries: list[dict[str, Any]] = []
    ratio_names = (
        "layout_only_pf_static_over_lf",
        "unified_over_pf_static",
        "pf_unified_over_lf",
    )
    for (stage, page, workload, graph), group in sorted(grouped.items()):
        output: dict[str, Any] = {
            "stage": stage,
            "page_size": page,
            "workload": workload,
            "cuda_graph_mode": graph,
            "pairs": len(group),
        }
        for name in ratio_names:
            values = [float(row[name]) for row in group]
            output[name + "_mean"] = mean(values)
            output[name + "_std"] = sample_std(values)
        summaries.append(output)
    return paired, summaries, errors


def parity_rows(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    fingerprints: dict[tuple[int, str, str], tuple[Any, Any]] = {}
    rows_by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    for match, variant, result, manifest, path in latest_results(root, PARITY_RE):
        run_name = path.parents[2].name
        label = f"{run_name}/{variant}"
        errors.extend(
            validate_configuration(
                variant=variant, result=result, manifest=manifest, label=label
            )
        )
        promotions = sorted(
            result.get("promotions", []), key=lambda item: item["index"]
        )
        comparisons = [item["comparison"] for item in promotions]
        if result.get("restored"):
            comparisons.append(result["restored"]["comparison"])
        fingerprint = (
            tuple(
                (int(item["index"]), tuple(item["result"]["output_ids"]))
                for item in promotions
            ),
            tuple(
                (int(item["index"]), tuple(item["result"]["logprob_token_ids"]))
                for item in promotions
            ),
        )
        key = (int(match["page"]), match["graph"], variant)
        fingerprints[key] = fingerprint
        row = {
            "repetition": int(match["repetition"]),
            "page_size": int(match["page"]),
            "cuda_graph_mode": match["graph"],
            "variant": variant,
            "configuration": VARIANT_LABELS[variant],
            "comparisons": len(comparisons),
            "output_ids_equal": all(item["output_ids_equal"] for item in comparisons),
            "logprob_token_ids_equal": all(
                item["logprob_token_ids_equal"] for item in comparisons
            ),
            "cross_layout_output_ids_equal": None,
            "cross_layout_logprob_token_ids_equal": None,
            "cross_graph_output_ids_equal": None,
            "cross_graph_logprob_token_ids_equal": None,
            "max_abs_logprob_diff": max(
                (float(item["max_abs_logprob_diff"]) for item in comparisons),
                default=0.0,
            ),
            "result_path": str(path.relative_to(REPO_ROOT)),
        }
        rows.append(row)
        rows_by_key[key] = row
        if not row["output_ids_equal"] or not row["logprob_token_ids_equal"]:
            errors.append(f"within-run parity failed: {label}")

    for page in sorted({key[0] for key in fingerprints}):
        for graph in GRAPH_MODES:
            canonical = fingerprints.get((page, graph, "l1-lf"))
            if canonical is None:
                errors.append(f"missing LF parity reference: p{page}/{graph}")
                continue
            for variant in VARIANTS:
                key = (page, graph, variant)
                observed = fingerprints.get(key)
                if observed is None:
                    errors.append(f"missing parity variant: p{page}/{graph}/{variant}")
                    continue
                row = rows_by_key[key]
                row["cross_layout_output_ids_equal"] = observed[0] == canonical[0]
                row["cross_layout_logprob_token_ids_equal"] = (
                    observed[1] == canonical[1]
                )
                if observed != canonical:
                    errors.append(
                        f"cross-layout parity failed: p{page}/{graph}/{variant}"
                    )
        for variant in VARIANTS:
            enabled = fingerprints.get((page, "enabled", variant))
            disabled = fingerprints.get((page, "disabled", variant))
            if enabled is None or disabled is None:
                continue
            for graph in GRAPH_MODES:
                row = rows_by_key[(page, graph, variant)]
                row["cross_graph_output_ids_equal"] = enabled[0] == disabled[0]
                row["cross_graph_logprob_token_ids_equal"] = enabled[1] == disabled[1]
            if enabled != disabled:
                errors.append(f"cross-graph parity failed: p{page}/{variant}")
    return rows, errors


def profile_rows(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    operations: list[dict[str, Any]] = []
    overviews: list[dict[str, Any]] = []
    errors: list[str] = []
    for match, variant, result, manifest, path in latest_results(root, PROFILE_RE):
        run_name = path.parents[2].name
        label = f"{run_name}/{variant}"
        errors.extend(
            validate_configuration(
                variant=variant, result=result, manifest=manifest, label=label
            )
        )
        profiles = result.get("memory_breakdown_profiles", [])
        if not profiles:
            errors.append(f"missing memory profile: {label}")
            continue
        combined: dict[tuple[str, str, str], dict[str, float]] = defaultdict(
            lambda: {"calls": 0.0, "rows": 0.0, "bytes": 0.0, "cpu_time_ns": 0.0}
        )
        for entry in profiles:
            for item in entry["profile"].get("metrics", []):
                key = (item["category"], item["operation"], item["pool"])
                for field in combined[key]:
                    combined[key][field] += float(item.get(field, 0.0))
        for (category, operation, pool), totals in sorted(combined.items()):
            operations.append(
                {
                    "repetition": int(match["repetition"]),
                    "page_size": int(match["page"]),
                    "variant": variant,
                    "configuration": VARIANT_LABELS[variant],
                    "category": category,
                    "operation": operation,
                    "pool": pool,
                    "calls": int(totals["calls"]),
                    "rows": int(totals["rows"]),
                    "bytes": int(totals["bytes"]),
                    "cpu_time_ms": totals["cpu_time_ns"] / 1e6,
                }
            )

        def category_seconds(name: str) -> float:
            return (
                sum(
                    totals["cpu_time_ns"]
                    for (category, _, _), totals in combined.items()
                    if category == name
                )
                / 1e9
            )

        metrics = result["measured_metric_delta"]
        overviews.append(
            {
                "repetition": int(match["repetition"]),
                "page_size": int(match["page"]),
                "variant": variant,
                "configuration": VARIANT_LABELS[variant],
                "total_tok_s": float(result["summary"]["total_token_throughput"]),
                "forward_s": float(
                    metrics.get("sglang:forward_execution_seconds_total", 0.0)
                ),
                "eviction_s": float(
                    metrics.get("sglang:eviction_duration_seconds_sum", 0.0)
                ),
                "allocator_cpu_s": category_seconds("allocator"),
                "compaction_cpu_s": category_seconds("compaction"),
                "translation_cpu_s": category_seconds("translation"),
            }
        )
    return operations, overviews, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPO_ROOT / "artifacts/qwen35_l1_layout_743cae2",
    )
    parser.add_argument("--expected-repetitions", type=int, default=3)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    output = root / "summary"

    runs, aggregates, performance_errors = performance_rows(root)
    pairs, pair_summaries, pair_errors = paired_ratios(runs)
    parity, parity_errors = parity_rows(root)
    operations, profiles, profile_errors = profile_rows(root)
    errors = performance_errors + pair_errors + parity_errors + profile_errors

    expected_resident = args.expected_repetitions * 3 * 3 * 2 * len(VARIANTS)
    expected_pressure = args.expected_repetitions * 3 * 3 * len(VARIANTS)
    expected_profiles = args.expected_repetitions * 3 * len(VARIANTS)
    expected_parity = 3 * 2 * len(VARIANTS)
    actual_resident = sum(row["stage"] == "resident" for row in runs)
    actual_pressure = sum(row["stage"] == "pressure" for row in runs)
    for label, actual, expected in (
        ("resident", actual_resident, expected_resident),
        ("pressure", actual_pressure, expected_pressure),
        ("profile", len(profiles), expected_profiles),
        ("parity", len(parity), expected_parity),
    ):
        if actual != expected:
            errors.append(f"expected {expected} {label} runs, got {actual}")

    write_csv(output / "run_summary.csv", runs)
    write_csv(output / "aggregate_summary.csv", aggregates)
    write_csv(output / "paired_runs.csv", pairs)
    write_csv(output / "paired_summary.csv", pair_summaries)
    write_csv(output / "parity_summary.csv", parity)
    write_csv(output / "profile_operations.csv", operations)
    write_csv(output / "profile_summary.csv", profiles)
    audit = {
        "schema_version": 1,
        "server_sha": SERVER_SHA,
        "resident_runs": actual_resident,
        "expected_resident_runs": expected_resident,
        "pressure_runs": actual_pressure,
        "expected_pressure_runs": expected_pressure,
        "profile_runs": len(profiles),
        "expected_profile_runs": expected_profiles,
        "parity_runs": len(parity),
        "expected_parity_runs": expected_parity,
        "passed": not errors,
        "errors": errors,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    print(json.dumps(audit, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
